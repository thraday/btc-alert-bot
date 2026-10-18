import os
import sys
import json
import warnings
from datetime import datetime, timezone

warnings.filterwarnings("ignore", message=".*OpenSSL.*")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ.setdefault("MPLCONFIGDIR", os.path.join(BASE_DIR, ".mplconfig"))

import ccxt
import pandas as pd
try:
    import pandas_ta as ta
except ImportError:
    import pandas_ta_classic as ta
import mplfinance as mpf
import requests
from dotenv import load_dotenv

load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")
SYMBOL = "BTC/USDT:USDT"
LIMIT = 100
STATE_FILE = os.path.join(BASE_DIR, ".alert_state.json")

# 멀티 타임프레임 설정: (직전 20캔들 박스권 최대 변동폭 %, 타임프레임 길이(분), 새로 마감된 캔들 유효 시간(분))
TIMEFRAMES = {
    "5m":  {"max_box_pct": 1.2, "tf_minutes": 5,   "fresh_window_min": 15},
    "15m": {"max_box_pct": 1.8, "tf_minutes": 15,  "fresh_window_min": 20},
    "1h":  {"max_box_pct": 3.0, "tf_minutes": 60,  "fresh_window_min": 20},
    "4h":  {"max_box_pct": 5.0, "tf_minutes": 240, "fresh_window_min": 20},
}


def load_alert_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_alert_state(state):
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[경고] 상태 파일 저장 실패: {e}")


def init_exchange():
    return ccxt.bitget({'options': {'defaultType': 'swap'}})


def fetch_common_info(exchange):
    """현재가 및 펀딩비 조회"""
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = float(ticker['last'])

    funding_rate_pct = 0.0
    try:
        funding_info = exchange.fetch_funding_rate(SYMBOL)
        if funding_info and funding_info.get('fundingRate') is not None:
            funding_rate_pct = float(funding_info['fundingRate']) * 100.0
    except Exception as e:
        print(f"[경고] 펀딩비 조회 실패: {e}")

    return current_price, funding_rate_pct


def fetch_tf_dataframe(exchange, tf):
    """타임프레임별 OHLCV 데이터 조회 및 VWAP / ATR(14) 계산"""
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=tf, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['UTC_Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Date'] = df['UTC_Date'] + pd.Timedelta(hours=9)  # KST
    df.set_index('Date', inplace=True)

    # 당일 기준 VWAP (4h 등 상위 프레임에서도 당일 또는 누적 VWAP 표시되도록 처리)
    vwap_series = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    if vwap_series is None or vwap_series.isna().all():
        tp = (df['High'] + df['Low'] + df['Close']) / 3.0
        tp_vol = tp * df['Volume']
        day_group = df.index.date
        vwap_series = tp_vol.groupby(day_group).cumsum() / df['Volume'].groupby(day_group).cumsum()
    df['VWAP'] = vwap_series

    # ATR(14) 계산
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    return df


def analyze_breakout_at_index(df, target_pos, max_box_pct):
    """
    특정 캔들 위치(target_pos: -2 또는 -1)를 돌파 캔들로 두고,
    그 직전 20개 캔들(돌파 캔들 제외)의 박스권 응축 여부, 거래량 폭발, 장대캔들 및 상/하향 돌파 판별
    """
    if target_pos == -1:
        box_df = df.iloc[-21:-1]
        atr_prev = float(df['ATR_14'].iloc[-2])
    else:
        box_df = df.iloc[-22:-2]
        atr_prev = float(df['ATR_14'].iloc[-3])

    candle = df.iloc[target_pos]

    # 1) 박스권 응축 (돌파 직전 20개 캔들)
    box_high = float(box_df['High'].max())
    box_low = float(box_df['Low'].min())
    box_pct = (box_high - box_low) / box_low * 100.0 if box_low > 0 else 999.0
    is_squeeze = box_pct <= max_box_pct

    # 2) 거래량 폭발 (직전 20개 평균 거래량 대비 2.5배 = 250% 이상)
    vol_sma20 = float(box_df['Volume'].mean())
    candle_vol = float(candle['Volume'])
    vol_ratio = (candle_vol / vol_sma20) if vol_sma20 > 0 else 0.0
    vol_pct = vol_ratio * 100.0
    is_vol_spike = vol_ratio >= 2.5

    # 3) 장대 캔들 (몸통 >= 직전 ATR(14) * 1.8 이고, 몸통/전체길이 >= 65%)
    open_p = float(candle['Open'])
    close_p = float(candle['Close'])
    high_p = float(candle['High'])
    low_p = float(candle['Low'])

    body_size = abs(close_p - open_p)
    full_range = high_p - low_p
    body_ratio_pct = (body_size / full_range * 100.0) if full_range > 0 else 0.0
    atr_ratio = (body_size / atr_prev) if atr_prev > 0 else 0.0

    is_marubozu = (atr_ratio >= 1.8) and (body_ratio_pct >= 65.0)

    # 4) 방향 및 박스권 돌파 여부
    is_bull_break = (close_p > open_p) and (close_p > box_high)
    is_bear_break = (close_p < open_p) and (close_p < box_low)

    signal = None
    if is_squeeze and is_vol_spike and is_marubozu:
        if is_bull_break:
            signal = "LONG"
        elif is_bear_break:
            signal = "SHORT"

    return {
        "target_pos": target_pos,
        "candle_ts": int(candle['timestamp']),
        "candle_time_kst": df.index[target_pos].strftime('%Y-%m-%d %H:%M'),
        "open": open_p,
        "high": high_p,
        "low": low_p,
        "close": close_p,
        "box_high": box_high,
        "box_low": box_low,
        "box_pct": box_pct,
        "max_box_pct": max_box_pct,
        "is_squeeze": is_squeeze,
        "vol_sma20": vol_sma20,
        "candle_vol": candle_vol,
        "vol_ratio": vol_ratio,
        "vol_pct": vol_pct,
        "is_vol_spike": is_vol_spike,
        "atr_prev": atr_prev,
        "body_size": body_size,
        "atr_ratio": atr_ratio,
        "body_ratio_pct": body_ratio_pct,
        "is_marubozu": is_marubozu,
        "is_bull_break": is_bull_break,
        "is_bear_break": is_bear_break,
        "signal": signal,
    }


def evaluate_timeframe(df, tf, tf_cfg, state):
    """
    직전 완성 캔들(iloc[-2]) 우선 검사 + 현재 진행 캔들(iloc[-1])이 마감 1분 전 이내일 때 검사.
    중복 알림 방지:
      1) 로컬/캐시 .alert_state.json 에 이미 기록된 캔들 타임스탬프면 스킵
      2) GitHub Actions(5분 주기) 무상태 환경을 위해, 완성 캔들(iloc[-2])이 마감된 지
         fresh_window_min(예: 1h/4h봉은 마감 후 20분 이내)이 지났으면 중복 발송하지 않음
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    tf_ms = tf_cfg["tf_minutes"] * 60 * 1000
    fresh_ms = tf_cfg["fresh_window_min"] * 60 * 1000

    # 기본 분석은 직전 마감 완성 캔들(iloc[-2])
    res_closed = analyze_breakout_at_index(df, -2, tf_cfg["max_box_pct"])
    closed_end_ms = res_closed["candle_ts"] + tf_ms
    closed_age_min = (now_ms - closed_end_ms) / 60000.0
    res_closed["candle_type"] = "완성봉(iloc[-2])"

    # 현재 캔들(iloc[-1]) 마감까지 남은 시간 계산
    res_curr = analyze_breakout_at_index(df, -1, tf_cfg["max_box_pct"])
    curr_end_ms = res_curr["candle_ts"] + tf_ms
    rem_min = (curr_end_ms - now_ms) / 60000.0
    res_curr["candle_type"] = f"마감임박봉(iloc[-1], 잔여 {max(0.0, rem_min):.1f}분)"

    chosen = res_closed
    already_sent = False

    # 1) 완성봉(iloc[-2])에서 시그널이 발생했고, 최근 마감된 신선한 캔들인 경우
    if res_closed["signal"] is not None:
        if (now_ms - closed_end_ms) <= fresh_ms and state.get(tf) != res_closed["candle_ts"]:
            chosen = res_closed
        else:
            already_sent = True
            chosen = res_closed

    # 2) 완성봉 시그널이 없을 때, 현재 캔들(iloc[-1])이 마감 1분 전 이내이면서 강하게 조건을 돌파한 경우
    elif 0 <= rem_min <= 1.0 and res_curr["signal"] is not None:
        if state.get(tf) != res_curr["candle_ts"]:
            chosen = res_curr
        else:
            already_sent = True
            chosen = res_curr

    chosen["already_sent"] = already_sent
    chosen["rem_min_curr"] = rem_min
    return chosen


def create_breakout_chart(df, tf, eval_res, current_price, filename=None):
    """최근 60개 캔들 + 거래량 바 + 박스권 상/하단 수평선(점선) + VWAP 차트 생성"""
    if filename is None:
        filename = f"btc_breakout_{tf}.png"

    plot_df = df.tail(60).copy()

    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        gridstyle='--',
        y_on_right=True
    )

    apds = [
        mpf.make_addplot(plot_df['VWAP'], color='#ffeb3b', width=1.5, label='VWAP'),
    ]

    hlines_vals = [eval_res['box_high'], eval_res['box_low']]
    hlines_colors = ['#00e676', '#ff5252']

    sig_text = eval_res['signal'] if eval_res['signal'] else "BRIEFING"
    title = (
        f"Bitget BTC/USDT [{tf} {sig_text}] | Price: ${current_price:,.1f} | "
        f"Box: {eval_res['box_pct']:.2f}% | Vol: {eval_res['vol_pct']:.0f}%"
    )

    mpf.plot(
        plot_df,
        type='candle',
        volume=True,
        addplot=apds,
        hlines=dict(hlines=hlines_vals, colors=hlines_colors, linestyle='--', linewidths=1.3),
        style=style,
        title=title,
        savefig=dict(fname=filename, dpi=150, bbox_inches='tight'),
        figsize=(12, 7)
    )
    return filename


def build_breakout_message(tf, eval_res, current_price, funding_rate_pct, force_mode=False):
    """요청된 포맷에 맞춘 텔레그램 돌파 알림 메시지 생성"""
    sig = eval_res["signal"]
    if sig is None:
        effective_dir = "LONG" if eval_res["close"] >= eval_res["open"] else "SHORT"
    else:
        effective_dir = sig

    if effective_dir == "LONG":
        dir_text = "🟢 장대양봉 상향 돌파 (LONG)"
    else:
        dir_text = "🔴 장대음봉 하향 돌파 (SHORT)"

    # 권장 손절가: 돌파 캔들의 시가 (현재가와 너무 가까울 경우 중심값 병기)
    stop_price = eval_res["open"]
    mid_price = (eval_res["open"] + eval_res["close"]) / 2.0
    stop_pct = abs(current_price - stop_price) / current_price * 100.0
    risk_50x = stop_pct * 50.0

    header_tag = f"🚀 <b>[BTC 돌파 타점 포착 | 타임프레임: {tf}]</b>"
    if force_mode and sig is None:
        header_tag = f"🚀 <b>[BTC 돌파 타점 감시 브리핑 (수동 테스트) | 타임프레임: {tf}]</b>"

    msg = (
        f"{header_tag}\n"
        f"• 방향: {dir_text}\n"
        f"• 돌파 강도: 거래량 평균 대비 <code>{eval_res['vol_pct']:.0f}%</code> 폭발 / 박스권 응축 폭 <code>{eval_res['box_pct']:.2f}%</code>\n"
        f"• 현재가(진입가): <code>${current_price:,.0f}</code>\n"
        f"• 박스권 상단: <code>${eval_res['box_high']:,.0f}</code> / 박스권 하단: <code>${eval_res['box_low']:,.0f}</code>\n"
        f"• 권장 손절가(Stop-Loss): 돌파 캔들 시가 <code>${stop_price:,.0f}</code> [중심값 <code>${mid_price:,.0f}</code>] "
        f"(손절폭 -{stop_pct:.2f}% / 50배 기준 리스크 -{risk_50x:.0f}%)\n"
        f"• 현재 펀딩비: <code>{funding_rate_pct:.4f}%</code>"
    )
    return msg


def send_telegram_photo(image_path, caption):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        payload = {
            "chat_id": CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML"
        }
        files = {"photo": photo}
        response = requests.post(url, data=payload, files=files, timeout=20)
        response.raise_for_status()
        return response.json()


def main():
    force_send = "--force" in sys.argv
    exchange = init_exchange()
    state = load_alert_state()

    current_price, funding_rate_pct = fetch_common_info(exchange)
    now_kst = (datetime.now(timezone.utc) + pd.Timedelta(hours=9)).strftime('%Y-%m-%d %H:%M:%S')

    print("=" * 92)
    print(f"📊 [Bitget BTC/USDT 멀티 타임프레임 박스권 거래량 돌파 감시] | 기준시각(KST): {now_kst}")
    print(f"   현재가: ${current_price:,.2f} | 현재 펀딩비: {funding_rate_pct:.4f}%")
    print("=" * 92)
    print(f"{'TF':<4} | {'박스변동폭(기준)':<17} | {'박스 상단 ~ 하단':<23} | {'거래량배수(기준>=2.5x)':<21} | {'장대캔들(ATR/몸통%)':<20} | {'판정'}")
    print("-" * 92)

    triggered_count = 0
    summary_rows = []
    last_df = None
    last_eval = None

    for tf, tf_cfg in TIMEFRAMES.items():
        df = fetch_tf_dataframe(exchange, tf)
        eval_res = evaluate_timeframe(df, tf, tf_cfg, state)

        box_str = f"{eval_res['box_pct']:.2f}% (<={tf_cfg['max_box_pct']:.1f}%)"
        box_mark = "O" if eval_res['is_squeeze'] else "X"
        range_str = f"${eval_res['box_high']:,.0f}~${eval_res['box_low']:,.0f}"
        vol_str = f"{eval_res['vol_ratio']:.2f}x ({eval_res['vol_pct']:.0f}%)"
        vol_mark = "O" if eval_res['is_vol_spike'] else "X"
        candle_str = f"{eval_res['atr_ratio']:.2f}x / {eval_res['body_ratio_pct']:.0f}%"
        candle_mark = "O" if eval_res['is_marubozu'] else "X"

        if eval_res['signal'] and not eval_res['already_sent']:
            status_str = f"🚨 {eval_res['signal']} 돌파 포착!"
        elif eval_res['signal'] and eval_res['already_sent']:
            status_str = f"⏭️ {eval_res['signal']} (기전송 스킵)"
        else:
            status_str = f"대기 (박:{box_mark} 볼:{vol_mark} 캔:{candle_mark})"

        print(f"{tf:<4} | {box_str:<17} | {range_str:<23} | {vol_str:<21} | {candle_str:<20} | {status_str}")

        summary_rows.append((tf, eval_res))
        if tf == "15m" or last_df is None:
            last_df = df
            last_eval = eval_res

        # 실제 조건 충족 및 미전송 타점일 때 알림 발송
        if eval_res['signal'] is not None and not eval_res['already_sent']:
            triggered_count += 1
            chart_path = create_breakout_chart(df, tf, eval_res, current_price)
            caption = build_breakout_message(tf, eval_res, current_price, funding_rate_pct, force_mode=False)
            res = send_telegram_photo(chart_path, caption)
            if res.get("ok"):
                print(f"   -> ✅ [{tf}] 텔레그램 돌파 타점 알림 전송 완료!")
                state[tf] = eval_res["candle_ts"]
                save_alert_state(state)

    print("=" * 92)

    # 수동 실행(--force) 시 조건이 하나도 안 맞았더라도 정상 작동 확인용 멀티 타임프레임 요약 브리핑 전송
    if force_send and triggered_count == 0 and last_df is not None and last_eval is not None:
        print("수동 테스트 모드(--force): 현재 돌파 타점 미충족 상태이나 4개 타임프레임 요약 브리핑을 전송합니다.")
        chart_path = create_breakout_chart(last_df, "15m", last_eval, current_price)
        caption = build_breakout_message("15m", last_eval, current_price, funding_rate_pct, force_mode=True)
        lines = ["\n\n📋 <b>[멀티 타임프레임 실시간 스캔 요약]</b>"]
        for tf, ev in summary_rows:
            sq = "🟢응축" if ev["is_squeeze"] else "⚪비응축"
            lines.append(
                f"• <b>{tf}</b>: 박스 <code>{ev['box_pct']:.2f}%</code>({sq}) | "
                f"거래량 <code>{ev['vol_ratio']:.2f}배({ev['vol_pct']:.0f}%)</code> | "
                f"몸통 <code>{ev['body_ratio_pct']:.0f}%</code>"
            )
        caption += "\n".join(lines)
        res = send_telegram_photo(chart_path, caption)
        if res.get("ok"):
            print("✅ 수동 브리핑 텔레그램 전송 완료!")


if __name__ == "__main__":
    main()
