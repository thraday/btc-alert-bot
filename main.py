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

# 손절가 여유 버퍼 (돌파 캔들 최저점 바로 아래 / 최고점 바로 위: 0.03%)
SL_BUFFER_PCT = 0.0003


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
    """타임프레임별 OHLCV 데이터 조회 및 VWAP / VWMA(20) / EMA(9) / ATR(14) 계산"""
    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=tf, limit=LIMIT)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    df['UTC_Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Date'] = df['UTC_Date'] + pd.Timedelta(hours=9)  # KST
    df.set_index('Date', inplace=True)

    # 당일 기준 VWAP 계산
    vwap_series = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    if vwap_series is None or vwap_series.isna().all():
        tp = (df['High'] + df['Low'] + df['Close']) / 3.0
        tp_vol = tp * df['Volume']
        day_group = df.index.date
        vwap_series = tp_vol.groupby(day_group).cumsum() / df['Volume'].groupby(day_group).cumsum()
    df['VWAP'] = vwap_series

    # 상위 추세 판별용 VWMA(20) & 단기 EMA(9) & ATR(14)
    df['VWMA_20'] = ta.vwma(df['Close'], df['Volume'], length=20)
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    return df


def analyze_htf_trend(df_1h, df_4h):
    """
    [1. 상위 봉(1h / 4h) 주추세 분석]
    - 각 상위 봉의 현재 종가 vs VWMA(20) 위치 및 VWMA(20)/단기 방향성을 종합해 주추세 산출
    """
    def get_single_tf_trend(df, label):
        last = df.iloc[-1]
        prev = df.iloc[-2]
        close_p = float(last['Close'])
        vwma20 = float(last['VWMA_20'])
        vwma20_prev = float(prev['VWMA_20'])
        ema9 = float(last['EMA_9'])

        above_vwma = close_p >= vwma20
        vwma_rising = vwma20 >= vwma20_prev
        short_bull = close_p >= ema9

        if above_vwma and (vwma_rising or short_bull):
            direction = "BULL"
            desc = f"{label} 상승추세 (현재가 &gt; VWMA20 ${vwma20:,.0f})"
        elif (not above_vwma) and ((not vwma_rising) or (not short_bull)):
            direction = "BEAR"
            desc = f"{label} 하락추세 (현재가 &lt; VWMA20 ${vwma20:,.0f})"
        else:
            direction = "BULL" if above_vwma else "BEAR"
            desc = f"{label} {'완만한 상승' if above_vwma else '완만한 하락'} (VWMA20 ${vwma20:,.0f})"

        return {
            "tf": label,
            "direction": direction,
            "close": close_p,
            "vwma20": vwma20,
            "vwma_rising": vwma_rising,
            "desc": desc,
        }

    t_1h = get_single_tf_trend(df_1h, "1h")
    t_4h = get_single_tf_trend(df_4h, "4h")

    return {"1h": t_1h, "4h": t_4h}


def check_htf_alignment(tf, signal_dir, htf_info):
    """
    하위 봉(5m, 15m) 및 중위 봉(1h) 돌파 시 상위 봉(1h/4h) 추세 동조 여부 검증
    - LONG 시그널: 1h 또는 4h 주추세가 상승(BULL)이어야 하며, 특히 1h 주추세와 정면 역행하면 휩소로 필터링
    - SHORT 시그널: 1h 또는 4h 주추세가 하락(BEAR)이어야 하며, 특히 1h 주추세와 정면 역행하면 휩소로 필터링
    """
    if signal_dir is None:
        return True, "시그널 없음"

    dir_1h = htf_info["1h"]["direction"]
    dir_4h = htf_info["4h"]["direction"]

    target_trend = "BULL" if signal_dir == "LONG" else "BEAR"
    kor_dir = "상승" if target_trend == "BULL" else "하락"

    if tf in ("5m", "15m"):
        # 5m/15m은 1h 주추세 동조 필수 (1h & 4h 모두 동조 시 초강력, 1h 동조 시 통과, 1h 역행 시 휩소 차단)
        if dir_1h == target_trend and dir_4h == target_trend:
            return True, f"✅ 상위봉 완벽 동조 (1h·4h 모두 {kor_dir} 추세)"
        elif dir_1h == target_trend:
            return True, f"✅ 1h 주추세 동조 (1h {kor_dir} / 4h {'상승' if dir_4h == 'BULL' else '하락'})"
        else:
            return False, f"❌ 상위봉 역추세 필터링됨 (1h {htf_info['1h']['desc']} 역행 - 휩소 고위험)"
    elif tf == "1h":
        if dir_4h == target_trend or dir_1h == target_trend:
            return True, f"✅ 상위봉 추세 동조 (1h {'상승' if dir_1h == 'BULL' else '하락'} / 4h {'상승' if dir_4h == 'BULL' else '하락'})"
        else:
            return False, f"❌ 4h 대추세 역행 필터링됨 ({htf_info['4h']['desc']})"
    else:  # 4h
        return True, f"✅ 4h 메이저 추세 돌파 ({htf_info['4h']['desc']})"


def detect_liquidity_sweep(df, box_high, box_low):
    """
    [2. 유동성 흡수(Liquidity Sweep / 가짜 돌파 휩소) 감지]
    - 직전 완성 캔들(iloc[-2]) 또는 진행 중인 캔들(iloc[-1])이
      박스권 상단(box_high) 또는 하단(box_low)을 고가/저가 꼬리로 찔렀으나,
      종가(또는 현재가)가 박스권 내부로 깊게 말아 들어와 복귀한 패턴 감지
    """
    warnings_list = []

    for pos, tag in [(-2, "직전 마감봉"), (-1, "현재 진행봉")]:
        c = df.iloc[pos]
        h, l, o, cl = float(c['High']), float(c['Low']), float(c['Open']), float(c['Close'])
        rng = h - l
        if rng <= 0:
            continue

        upper_wick = h - max(o, cl)
        lower_wick = min(o, cl) - l
        upper_wick_pct = upper_wick / rng * 100.0
        lower_wick_pct = lower_wick / rng * 100.0

        # 상단 유동성 스윕 (Upthrust / Bull Trap): 고가는 box_high 돌파했으나 종가는 box_high 아래로 복귀
        if h > box_high and cl <= box_high:
            penetration_pct = (h - box_high) / box_high * 100.0
            warnings_list.append(
                f"⚠️ [{tag} 상단 유동성 스윕(Bull Trap) 감지] 고가(${h:,.0f})가 박스 상단(${box_high:,.0f})을 "
                f"+{penetration_pct:.2f}% 찔렀으나 종가(${cl:,.0f})가 박스 내부로 복귀 (윗꼬리 {upper_wick_pct:.0f}%)"
            )

        # 하단 유동성 스윕 (Spring / Bear Trap): 저가는 box_low 이탈했으나 종가는 box_low 위로 복귀
        if l < box_low and cl >= box_low:
            penetration_pct = (box_low - l) / box_low * 100.0
            warnings_list.append(
                f"⚠️ [{tag} 하단 유동성 스윕(Bear Trap) 감지] 저가(${l:,.0f})가 박스 하단(${box_low:,.0f})을 "
                f"-{penetration_pct:.2f}% 찔렀으나 종가(${cl:,.0f})가 박스 내부로 복귀 (아랫꼬리 {lower_wick_pct:.0f}%)"
            )

    # 또한 돌파 캔들(iloc[-2])이 마감 돌파했더라도 현재 캔들(iloc[-1])이 즉시 박스 안으로 깊게 되돌아온 경우 경고
    c_prev = df.iloc[-2]
    c_curr = df.iloc[-1]
    if float(c_prev['Close']) > box_high and float(c_curr['Close']) < box_high:
        warnings_list.append(
            f"⚠️ [돌파 직후 박스 내부 재진입 경고] 직전 봉이 상단 돌파 마감했으나 현재가(${float(c_curr['Close']):,.0f})가 "
            f"다시 박스 상단(${box_high:,.0f}) 아래로 밀려 내려옴 (페이크 돌파 주의)"
        )
    elif float(c_prev['Close']) < box_low and float(c_curr['Close']) > box_low:
        warnings_list.append(
            f"⚠️ [이탈 직후 박스 내부 재진입 경고] 직전 봉이 하단 이탈 마감했으나 현재가(${float(c_curr['Close']):,.0f})가 "
            f"다시 박스 하단(${box_low:,.0f}) 위로 말아 올림 (페이크 이탈 주의)"
        )

    return warnings_list


def analyze_closed_breakout(df, tf, tf_cfg, htf_info):
    """
    [2 & 3. 종가(Close) 마감 기반 돌파 판정 + 손절가(돌파 캔들 Low/High 바깥) 계산]
    - 미완성 캔들의 꼬리 찌르기 휩소를 원천 배제하기 위해 반드시 '직전 마감 완성 캔들(iloc[-2])'의 종가(Close)만 돌파로 인정!
    - 그 직전 20개 캔들(iloc[-22:-2], 돌파 캔들 제외)을 박스권 기준 구간으로 설정.
    """
    max_box_pct = tf_cfg["max_box_pct"]
    box_df = df.iloc[-22:-2]
    atr_prev = float(df['ATR_14'].iloc[-3])
    candle = df.iloc[-2]  # 직전 마감 완성 캔들

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

    # 3) 장대 캔들 및 몸통 비율 (몸통 >= 직전 ATR(14) * 1.8 이고, 몸통/전체길이 >= 65%)
    open_p = float(candle['Open'])
    close_p = float(candle['Close'])
    high_p = float(candle['High'])
    low_p = float(candle['Low'])

    body_size = abs(close_p - open_p)
    full_range = high_p - low_p
    body_ratio_pct = (body_size / full_range * 100.0) if full_range > 0 else 0.0
    atr_ratio = (body_size / atr_prev) if atr_prev > 0 else 0.0
    is_marubozu = (atr_ratio >= 1.8) and (body_ratio_pct >= 65.0)

    # 4) 종가(Close) 기준 확실한 몸통 돌파 여부 (단순 High/Low 꼬리 돌파 불인정)
    is_bull_close_break = (close_p > open_p) and (close_p > box_high)
    is_bear_close_break = (close_p < open_p) and (close_p < box_low)

    raw_signal = None
    if is_squeeze and is_vol_spike and is_marubozu:
        if is_bull_close_break:
            raw_signal = "LONG"
        elif is_bear_close_break:
            raw_signal = "SHORT"

    # 5) 상위 봉(1h/4h) 추세 동조 필터 적용
    htf_aligned, htf_reason = check_htf_alignment(tf, raw_signal, htf_info)
    final_signal = raw_signal if (raw_signal and htf_aligned) else None
    filtered_by_htf = (raw_signal is not None) and (not htf_aligned)

    # 6) 유동성 스윕(가짜 돌파 꼬리 찌르기) 감지
    sweep_warnings = detect_liquidity_sweep(df, box_high, box_low)

    # 7) 권장 손절가(Stop-Loss) 계산:
    #    - LONG: 돌파 장대 캔들의 최저점(Low) 바로 아래 (-0.03% 버퍼)
    #    - SHORT: 돌파 장대 캔들의 최고점(High) 바로 위 (+0.03% 버퍼)
    effective_dir = final_signal or raw_signal or ("LONG" if close_p >= open_p else "SHORT")
    if effective_dir == "LONG":
        stop_price = low_p * (1.0 - SL_BUFFER_PCT)
        stop_basis_label = f"돌파캔들 최저점(${low_p:,.0f}) 바로 아래"
    else:
        stop_price = high_p * (1.0 + SL_BUFFER_PCT)
        stop_basis_label = f"돌파캔들 최고점(${high_p:,.0f}) 바로 위"

    return {
        "candle_ts": int(candle['timestamp']),
        "candle_time_kst": df.index[-2].strftime('%Y-%m-%d %H:%M'),
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
        "is_bull_close_break": is_bull_close_break,
        "is_bear_close_break": is_bear_close_break,
        "raw_signal": raw_signal,
        "htf_aligned": htf_aligned,
        "htf_reason": htf_reason,
        "filtered_by_htf": filtered_by_htf,
        "signal": final_signal,
        "effective_dir": effective_dir,
        "stop_price": stop_price,
        "stop_basis_label": stop_basis_label,
        "sweep_warnings": sweep_warnings,
    }


def evaluate_timeframe(df, tf, tf_cfg, htf_info, state):
    """타임프레임별 종가 마감 돌파 판정 및 중복 알림 방지"""
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    tf_ms = tf_cfg["tf_minutes"] * 60 * 1000
    fresh_ms = tf_cfg["fresh_window_min"] * 60 * 1000

    res = analyze_closed_breakout(df, tf, tf_cfg, htf_info)
    closed_end_ms = res["candle_ts"] + tf_ms

    already_sent = False
    if res["signal"] is not None:
        if (now_ms - closed_end_ms) <= fresh_ms and state.get(tf) != res["candle_ts"]:
            already_sent = False
        else:
            already_sent = True

    res["already_sent"] = already_sent
    return res


def create_breakout_chart(df, tf, eval_res, current_price, filename=None):
    """
    [4. 차트 시각화 고도화]
    - 최근 60개 캔들 + 거래량(Volume) + VWAP(노란선) + VWMA20(하늘선)
    - 박스권 상단(초록 점선) / 박스권 하단(주황 점선) / 권장 손절 라인(빨간 굵은 실선) 수평선 시각화
    """
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
        mpf.make_addplot(plot_df['VWAP'], color='#ffeb3b', width=1.4, label='VWAP'),
        mpf.make_addplot(plot_df['VWMA_20'], color='#29b6f6', width=1.2, linestyle='dotted', label='VWMA(20)'),
    ]

    # 박스권 상단, 박스권 하단, 손절가(Stop-Loss) 수평선 표시
    hlines_vals = [eval_res['box_high'], eval_res['box_low'], eval_res['stop_price']]
    hlines_colors = ['#00e676', '#ff9800', '#ff1744']
    hlines_styles = ['--', '--', '-']
    hlines_widths = [1.2, 1.2, 1.8]

    sig_text = eval_res['signal'] if eval_res['signal'] else "MONITOR"
    title = (
        f"Bitget BTC/USDT [{tf} {sig_text}] | Entry: ${current_price:,.0f} | "
        f"SL: ${eval_res['stop_price']:,.0f} | Box: {eval_res['box_pct']:.2f}% | Vol: {eval_res['vol_pct']:.0f}%"
    )

    mpf.plot(
        plot_df,
        type='candle',
        volume=True,
        addplot=apds,
        hlines=dict(
            hlines=hlines_vals,
            colors=hlines_colors,
            linestyle=hlines_styles,
            linewidths=hlines_widths
        ),
        style=style,
        title=title,
        savefig=dict(fname=filename, dpi=150, bbox_inches='tight'),
        figsize=(12, 7)
    )
    return filename


def build_breakout_message(tf, eval_res, htf_info, current_price, funding_rate_pct, force_mode=False):
    """
    [4. 텔레그램 알림 포맷 고도화]
    - 상위 봉(1h/4h) 추세 일치 여부, 종가 마감 확인 여부, 손절가(돌파 캔들 Low/High 바깥) 및 50배 격리 리스크(%),
      유동성 스윕(휩소) 경고 문구 포함
    """
    sig = eval_res["signal"]
    effective_dir = eval_res["effective_dir"]

    if effective_dir == "LONG":
        dir_text = "🟢 장대양봉 상향 돌파 (LONG)"
        close_confirm = (
            f"✅ 종가(${eval_res['close']:,.0f}) &gt; 박스상단(${eval_res['box_high']:,.0f}) 몸통 마감 확정"
            if eval_res["is_bull_close_break"]
            else f"⚪ 종가(${eval_res['close']:,.0f}) 박스권 내부 마감 (미돌파)"
        )
    else:
        dir_text = "🔴 장대음봉 하향 돌파 (SHORT)"
        close_confirm = (
            f"✅ 종가(${eval_res['close']:,.0f}) &lt; 박스하단(${eval_res['box_low']:,.0f}) 몸통 마감 확정"
            if eval_res["is_bear_close_break"]
            else f"⚪ 종가(${eval_res['close']:,.0f}) 박스권 내부 마감 (미돌파)"
        )

    stop_price = eval_res["stop_price"]
    stop_pct = abs(current_price - stop_price) / current_price * 100.0
    risk_50x = stop_pct * 50.0

    header_tag = f"🚀 <b>[BTC 고신뢰 추세 돌파 포착 | 타임프레임: {tf}]</b>"
    if force_mode and sig is None:
        header_tag = f"🚀 <b>[BTC 멀티 타임프레임 추세 돌파 감시 브리핑 | 기준봉: {tf}]</b>"

    htf_summary = (
        f"1h({'🟢상승' if htf_info['1h']['direction']=='BULL' else '🔴하락'}) / "
        f"4h({'🟢상승' if htf_info['4h']['direction']=='BULL' else '🔴하락'}) — {eval_res['htf_reason']}"
    )

    if eval_res["sweep_warnings"]:
        sweep_section = "\n".join(eval_res["sweep_warnings"])
    else:
        sweep_section = "✅ 유동성 스윕(꼬리 말아올림/내림) 이상 징후 없음"

    msg = (
        f"{header_tag}\n"
        f"• <b>방향</b>: {dir_text}\n"
        f"• <b>상위봉 추세 동조</b>: {htf_summary}\n"
        f"• <b>종가 마감 확인</b>: {close_confirm} ({eval_res['candle_time_kst']} 마감봉)\n"
        f"• <b>돌파 강도</b>: 거래량 평균 대비 <code>{eval_res['vol_pct']:.0f}%</code> ({eval_res['vol_ratio']:.2f}배) / "
        f"박스권 응축 폭 <code>{eval_res['box_pct']:.2f}%</code> (몸통비율 <code>{eval_res['body_ratio_pct']:.0f}%</code>)\n"
        f"• <b>현재가(진입가)</b>: <code>${current_price:,.0f}</code>\n"
        f"• <b>박스권 구간</b>: 상단 <code>${eval_res['box_high']:,.0f}</code> / 하단 <code>${eval_res['box_low']:,.0f}</code> (차트 초록/주황 점선)\n"
        f"• <b>권장 손절가(Stop-Loss)</b>: <code>${stop_price:,.0f}</code> ({eval_res['stop_basis_label']} / 차트 빨간 실선)\n"
        f"  └ <b>예상 리스크</b>: 손절폭 <code>-{stop_pct:.2f}%</code> / <b>50배 격리 기준 <code>-{risk_50x:.1f}%</code></b>\n"
        f"• <b>현재 펀딩비</b>: <code>{funding_rate_pct:.4f}%</code>\n"
        f"• <b>휩소/유동성 스윕 체크</b>:\n  {sweep_section}"
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

    # 4개 타임프레임 데이터 모두 선조회 (1h, 4h 상위 추세를 먼저 파악하기 위함)
    dfs = {}
    for tf in TIMEFRAMES:
        dfs[tf] = fetch_tf_dataframe(exchange, tf)

    htf_info = analyze_htf_trend(dfs["1h"], dfs["4h"])

    print("=" * 105)
    print(f"📊 [Bitget BTC/USDT 고신뢰도 멀티 타임프레임 추세 돌파 & 휩소 필터링 시스템] | 시각(KST): {now_kst}")
    print(f"   현재가: ${current_price:,.2f} | 현재 펀딩비: {funding_rate_pct:.4f}%")
    print(f"   [상위봉 추세] 1h: {htf_info['1h']['desc']} | 4h: {htf_info['4h']['desc']}")
    print("=" * 105)
    print(f"{'TF':<4} | {'박스변동폭(기준)':<16} | {'박스 상단 ~ 하단':<21} | {'거래량(>=2.5x)':<15} | {'장대캔들(ATR/몸통)':<18} | {'종가돌파':<8} | {'판정 및 휩소/역추세 필터'}")
    print("-" * 105)

    triggered_count = 0
    summary_rows = []

    for tf, tf_cfg in TIMEFRAMES.items():
        df = dfs[tf]
        eval_res = evaluate_timeframe(df, tf, tf_cfg, htf_info, state)

        box_str = f"{eval_res['box_pct']:.2f}% (<={tf_cfg['max_box_pct']:.1f}%)"
        box_mark = "O" if eval_res['is_squeeze'] else "X"
        range_str = f"${eval_res['box_high']:,.0f}~${eval_res['box_low']:,.0f}"
        vol_str = f"{eval_res['vol_ratio']:.2f}x ({eval_res['vol_pct']:.0f}%)"
        vol_mark = "O" if eval_res['is_vol_spike'] else "X"
        candle_str = f"{eval_res['atr_ratio']:.2f}x / {eval_res['body_ratio_pct']:.0f}%"
        candle_mark = "O" if eval_res['is_marubozu'] else "X"

        if eval_res["is_bull_close_break"]:
            brk_str = "상향마감"
        elif eval_res["is_bear_close_break"]:
            brk_str = "하향마감"
        else:
            brk_str = "내부마감"

        if eval_res['signal'] and not eval_res['already_sent']:
            status_str = f"🚨 {eval_res['signal']} 정배열 돌파 확정!"
        elif eval_res['signal'] and eval_res['already_sent']:
            status_str = f"⏭️ {eval_res['signal']} (기전송 스킵)"
        elif eval_res['filtered_by_htf']:
            status_str = f"🛡️ 역추세 휩소 차단 ({eval_res['raw_signal']} 무시)"
        else:
            sweep_tag = " [⚠️스윕감지]" if eval_res["sweep_warnings"] else ""
            status_str = f"대기 (박:{box_mark} 볼:{vol_mark} 캔:{candle_mark}){sweep_tag}"

        print(f"{tf:<4} | {box_str:<16} | {range_str:<21} | {vol_str:<15} | {candle_str:<18} | {brk_str:<8} | {status_str}")
        if eval_res["sweep_warnings"]:
            for w in eval_res["sweep_warnings"]:
                print(f"     └ {w}")

        summary_rows.append((tf, eval_res))

        # 실제 조건 충족(상위봉 추세 동조 + 종가 돌파 + 거래량/장대캔들) 및 미전송 타점일 때 알림 발송
        if eval_res['signal'] is not None and not eval_res['already_sent']:
            triggered_count += 1
            chart_path = create_breakout_chart(df, tf, eval_res, current_price)
            caption = build_breakout_message(tf, eval_res, htf_info, current_price, funding_rate_pct, force_mode=False)
            res = send_telegram_photo(chart_path, caption)
            if res.get("ok"):
                print(f"   -> ✅ [{tf}] 텔레그램 고신뢰 돌파 타점 알림 전송 완료!")
                state[tf] = eval_res["candle_ts"]
                save_alert_state(state)

    print("=" * 105)

    # 수동 실행(--force) 시 조건이 안 맞았더라도 정상 작동 확인용 멀티 타임프레임 요약 브리핑 전송
    if force_send and triggered_count == 0:
        print("수동 테스트 모드(--force): 현재 돌파 타점 미충족 상태이나 4개 타임프레임 상태 브리핑을 전송합니다.")
        ref_tf = "15m"
        ref_df = dfs[ref_tf]
        ref_eval = dict(summary_rows)[ref_tf]

        chart_path = create_breakout_chart(ref_df, ref_tf, ref_eval, current_price)
        caption = build_breakout_message(ref_tf, ref_eval, htf_info, current_price, funding_rate_pct, force_mode=True)

        lines = ["\n\n📋 <b>[멀티 타임프레임 실시간 스캔 요약]</b>"]
        for tf, ev in summary_rows:
            sq = "🟢응축" if ev["is_squeeze"] else "⚪비응축"
            sw = " | ⚠️스윕감지" if ev["sweep_warnings"] else ""
            lines.append(
                f"• <b>{tf}</b>: 박스 <code>{ev['box_pct']:.2f}%</code>({sq}) | "
                f"거래량 <code>{ev['vol_ratio']:.2f}배({ev['vol_pct']:.0f}%)</code> | "
                f"몸통 <code>{ev['body_ratio_pct']:.0f}%</code>{sw}"
            )
        caption += "\n".join(lines)
        res = send_telegram_photo(chart_path, caption)
        if res.get("ok"):
            print("✅ 수동 브리핑 텔레그램 전송 완료!")


if __name__ == "__main__":
    main()
