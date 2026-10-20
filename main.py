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

# [1. 멀티 타임프레임 확장 및 변동폭 기준 최적화]
# - 5m : 직전 20캔들 변동폭 1.2% 이내 (5분 정각 직후 0~2분)
# - 15m: 직전 20캔들 변동폭 1.8% 이내 (0, 15, 30, 45분 직후 0~3분)
# - 1h : 직전 20캔들 변동폭 3.0% 이내 (매 시간 정각 직후 0~4분)
# - 4h : 직전 20캔들 변동폭 5.0% 이내 (4시간 단위 정각 직후 0~4분)
# - 1d : 직전 20캔들 변동폭 10.0% 이내 (하루가 바뀌는 자정 00:00 직후 0~4분)
TIMEFRAMES = {
    "5m":  {"max_box_pct": 1.2,  "tf_minutes": 5,    "allowed_lag_min": 2},
    "15m": {"max_box_pct": 1.8,  "tf_minutes": 15,   "allowed_lag_min": 3},
    "1h":  {"max_box_pct": 3.0,  "tf_minutes": 60,   "allowed_lag_min": 4},
    "4h":  {"max_box_pct": 5.0,  "tf_minutes": 240,  "allowed_lag_min": 4},
    "1d":  {"max_box_pct": 10.0, "tf_minutes": 1440, "allowed_lag_min": 4},
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

    # VWAP 계산 (1d의 경우 20일 누적 VWAP, 하위봉은 당일 누적 VWAP)
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0
    tp_vol = tp * df['Volume']
    if tf == "1d":
        df['VWAP'] = tp_vol.rolling(20).sum() / df['Volume'].rolling(20).sum()
    else:
        vwap_series = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
        if vwap_series is None or vwap_series.isna().all():
            day_group = df.index.date
            vwap_series = tp_vol.groupby(day_group).cumsum() / df['Volume'].groupby(day_group).cumsum()
        df['VWAP'] = vwap_series

    # 상위 추세 판별용 VWMA(20) & 단기 EMA(9) & ATR(14)
    df['VWMA_20'] = ta.vwma(df['Close'], df['Volume'], length=20)
    df['EMA_9'] = ta.ema(df['Close'], length=9)
    df['ATR_14'] = ta.atr(df['High'], df['Low'], df['Close'], length=14)
    return df


def is_within_close_window(tf, now_utc, df):
    """
    [2. 봉 마감 확정 조건 및 시간대별 실행 타이밍 필터]
    - 5m : 5분 정각 직후 (0~2분)
    - 15m: 0분, 15분, 30분, 45분 직후 (0~3분)
    - 1h : 매 시간 정각 직후 (0~4분)
    - 4h : 4시간 단위 정각 직후 (00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC 또는 캔들 마감 직후 0~4분)
    - 1d : 하루가 바뀌는 자정(00:00 UTC 또는 00:00 KST) 직후 (0~4분)에만 1일봉 데이터 검사
    """
    now_kst = now_utc + pd.Timedelta(hours=9)
    minute = now_utc.minute
    hour_utc = now_utc.hour
    hour_kst = now_kst.hour

    tf_cfg = TIMEFRAMES[tf]
    tf_min = tf_cfg["tf_minutes"]
    allowed_lag = tf_cfg["allowed_lag_min"]

    now_ms = int(now_utc.timestamp() * 1000)
    closed_open_ts = int(df['timestamp'].iloc[-2])
    closed_end_ts = closed_open_ts + (tf_min * 60 * 1000)
    elapsed_min_since_close = (now_ms - closed_end_ts) / 60000.0
    ts_ok = 0.0 <= elapsed_min_since_close < (allowed_lag + 1.0)

    if tf == "5m":
        clock_ok = (minute % 5) <= allowed_lag
        rule_desc = f"5m 정각 직후(0~{allowed_lag}분)"
        is_valid = clock_ok and ts_ok
    elif tf == "15m":
        clock_ok = (minute % 15) <= allowed_lag
        rule_desc = f"0·15·30·45분 직후(0~{allowed_lag}분)"
        is_valid = clock_ok and ts_ok
    elif tf == "1h":
        clock_ok = minute <= allowed_lag
        rule_desc = f"매시 정각 직후(0~{allowed_lag}분)"
        is_valid = clock_ok and ts_ok
    elif tf == "4h":
        # 4시간 단위 정각 직후 (UTC 0,4,8,12,16,20시 또는 거래소 4h 캔들 마감 직후 0~4분)
        clock_ok = (minute <= allowed_lag) and ((hour_utc % 4 == 0) or ts_ok)
        rule_desc = f"4h 정각 직후(0~{allowed_lag}분)"
        is_valid = clock_ok and ts_ok
    elif tf == "1d":
        # 하루가 바뀌는 자정(00시 UTC 또는 00시 KST) 직후 0~4분
        clock_ok = (minute <= allowed_lag) and (hour_utc == 0 or hour_kst == 0)
        rule_desc = f"자정(00시) 직후(0~{allowed_lag}분)"
        is_valid = clock_ok and ts_ok
    else:
        rule_desc = "알 수 없음"
        is_valid = False

    return is_valid, elapsed_min_since_close, rule_desc


def analyze_macro_htf_trend(dfs):
    """
    [3. 상위 추세(1h / 4h / 1d) 복합 분석 - 직전 마감 완성 캔들(iloc[-2]) 기준]
    - 각 상위 봉의 마감 종가 vs VWMA(20) 위치 및 직전 20캔들 박스권 내 상대 위치(상단/하단)를 복합 체크
    """
    def get_single_tf_trend(df, label):
        closed_c = df.iloc[-2]  # 완전히 마감된 직전 캔들
        prev_c = df.iloc[-3]
        box_df = df.iloc[-22:-2]

        close_p = float(closed_c['Close'])
        vwma20 = float(closed_c['VWMA_20'])
        vwma20_prev = float(prev_c['VWMA_20'])
        ema9 = float(closed_c['EMA_9'])

        box_high = float(box_df['High'].max())
        box_low = float(box_df['Low'].min())
        box_mid = (box_high + box_low) / 2.0

        above_vwma = close_p >= vwma20
        vwma_rising = vwma20 >= vwma20_prev
        above_box_mid = close_p >= box_mid

        if close_p > box_high:
            box_pos = "박스상단돌파"
        elif close_p < box_low:
            box_pos = "박스하단이탈"
        elif above_box_mid:
            box_pos = "박스상단권"
        else:
            box_pos = "박스하단권"

        # VWMA(20) 위치와 박스 중심/기울기 복합 판정
        if above_vwma and (vwma_rising or above_box_mid or close_p >= ema9):
            direction = "BULL"
            desc = f"{label} 상승(종가&gt;VWMA20 ${vwma20:,.0f}, {box_pos})"
        elif (not above_vwma) and ((not vwma_rising) or (not above_box_mid) or close_p < ema9):
            direction = "BEAR"
            desc = f"{label} 하락(종가&lt;VWMA20 ${vwma20:,.0f}, {box_pos})"
        else:
            direction = "BULL" if above_vwma else "BEAR"
            desc = f"{label} {'강세우위' if above_vwma else '약세우위'}(VWMA20 ${vwma20:,.0f}, {box_pos})"

        return {
            "tf": label,
            "direction": direction,
            "close": close_p,
            "vwma20": vwma20,
            "box_pos": box_pos,
            "desc": desc,
        }

    return {
        "1h": get_single_tf_trend(dfs["1h"], "1h"),
        "4h": get_single_tf_trend(dfs["4h"], "4h"),
        "1d": get_single_tf_trend(dfs["1d"], "1d"),
    }


def check_macro_alignment(tf, signal_dir, htf_info):
    """
    하위 봉(5m, 15m, 1h) 및 중장기 봉(4h, 1d) 시그널 발생 시
    거시 상위 봉(4h & 1d, 그리고 단기봉의 경우 1h 포함) 추세 일치 여부 복합 판정
    """
    d1h = htf_info["1h"]["direction"]
    d4h = htf_info["4h"]["direction"]
    d1d = htf_info["1d"]["direction"]

    target = "BULL" if signal_dir == "LONG" else "BEAR"
    kor_target = "상승" if target == "BULL" else "하락"

    s1h = "상승" if d1h == "BULL" else "하락"
    s4h = "상승" if d4h == "BULL" else "하락"
    s1d = "상승" if d1d == "BULL" else "하락"

    if tf in ("5m", "15m"):
        # 4h & 1d 거시 추세와 1h 중기 추세 복합 체크
        if d1h == target and d4h == target and d1d == target:
            return True, f"1h·4h·1d 전구간 {kor_target} 정배열 동조"
        elif d4h == target and d1d == target:
            return True, f"상위봉 4h·1d {kor_target} 동조 (1h {s1h})"
        elif d1h == target and (d4h == target or d1d == target):
            return True, f"상위봉 1h·{'4h' if d4h == target else '1d'} {kor_target} 동조"
        else:
            return False, f"상위봉 4h({s4h})·1d({s1d}) 역추세 불일치"
    elif tf == "1h":
        if d4h == target and d1d == target:
            return True, f"거시 상위봉 4h·1d {kor_target} 완벽 동조"
        elif d4h == target or d1d == target:
            return True, f"상위봉 4h({s4h})·1d({s1d}) 부분 동조"
        else:
            return False, f"거시 상위봉 4h·1d 모두 역방향({s4h})"
    elif tf == "4h":
        if d1d == target:
            return True, f"1d 일봉 대추세({s1d}) 동조"
        else:
            return False, f"1d 일봉 대추세({s1d}) 역행"
    else:  # 1d
        return True, f"1d 거시 일봉 추세({s1d}) 직접 돌파"


def detect_liquidity_sweep_closed_only(df, box_high, box_low):
    """
    [유동성 흡수(Liquidity Sweep) 판정 - 오직 완전히 마감된 직전 캔들(iloc[-2])만 기준]
    """
    c_closed = df.iloc[-2]       # 완전히 마감된 직전 캔들
    c_prev_closed = df.iloc[-3]  # 그 직전 마감 캔들

    cl_h, cl_l, cl_close = float(c_closed['High']), float(c_closed['Low']), float(c_closed['Close'])
    prev_close = float(c_prev_closed['Close'])

    sweep_type = None
    supply_note = None
    warnings_list = []

    # 1) 상단 유동성 스윕 (완성봉 iloc[-2] 기준)
    if prev_close > box_high and cl_close < box_high:
        sweep_type = "UPPER_SWEEP"
        supply_note = "상단 스윕 후 박스 회귀 마감"
        warnings_list.append("상단 유동성 스윕(Bull Trap) 후 박스 내부 재진입 마감")
    elif cl_h > box_high and cl_close <= box_high:
        sweep_type = "UPPER_SWEEP"
        supply_note = "상단 스윕 후 꼬리 마감"
        warnings_list.append("상단 유동성 스윕(Bull Trap) 꼬리 찌르기 마감")

    # 2) 하단 유동성 스윕 (완성봉 iloc[-2] 기준)
    if prev_close < box_low and cl_close > box_low:
        sweep_type = "LOWER_SWEEP"
        supply_note = "하단 스윕 후 박스 회귀 마감"
        warnings_list.append("하단 유동성 스윕(Bear Trap) 후 박스 내부 재진입 마감")
    elif cl_l < box_low and cl_close >= box_low:
        sweep_type = "LOWER_SWEEP"
        supply_note = "하단 스윕 후 꼬리 마감"
        warnings_list.append("하단 유동성 스윕(Bear Trap) 꼬리 찌르기 마감")

    return sweep_type, supply_note, warnings_list


def analyze_closed_breakout(df, tf, tf_cfg, htf_info):
    """
    [완전히 마감된 직전 캔들(iloc[-2]) 전용 돌파 및 유동성 스윕 분석]
    """
    max_box_pct = tf_cfg["max_box_pct"]
    box_df = df.iloc[-22:-2]  # 돌파 캔들(iloc[-2]) 직전 20개 완성 캔들
    atr_prev = float(df['ATR_14'].iloc[-3])
    candle = df.iloc[-2]      # 완전히 마감된 직전 캔들 (iloc[-2])

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

    # 4) 직전 완성 캔들(iloc[-2]) 종가(Close) 기준 확실한 몸통 돌파 여부
    is_bull_close_break = (close_p > open_p) and (close_p > box_high)
    is_bear_close_break = (close_p < open_p) and (close_p < box_low)

    raw_signal = None
    if is_squeeze and is_vol_spike and is_marubozu:
        if is_bull_close_break:
            raw_signal = "LONG"
        elif is_bear_close_break:
            raw_signal = "SHORT"

    # 5) 직전 완성 캔들(iloc[-2]) 전용 유동성 스윕 감지
    sweep_type, sweep_supply_note, sweep_warnings = detect_liquidity_sweep_closed_only(df, box_high, box_low)
    is_sweep_event = is_squeeze and is_vol_spike and (sweep_type is not None)

    if raw_signal is not None:
        effective_dir = raw_signal
    elif sweep_type == "UPPER_SWEEP":
        effective_dir = "LONG"
    elif sweep_type == "LOWER_SWEEP":
        effective_dir = "SHORT"
    else:
        effective_dir = "LONG" if close_p >= open_p else "SHORT"

    # 6) 상위 봉(1h / 4h / 1d) 복합 추세 동조 확인
    htf_aligned, htf_summary = check_macro_alignment(tf, effective_dir, htf_info)

    # 7) 전략 타입 판정 (Type A 돌파 추격 vs Type B 유동성 스윕/역추세 반전)
    final_signal = None
    strategy_type = "NONE"

    if raw_signal is not None and htf_aligned and sweep_type is None:
        strategy_type = "TYPE_A"
        final_signal = raw_signal
        strategy_text = f"🚀 돌파 추격 매매 권장 ({htf_summary})"
        supply_detail = "장대몸통 종가 돌파 마감"
    elif (raw_signal is not None) or is_sweep_event:
        strategy_type = "TYPE_B"
        final_signal = effective_dir
        rev_word = "반등" if effective_dir == "SHORT" else "조정"
        if sweep_type is not None:
            strategy_text = f"⚠️ 유동성 스윕 경고 (추격 금지 / 반대 방향 {rev_word} 관점 · {htf_summary})"
            supply_detail = sweep_supply_note
        else:
            strategy_text = f"⚠️ 상위봉 역추세 휩소 주의 (추격 금지 / {htf_summary} · 박스 회귀 주의)"
            supply_detail = "역추세 돌파 마감 (휩소 주의)"
    else:
        if sweep_type is not None:
            strategy_type = "TYPE_B"
            rev_word = "반등" if effective_dir == "SHORT" else "조정"
            strategy_text = f"⚠️ 유동성 스윕 경고 (추격 금지 / 반대 방향 {rev_word} 관점 · {htf_summary})"
            supply_detail = sweep_supply_note
        elif htf_aligned and (is_bull_close_break or is_bear_close_break):
            strategy_type = "TYPE_A"
            strategy_text = f"🚀 돌파 추격 매매 권장 ({htf_summary})"
            supply_detail = "장대몸통 종가 돌파 마감"
        else:
            strategy_type = "TYPE_B"
            rev_word = "반등" if effective_dir == "SHORT" else "조정"
            strategy_text = f"⚠️ 박스권 내부 수렴 중 ({htf_summary} / 돌파 전 추격 금지)"
            supply_detail = sweep_supply_note if sweep_supply_note else "박스권 내부 마감"

    # 8) 권장 손절가(Stop-Loss) 계산 (완성된 돌파 캔들 iloc[-2]의 최저점/최고점 기준)
    if effective_dir == "LONG":
        stop_price = low_p * (1.0 - SL_BUFFER_PCT)
    else:
        stop_price = high_p * (1.0 + SL_BUFFER_PCT)

    if tf == "1d":
        closed_open_str = df.index[-2].strftime('%m/%d')
        closed_label = f"완성일봉(iloc[-2]: {closed_open_str})"
    else:
        closed_open_kst = df.index[-2].strftime('%H:%M')
        closed_end_kst = (df.index[-2] + pd.Timedelta(minutes=tf_cfg["tf_minutes"])).strftime('%H:%M')
        closed_label = f"완성봉(iloc[-2]: {closed_open_kst}~{closed_end_kst})"

    return {
        "candle_ts": int(candle['timestamp']),
        "closed_candle_label": closed_label,
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
        "htf_summary": htf_summary,
        "signal": final_signal,
        "effective_dir": effective_dir,
        "strategy_type": strategy_type,
        "strategy_text": strategy_text,
        "supply_detail": supply_detail,
        "stop_price": stop_price,
        "sweep_warnings": sweep_warnings,
    }


def create_breakout_chart(df, tf, eval_res, current_price, filename=None):
    """
    [차트 시각화 - 완전히 마감된 캔들(iloc[:-1]) 기준 60개 표시]
    """
    if filename is None:
        filename = f"btc_breakout_{tf}.png"

    plot_df = df.iloc[:-1].tail(60).copy()

    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        gridstyle=':',
        y_on_right=True
    )

    apds = [
        mpf.make_addplot(plot_df['VWAP'], color='#ffeb3b', width=1.3),
    ]

    hlines_vals = [eval_res['box_high'], eval_res['box_low'], eval_res['stop_price']]
    hlines_colors = ['#00e5ff', '#00e5ff', '#ff1744']
    hlines_styles = ['--', '--', '-']
    hlines_widths = [1.1, 1.1, 1.8]

    sig_label = eval_res['effective_dir']
    type_tag = "CHASE" if eval_res['strategy_type'] == "TYPE_A" else "SWEEP-WARN"
    title = (
        f"BTC/USDT {tf} [{sig_label} - {type_tag}]  |  Entry: ${current_price:,.0f}  |  "
        f"SL: ${eval_res['stop_price']:,.0f}  |  Box: ${eval_res['box_low']:,.0f}~${eval_res['box_high']:,.0f}"
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
        figsize=(11, 6.5)
    )
    return filename


def build_breakout_message(tf, eval_res, current_price):
    """
    [초간결 미니 카드형 텔레그램 알림 포맷 (4h/1d 거시 추세 일치 여부 포함)]
    """
    effective_dir = eval_res["effective_dir"]
    emoji = "🟢" if effective_dir == "LONG" else "🔴"

    stop_price = eval_res["stop_price"]
    stop_pct = abs(current_price - stop_price) / current_price * 100.0
    risk_50x = stop_pct * 50.0

    vol_word = "폭발" if eval_res["vol_pct"] >= 200.0 else "발생"

    msg = (
        f"{emoji} <b>[BTC {effective_dir} 포착 - {tf}]</b>\n"
        f"• 매매 전략: {eval_res['strategy_text']}\n"
        f"• 현재가(진입): <b>${current_price:,.0f}</b>\n"
        f"• 손절가(SL): <b>${stop_price:,.0f}</b> (-{stop_pct:.2f}% / 50배 리스크 -{risk_50x:.1f}%)\n"
        f"• 박스권 구간: 상단 ${eval_res['box_high']:,.0f} / 하단 ${eval_res['box_low']:,.0f}\n"
        f"• 수급 상태: 거래량 평균 대비 <b>{eval_res['vol_pct']:.0f}%</b> {vol_word} ({eval_res['supply_detail']})"
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

    now_utc = datetime.now(timezone.utc)
    now_kst_dt = now_utc + pd.Timedelta(hours=9)
    now_kst = now_kst_dt.strftime('%Y-%m-%d %H:%M:%S')

    current_price, funding_rate_pct = fetch_common_info(exchange)

    # 5개 타임프레임(['5m', '15m', '1h', '4h', '1d']) 데이터 조회
    dfs = {}
    for tf in TIMEFRAMES:
        dfs[tf] = fetch_tf_dataframe(exchange, tf)

    htf_info = analyze_macro_htf_trend(dfs)

    print("=" * 124)
    print(f"📊 [Bitget BTC/USDT 5대 타임프레임(5m~1d) 봉마감 돌파/스윕 감시 시스템] | 실행시각(KST): {now_kst}")
    print(f"   현재가: ${current_price:,.2f} | 현재 펀딩비: {funding_rate_pct:.4f}%")
    print(f"   [상위봉 마감추세] 1h: {htf_info['1h']['desc']} | 4h: {htf_info['4h']['desc']} | 1d: {htf_info['1d']['desc']}")
    print("=" * 124)
    print(f"{'TF':<4} | {'타이밍 게이트 (봉마감 직후 필터)':<34} | {'직전 마감봉(iloc[-2]) 박스폭/거래량':<28} | {'상위추세 동조 및 판정 결과'}")
    print("-" * 124)

    triggered_count = 0
    evaluated_rows = []

    for tf, tf_cfg in TIMEFRAMES.items():
        df = dfs[tf]
        is_valid_timing, elapsed_min, rule_desc = is_within_close_window(tf, now_utc, df)
        eval_res = analyze_closed_breakout(df, tf, tf_cfg, htf_info)
        metric_str = f"박스 {eval_res['box_pct']:.2f}% (<={tf_cfg['max_box_pct']:.1f}%) / 거래량 {eval_res['vol_pct']:.0f}%"

        # 타이밍 게이트 미통과 시 시그널 발송 원천 차단 (터미널에는 직전 완성봉 수치 참고 표시)
        if not is_valid_timing:
            gate_str = f"⏸️ 비활성 (+{elapsed_min:.0f}분 경과, {rule_desc})"
            print(f"{tf:<4} | {gate_str:<34} | {metric_str:<28} | ⛔ 봉 진행 중 (시그널 검사 차단 | {eval_res['htf_summary']})")
            continue

        # 타이밍 게이트 통과: 직전 마감 완성 캔들(iloc[-2]) 기준으로만 엄격 판정 수행
        already_sent = (state.get(tf) == eval_res["candle_ts"])
        eval_res["already_sent"] = already_sent
        evaluated_rows.append((tf, eval_res))

        gate_str = f"✅ 활성 (마감+{elapsed_min:.1f}분, {rule_desc})"

        if eval_res['signal'] and not already_sent:
            status_str = f"🚨 [{eval_res['strategy_type']}] {eval_res['strategy_text']}"
        elif eval_res['signal'] and already_sent:
            status_str = f"⏭️ {eval_res['signal']} (기전송 완료 스킵)"
        else:
            status_str = f"대기 ({eval_res['supply_detail']} | {eval_res['htf_summary']})"

        print(f"{tf:<4} | {gate_str:<34} | {metric_str:<28} | {status_str}")

        # 실제 시그널 포착 시 텔레그램 알림 발송
        if eval_res['signal'] is not None and not already_sent:
            triggered_count += 1
            chart_path = create_breakout_chart(df, tf, eval_res, current_price)
            caption = build_breakout_message(tf, eval_res, current_price)
            res = send_telegram_photo(chart_path, caption)
            if res.get("ok"):
                print(f"   -> ✅ [{tf}] 텔레그램 미니 카드 알림 전송 완료!")
                state[tf] = eval_res["candle_ts"]
                save_alert_state(state)

    print("=" * 124)

    # 수동 테스트 실행(--force) 시에만 1회 샘플 브리핑 전송
    if force_send and triggered_count == 0:
        ref_tf = evaluated_rows[-1][0] if evaluated_rows else "15m"
        ref_df = dfs[ref_tf]
        ref_eval = analyze_closed_breakout(ref_df, ref_tf, TIMEFRAMES[ref_tf], htf_info)
        print(f"수동 테스트 모드(--force): [{ref_tf}] {ref_eval['closed_candle_label']} 기준 미니 카드 브리핑을 전송합니다.")
        chart_path = create_breakout_chart(ref_df, ref_tf, ref_eval, current_price)
        caption = build_breakout_message(ref_tf, ref_eval, current_price)
        print("\n[전송될 미니 카드 메시지 미리보기]")
        print(caption)
        res = send_telegram_photo(chart_path, caption)
        if res.get("ok"):
            print("✅ 수동 테스트 미니 카드 전송 완료!")


if __name__ == "__main__":
    main()
