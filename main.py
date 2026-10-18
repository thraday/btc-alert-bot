import os
import sys
import warnings

warnings.filterwarnings("ignore", message=".*OpenSSL.*")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), ".mplconfig"))

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
TIMEFRAME = "15m"
LIMIT = 100


def fetch_market_data():
    """Bitget 선물에서 BTC/USDT 15분봉 최근 100개, 현재가, 현재 펀딩비 조회"""
    exchange = ccxt.bitget({'options': {'defaultType': 'swap'}})

    ohlcv = exchange.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=LIMIT)
    ticker = exchange.fetch_ticker(SYMBOL)
    current_price = float(ticker['last'])

    # 펀딩비 조회 (소수점 형태 예: 0.0001 -> 0.0100%)
    funding_rate_pct = 0.0
    try:
        funding_info = exchange.fetch_funding_rate(SYMBOL)
        if funding_info and funding_info.get('fundingRate') is not None:
            funding_rate_pct = float(funding_info['fundingRate']) * 100
    except Exception as e:
        print(f"[경고] 펀딩비 조회 실패: {e}")

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    # UTC 시간 인덱스 (당일 VWAP 계산 기준) 및 KST 시간 컬럼
    df['UTC_Date'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['Date'] = df['UTC_Date'] + pd.Timedelta(hours=9)  # KST
    df.set_index('Date', inplace=True)

    return current_price, funding_rate_pct, df


def calculate_indicators(df):
    """
    1. VWAP (당일 기준 - KST 일자 변경 기준 누적, pandas_ta 호환)
    2. VWMA (길이 20)
    3. RSI (길이 14)
    4. 최근 100개 캔들 Swing High / Swing Low 기준 피보나치 되돌림 (0.382, 0.5, 0.618)
    """
    # pandas_ta VWAP (DatetimeIndex 정렬 기반 일별 리셋)
    df['VWAP'] = ta.vwap(df['High'], df['Low'], df['Close'], df['Volume'])
    if df['VWAP'].isna().all():
        # Fallback: 당일 기준 수동 VWAP 계산
        tp = (df['High'] + df['Low'] + df['Close']) / 3.0
        tp_vol = tp * df['Volume']
        day_group = df.index.date
        df['VWAP'] = tp_vol.groupby(day_group).cumsum() / df['Volume'].groupby(day_group).cumsum()

    # VWMA (길이 20)
    df['VWMA_20'] = ta.vwma(df['Close'], df['Volume'], length=20)

    # RSI (길이 14)
    df['RSI_14'] = ta.rsi(df['Close'], length=14)

    # 최근 100개 캔들 최고점(Swing High) / 최저점(Swing Low)
    swing_high = float(df['High'].max())
    swing_low = float(df['Low'].min())
    diff = swing_high - swing_low

    # 고점 대비 되돌림 레벨 계산:
    # 0.382 되돌림 (상단 저항 측), 0.5 중심, 0.618 되돌림 (하단 지지 측)
    fib_382 = swing_high - diff * 0.382
    fib_500 = swing_high - diff * 0.500
    fib_618 = swing_high - diff * 0.618

    fib_levels = {
        'swing_high': swing_high,
        'swing_low': swing_low,
        '0.382': fib_382,
        '0.5': fib_500,
        '0.618': fib_618,
    }

    return df, fib_levels


def evaluate_signals(current_price, df, fib_levels):
    """타점 감지 조건 판별"""
    latest = df.iloc[-1]
    vwap = float(latest['VWAP'])
    vwma20 = float(latest['VWMA_20'])
    rsi14 = float(latest['RSI_14'])

    fib_382 = fib_levels['0.382']
    fib_500 = fib_levels['0.5']
    fib_618 = fib_levels['0.618']

    # 1) 가격 조건
    # 피보나치 0.5 ~ 0.618 구간 (fib_618 <= price <= fib_500)
    in_long_fib = min(fib_500, fib_618) <= current_price <= max(fib_500, fib_618)
    # 피보나치 0.382 ~ 0.5 구간 (fib_500 <= price <= fib_382)
    in_short_fib = min(fib_382, fib_500) <= current_price <= max(fib_382, fib_500)
    # VWAP 라인과 ±0.2% 이내 근접
    vwap_diff_pct = abs(current_price - vwap) / vwap * 100.0
    near_vwap = vwap_diff_pct <= 0.2

    # 2) 타점 결합 조건
    long_price_ok = in_long_fib or near_vwap
    long_rsi_ok = rsi14 <= 42.0
    is_long = long_price_ok and long_rsi_ok

    short_price_ok = in_short_fib or near_vwap
    short_rsi_ok = rsi14 >= 58.0
    is_short = short_price_ok and short_rsi_ok

    signal = None
    if is_long:
        signal = "LONG"
    elif is_short:
        signal = "SHORT"

    metrics = {
        'current_price': current_price,
        'vwap': vwap,
        'vwma20': vwma20,
        'rsi14': rsi14,
        'vwap_diff_pct': vwap_diff_pct,
        'in_long_fib': in_long_fib,
        'in_short_fib': in_short_fib,
        'near_vwap': near_vwap,
        'long_rsi_ok': long_rsi_ok,
        'short_rsi_ok': short_rsi_ok,
    }
    return signal, metrics


def create_signal_chart(df, fib_levels, current_price, signal_label, filename="btc_signal_chart.png"):
    """최근 80개 캔들 + VWAP + VWMA(20) + 피보나치 수평선이 그려진 차트 생성"""
    plot_df = df.tail(80).copy()

    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        gridstyle='--',
        y_on_right=True
    )

    apds = [
        mpf.make_addplot(plot_df['VWAP'], color='#ffeb3b', width=1.6, label='VWAP'),
        mpf.make_addplot(plot_df['VWMA_20'], color='#29b6f6', width=1.2, linestyle='dotted', label='VWMA(20)'),
    ]

    hlines_vals = [
        fib_levels['swing_high'],
        fib_levels['0.382'],
        fib_levels['0.5'],
        fib_levels['0.618'],
        fib_levels['swing_low'],
    ]
    hlines_colors = ['#ef5350', '#ff9800', '#e040fb', '#00e676', '#26a69a']

    title = (
        f"Bitget BTC/USDT 15m [{signal_label}] | Price: ${current_price:,.1f} | "
        f"VWAP: ${plot_df['VWAP'].iloc[-1]:,.1f} | RSI: {plot_df['RSI_14'].iloc[-1]:.1f}"
    )

    mpf.plot(
        plot_df,
        type='candle',
        volume=True,
        addplot=apds,
        hlines=dict(hlines=hlines_vals, colors=hlines_colors, linestyle='--', linewidths=1.0),
        style=style,
        title=title,
        savefig=dict(fname=filename, dpi=150, bbox_inches='tight'),
        figsize=(12, 7)
    )
    return filename


def build_message(signal, metrics, fib_levels, funding_rate_pct, force_mode=False):
    """요청된 포맷에 맞춘 텔레그램 알림 메시지 생성"""
    price = metrics['current_price']
    vwap = metrics['vwap']
    rsi = metrics['rsi14']

    # 실제 시그널이 없으면(강제 테스트 브리핑 시) 현재 RSI가 어느 쪽에 가까운지 기준으로 예시 방향 설정
    effective_dir = signal if signal else ("LONG" if rsi < 50 else "SHORT")

    if signal == "LONG":
        header = "🚨 <b>[BTC 15m 타점 포착: 🟢 LONG]</b>"
    elif signal == "SHORT":
        header = "🚨 <b>[BTC 15m 타점 포착: 🔴 SHORT]</b>"
    else:
        header = f"🚨 <b>[BTC 15m 타점 포착: {effective_dir} (강제 브리핑 테스트)]</b>"

    if effective_dir == "LONG":
        fib_line = f"피보나치 구간: 0.5(${fib_levels['0.5']:,.0f}) ~ 0.618(${fib_levels['0.618']:,.0f})"
        stop_price = fib_levels['swing_low']
        stop_label = "직전 스윙 로우"
        stop_pct = abs(price - stop_price) / price * 100.0
    else:
        fib_line = f"피보나치 구간: 0.382(${fib_levels['0.382']:,.0f}) ~ 0.5(${fib_levels['0.5']:,.0f})"
        stop_price = fib_levels['swing_high']
        stop_label = "직전 스윙 하이"
        stop_pct = abs(stop_price - price) / price * 100.0

    risk_50x = stop_pct * 50.0

    msg = (
        f"{header}\n"
        f"• 현재가: <code>${price:,.0f}</code> / VWAP: <code>${vwap:,.0f}</code>\n"
        f"• {fib_line}\n"
        f"• RSI(14): <code>{rsi:.1f}</code> / 현재 펀딩비: <code>{funding_rate_pct:.4f}%</code>\n"
        f"• 권장 손절가({stop_label}): <code>${stop_price:,.0f}</code> "
        f"(손절폭 -{stop_pct:.2f}% / 50배 기준 약 -{risk_50x:.0f}% 리스크)"
    )

    if force_mode and signal is None:
        msg += (
            f"\n\nℹ️ <i>현재 실시간 타점 조건은 미충족 상태이나, 브리핑 포맷 확인을 위해 강제 전송된 메시지입니다.</i>\n"
            f"• VWMA(20): <code>${metrics['vwma20']:,.0f}</code> / VWAP 괴리율: <code>{metrics['vwap_diff_pct']:.2f}%</code>\n"
            f"• 100봉 스윙 하이/로우: <code>${fib_levels['swing_high']:,.0f}</code> / <code>${fib_levels['swing_low']:,.0f}</code>"
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

    print("=" * 60)
    print("📊 [Bitget BTC/USDT 15m 타점 감시 봇 실행]")
    print("=" * 60)

    current_price, funding_rate_pct, df = fetch_market_data()
    df, fib_levels = calculate_indicators(df)
    signal, metrics = evaluate_signals(current_price, df, fib_levels)

    print(f"기준 시간 (KST)       : {df.index[-1].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"현재가 (Current)      : ${current_price:,.2f}")
    print(f"당일 VWAP             : ${metrics['vwap']:,.2f} (현재가와 괴리율: {metrics['vwap_diff_pct']:.3f}%)")
    print(f"VWMA (20)             : ${metrics['vwma20']:,.2f}")
    print(f"RSI (14)              : {metrics['rsi14']:.2f}")
    print(f"현재 펀딩비           : {funding_rate_pct:.4f}%")
    print("-" * 60)
    print(f"100캔들 Swing High    : ${fib_levels['swing_high']:,.2f}")
    print(f"피보나치 0.382 저항   : ${fib_levels['0.382']:,.2f}")
    print(f"피보나치 0.500 중심   : ${fib_levels['0.5']:,.2f}")
    print(f"피보나치 0.618 지지   : ${fib_levels['0.618']:,.2f}")
    print(f"100캔들 Swing Low     : ${fib_levels['swing_low']:,.2f}")
    print("-" * 60)
    print(f"[🟢 LONG 조건 체크]   : Fib(0.5~0.618)진입={metrics['in_long_fib']} | VWAP(±0.2%)근접={metrics['near_vwap']} | RSI<=42={metrics['long_rsi_ok']}")
    print(f"[🔴 SHORT 조건 체크]  : Fib(0.382~0.5)진입={metrics['in_short_fib']} | VWAP(±0.2%)근접={metrics['near_vwap']} | RSI>=58={metrics['short_rsi_ok']}")
    print(f"=> 최종 판정          : {signal if signal else '타점 조건 미충족 (대기)'}")
    print("=" * 60)

    if signal is not None or force_send:
        label = signal if signal else "TEST BRIEFING"
        print(f"차트 생성 및 텔레그램 알림 전송 진행 중... (모드: {label})")
        chart_file = create_signal_chart(df, fib_levels, current_price, label)
        caption = build_message(signal, metrics, fib_levels, funding_rate_pct, force_mode=force_send)
        print("\n[전송될 텔레그램 메시지 미리보기]")
        print(caption)
        print("-" * 60)
        res = send_telegram_photo(chart_file, caption)
        if res.get("ok"):
            print("✅ 텔레그램 브리핑 전송 완료!")
        else:
            print(f"❌ 텔레그램 전송 실패: {res}")
    else:
        print("조건이 충족되지 않아 텔레그램 알림을 전송하지 않고 종료합니다.")


if __name__ == "__main__":
    main()
