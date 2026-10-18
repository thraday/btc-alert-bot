import os
import warnings
warnings.filterwarnings("ignore", message=".*OpenSSL.*")
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.dirname(__file__), ".mplconfig"))

import ccxt
import pandas as pd
import pandas_ta as ta
import mplfinance as mpf
import requests
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

def fetch_btc_data():
    """Bitget(또는 Binance) 선물에서 BTC/USDT 15분봉 최근 50개 캔들과 현재가 조회"""
    exchanges = [
        ("Bitget", ccxt.bitget({'options': {'defaultType': 'swap'}}), 'BTC/USDT:USDT'),
        ("Binance", ccxt.binance({'options': {'defaultType': 'future'}}), 'BTC/USDT'),
    ]
    
    last_err = None
    for name, exchange, symbol in exchanges:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe='15m', limit=50)
            ticker = exchange.fetch_ticker(symbol)
            current_price = ticker['last']
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
            df['Date'] = pd.to_datetime(df['timestamp'], unit='ms') + pd.Timedelta(hours=9)  # KST 변환
            df.set_index('Date', inplace=True)
            return name, current_price, df
        except Exception as e:
            print(f"[{name}] 데이터 조회 실패, 다음 거래소 시도: {e}")
            last_err = e
            
    raise RuntimeError(f"모든 거래소 조회 실패: {last_err}")

def create_chart(df, exchange_name, current_price, filename="btc_15m_chart.png"):
    """mplfinance를 이용해 15분봉 50개 캔들 차트 이미지 생성"""
    mc = mpf.make_marketcolors(up='#26a69a', down='#ef5350', inherit=True)
    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        gridstyle='--',
        y_on_right=True
    )
    
    title = f"{exchange_name} BTC/USDT Perpetual (15m) - Last: ${current_price:,.2f}"
    mpf.plot(
        df,
        type='candle',
        volume=True,
        style=style,
        title=title,
        mav=(7, 20),
        savefig=dict(fname=filename, dpi=150, bbox_inches='tight'),
        figsize=(10, 6)
    )
    return filename

def send_telegram_photo(image_path, caption):
    """텔레그램으로 차트 이미지 및 텍스트 전송"""
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendPhoto"
    with open(image_path, "rb") as photo:
        payload = {
            "chat_id": CHAT_ID,
            "caption": caption,
            "parse_mode": "HTML"
        }
        files = {"photo": photo}
        response = requests.post(url, data=payload, files=files, timeout=15)
        response.raise_for_status()
        return response.json()

def main():
    if not TELEGRAM_TOKEN or not CHAT_ID:
        raise ValueError(".env 파일에서 TELEGRAM_TOKEN 또는 CHAT_ID를 찾을 수 없습니다.")

    print("1. 선물 시세 및 15분봉 데이터 조회 중...")
    exchange_name, current_price, df = fetch_btc_data()
    last_candle = df.iloc[-1]
    print(f"   -> [{exchange_name}] 현재 BTC/USDT 가격: ${current_price:,.2f}")

    print("2. 15분봉 최근 50개 캔들 차트 생성 중...")
    chart_path = create_chart(df, exchange_name, current_price)
    print(f"   -> 차트 저장 완료: {chart_path}")

    caption = (
        f"🔔 <b>[테스트 알림] {exchange_name} BTC/USDT 선물 시세</b>\n\n"
        f"• <b>현재가</b>: <code>${current_price:,.2f}</code>\n"
        f"• <b>최근 15분봉 시가/종가</b>: <code>${last_candle['Open']:,.2f}</code> / <code>${last_candle['Close']:,.2f}</code>\n"
        f"• <b>최근 15분봉 고가/저가</b>: <code>${last_candle['High']:,.2f}</code> / <code>${last_candle['Low']:,.2f}</code>\n"
        f"• <b>기준 시간(KST)</b>: {df.index[-1].strftime('%Y-%m-%d %H:%M')}"
    )

    print("3. 텔레그램으로 메시지 및 차트 전송 중...")
    res = send_telegram_photo(chart_path, caption)
    if res.get("ok"):
        print("✅ 텔레그램 전송 성공!")
    else:
        print(f"❌ 텔레그램 전송 실패: {res}")

if __name__ == "__main__":
    main()
