import os
import requests


BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]


def get_quote(symbol):
    url = "https://finnhub.io/api/v1/quote"

    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        },
        timeout=30
    )

    response.raise_for_status()
    data = response.json()

    print("Finnhub response:", data)
    return data


def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": text
        },
        timeout=30
    )

    print("Telegram status:", response.status_code)
    print("Telegram response:", response.text)

    response.raise_for_status()


def main():
    symbol = "AAPL"
    quote = get_quote(symbol)

    current_price = float(quote.get("c", 0) or 0)
    change = float(quote.get("d", 0) or 0)
    change_percent = float(quote.get("dp", 0) or 0)
    day_high = float(quote.get("h", 0) or 0)
    day_low = float(quote.get("l", 0) or 0)
    previous_close = float(quote.get("pc", 0) or 0)

    if current_price <= 0:
        raise RuntimeError(
            f"{symbol} 현재가 조회 실패: {quote}"
        )

    message = (
        "🇺🇸 미국주식 스캐너 연결 테스트\n\n"
        f"종목: {symbol}\n"
        f"현재가: ${current_price:.2f}\n"
        f"등락: ${change:+.2f} ({change_percent:+.2f}%)\n"
        f"당일 고가: ${day_high:.2f}\n"
        f"당일 저가: ${day_low:.2f}\n"
        f"전일 종가: ${previous_close:.2f}\n\n"
        "✅ Finnhub와 텔레그램 연결 정상"
    )

    send_message(message)
    print("테스트 전송 완료")


if __name__ == "__main__":
    main()
