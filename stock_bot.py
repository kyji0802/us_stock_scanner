import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]

STATE_FILE = Path("signal_state.json")

# 먼저 미국 주요 종목으로 안정적으로 시작
WATCHLIST = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "AMD", "AVGO", "NFLX",
    "PLTR", "COIN", "MSTR", "ARM", "MU",
    "SMCI", "CRWD", "PANW", "ORCL", "INTC",
    "QCOM", "AMAT", "TSM", "UBER", "SHOP",
    "JPM", "BAC", "WMT", "COST", "LLY"
]

MIN_PRICE = 5.0
MAX_PRICE = 100.0
MIN_AVG_VOLUME = 1_000_000

# 같은 종목·같은 단계의 중복 알림 방지 시간
ALERT_COOLDOWN_HOURS = 12


def load_state():
    if not STATE_FILE.exists():
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=2
        )


def get_finnhub_quote(symbol):
    url = "https://finnhub.io/api/v1/quote"

    response = requests.get(
        url,
        params={
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        },
        timeout=20
    )

    response.raise_for_status()
    data = response.json()

    current_price = float(data.get("c", 0) or 0)

    if current_price <= 0:
        raise RuntimeError(
            f"{symbol} 현재가 조회 실패: {data}"
        )

    return {
        "price": current_price,
        "change": float(data.get("d", 0) or 0),
        "change_percent": float(data.get("dp", 0) or 0),
        "high": float(data.get("h", 0) or 0),
        "low": float(data.get("l", 0) or 0),
        "previous_close": float(data.get("pc", 0) or 0),
        "timestamp": int(data.get("t", 0) or 0)
    }


def get_daily_data(symbol):
    ticker = yf.Ticker(symbol)

    data = ticker.history(
        period="6mo",
        interval="1d",
        auto_adjust=False
    )

    if data.empty:
        raise RuntimeError(f"{symbol} 일봉 데이터 없음")

    data = data.dropna(
        subset=["Open", "High", "Low", "Close", "Volume"]
    )

    if len(data) < 65:
        raise RuntimeError(
            f"{symbol} 일봉 부족: {len(data)}개"
        )

    return data


def calculate_rsi(close, period=14):
    delta = close.diff()

    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    average_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    average_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = average_gain / average_loss.replace(0, float("nan"))
    rsi = 100 - (100 / (1 + rs))

    value = rsi.iloc[-1]

    if pd.isna(value):
        return 50.0

    return float(value)


def analyze_symbol(symbol):
    quote = get_finnhub_quote(symbol)
    data = get_daily_data(symbol)

    current_price = quote["price"]
    previous_close = quote["previous_close"]

    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float)

    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())

    # 현재 진행 중인 일봉을 제외하고 이전 20거래일 고점 계산
    prior_data = data.iloc[:-1]

    if len(prior_data) < 60:
        prior_data = data

    prior_20_high = float(
        prior_data["High"].tail(20).max()
    )

    prior_10_low = float(
        prior_data["Low"].tail(10).min()
    )

    average_volume_20 = float(
        volume.tail(20).mean()
    )

    current_volume = float(
        volume.iloc[-1]
    )

    if average_volume_20 <= 0:
        volume_ratio = 0.0
    else:
        volume_ratio = current_volume / average_volume_20

    rsi = calculate_rsi(close)

    distance_to_high = (
        (prior_20_high - current_price)
        / prior_20_high
        * 100
    )

    trend_ok = ma20 > ma60
    average_volume_ok = average_volume_20 >= MIN_AVG_VOLUME

    buy1 = (
        previous_close <= prior_20_high
        and current_price > prior_20_high
        and trend_ok
    )

    ready = (
        not buy1
        and 0 <= distance_to_high <= 2
        and trend_ok
    )

    sell = current_price < prior_10_low

    if sell:
        stage = "SELL"
    elif buy1:
        stage = "BUY1"
    elif ready:
        stage = "준비"
    else:
        return None

    if current_price < MIN_PRICE or current_price > MAX_PRICE:
        return None

    if not average_volume_ok and stage != "SELL":
        return None

    reasons = []

    if stage == "BUY1":
        reasons.append("20일 고점 돌파")
    elif stage == "준비":
        reasons.append(
            f"20일 고점까지 {distance_to_high:.2f}%"
        )
    elif stage == "SELL":
        reasons.append("10일 저점 이탈")

    if trend_ok:
        reasons.append("20일선 > 60일선")
    else:
        reasons.append("20일선 ≤ 60일선")

    if volume_ratio >= 2:
        reasons.append(
            f"거래량 {volume_ratio:.1f}배 급증"
        )
    elif volume_ratio >= 1:
        reasons.append(
            f"거래량 {volume_ratio:.1f}배"
        )

    reasons.append(f"RSI {rsi:.1f}")

    score = 0

    if stage == "BUY1":
        score += 45
    elif stage == "준비":
        score += 30
    elif stage == "SELL":
        score += 40

    if trend_ok:
        score += 20

    if 50 <= rsi <= 70:
        score += 15
    elif stage == "SELL" and rsi < 45:
        score += 15

    if volume_ratio >= 2:
        score += 20
    elif volume_ratio >= 1:
        score += 10

    score = min(score, 100)

    return {
        "symbol": symbol,
        "stage": stage,
        "price": current_price,
        "change": quote["change"],
        "change_percent": quote["change_percent"],
        "prior_20_high": prior_20_high,
        "prior_10_low": prior_10_low,
        "distance_to_high": distance_to_high,
        "volume_ratio": volume_ratio,
        "rsi": rsi,
        "ma20": ma20,
        "ma60": ma60,
        "score": score,
        "reasons": reasons
    }


def send_telegram_message(text):
    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": text,
            "disable_web_page_preview": True
        },
        timeout=30
    )

    print(
        "Telegram:",
        response.status_code,
        response.text
    )

    response.raise_for_status()


def can_send_alert(result, state):
    symbol = result["symbol"]
    stage = result["stage"]

    old = state.get(symbol)

    if not old:
        return True

    if old.get("stage") != stage:
        return True

    last_time_text = old.get("time")

    if not last_time_text:
        return True

    try:
        last_time = datetime.fromisoformat(
            last_time_text
        )

        now = datetime.now(timezone.utc)

        elapsed_hours = (
            now - last_time
        ).total_seconds() / 3600

        return elapsed_hours >= ALERT_COOLDOWN_HOURS

    except (TypeError, ValueError):
        return True


def update_state(result, state):
    state[result["symbol"]] = {
        "stage": result["stage"],
        "price": result["price"],
        "time": datetime.now(
            timezone.utc
        ).isoformat()
    }


def format_alert(result):
    stage = result["stage"]

    if stage == "BUY1":
        icon = "🚨"
        title = "미국주식 BUY1 발생"
    elif stage == "SELL":
        icon = "⚠️"
        title = "미국주식 SELL 발생"
    else:
        icon = "👀"
        title = "미국주식 돌파 준비"

    reason_text = "\n".join(
        f"• {reason}"
        for reason in result["reasons"]
    )

    return (
        f"{icon} {title}\n\n"
        f"종목: {result['symbol']}\n"
        f"현재가: ${result['price']:.2f}\n"
        f"등락률: {result['change_percent']:+.2f}%\n\n"
        f"{reason_text}\n\n"
        f"20일 고점: ${result['prior_20_high']:.2f}\n"
        f"10일 저점: ${result['prior_10_low']:.2f}\n"
        f"거래량: {result['volume_ratio']:.2f}배\n"
        f"신뢰도: {result['score']}점"
    )


def main():
    print("US STOCK SCANNER V1 START")

    state = load_state()
    results = []

    for index, symbol in enumerate(WATCHLIST, start=1):
        try:
            print(
                f"[{index}/{len(WATCHLIST)}] "
                f"{symbol} 분석 중"
            )

            result = analyze_symbol(symbol)

            if result:
                print(
                    symbol,
                    result["stage"],
                    result["price"]
                )

                results.append(result)

        except Exception as error:
            print(
                f"{symbol} 분석 오류: {error}"
            )

        # Finnhub 무료 API 호출 과속 방지
        time.sleep(1.1)

    stage_rank = {
        "BUY1": 3,
        "준비": 2,
        "SELL": 1
    }

    results.sort(
        key=lambda item: (
            stage_rank.get(
                item["stage"],
                0
            ),
            item["score"],
            item["volume_ratio"]
        ),
        reverse=True
    )

    sent_count = 0

    for result in results:
        if not can_send_alert(result, state):
            print(
                result["symbol"],
                "중복 알림 생략"
            )
            continue

        message = format_alert(result)
        send_telegram_message(message)

        update_state(result, state)
        sent_count += 1

        time.sleep(1)

    save_state(state)

    print(
        f"분석 완료: 신호 {len(results)}개, "
        f"전송 {sent_count}개"
    )


if __name__ == "__main__":
    main()