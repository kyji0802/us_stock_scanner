import hashlib
import json
import os
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]

STATE_FILE = Path("signal_state.json")

MAX_SCAN = 1000
BATCH_SIZE = 80
MIN_PRICE = 5.0
MAX_PRICE = 100.0
MIN_AVG_VOLUME = 1_000_000
READY_DISTANCE = 2.0
MAX_ALERTS = 30

ATR_PERIOD = 20
ADD_ATR = 0.5
STOP_ATR = 2.0
MAX_UNITS = 4

NASDAQ_URL = (
    "https://www.nasdaqtrader.com/dynamic/"
    "SymDir/nasdaqlisted.txt"
)

OTHER_URL = (
    "https://www.nasdaqtrader.com/dynamic/"
    "SymDir/otherlisted.txt"
)

def load_state():
    if not STATE_FILE.exists():
        return {}

    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            state = json.load(f)

        return state if isinstance(state, dict) else {}

    except (OSError, json.JSONDecodeError):
        return {}


def save_state(state):
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            state,
            f,
            ensure_ascii=False,
            indent=2
        )


def symbol_state(state, symbol):
    value = state.get(symbol)

    if not isinstance(value, dict):
        value = {}

    value.setdefault("last_stage", "")
    value.setdefault("last_time", "")
    value.setdefault("position", None)

    return value


def send_message(text, results):
    keyboard = []

    for item in results:
        symbol = item["symbol"]

        chart_url = (
            "https://www.tradingview.com/chart/"
            f"?symbol={symbol}"
        )

        keyboard.append(
            [
                {
                    "text": f"📈 {symbol} 차트",
                    "url": chart_url
                }
            ]
        )

    response = requests.post(
        f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage",
        json={
            "chat_id": CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
            "reply_markup": {
                "inline_keyboard": keyboard
            }
        },
        timeout=30
    )

    print(
        "Telegram:",
        response.status_code,
        response.text
    )

    response.raise_for_status()

def clean_symbol(value):
    symbol = str(value).strip().upper().replace(".", "-")

    if not symbol:
        return None

    if symbol in {
        "SYMBOL",
        "ACT SYMBOL",
        "FILE CREATION TIME"
    }:
        return None

    if any(c in symbol for c in ["$", "^", "/", "\\", " "]):
        return None

    if len(symbol) > 8:
        return None

    return symbol


def bad_name(name):
    text = str(name).upper()

    words = [
        "WARRANT",
        "RIGHT",
        "UNIT",
        "PREFERRED",
        "PREFERENCE",
        "DEPOSITARY",
        "DEPOSITORY",
        "ACQUISITION",
        "SPAC",
        "ETF",
        "ETN",
        "FUND",
        "NOTE",
        "BOND"
    ]

    return any(word in text for word in words)


def read_symbol_file(url, symbol_column):
    response = requests.get(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )

    response.raise_for_status()

    df = pd.read_csv(
        StringIO(response.text),
        sep="|"
    )

    symbols = []

    for _, row in df.iterrows():
        symbol = clean_symbol(
            row.get(symbol_column, "")
        )

        if not symbol:
            continue

        if str(row.get("Test Issue", "N")).upper() == "Y":
            continue

        if str(row.get("ETF", "N")).upper() == "Y":
            continue

        if bad_name(row.get("Security Name", "")):
            continue

        symbols.append(symbol)

    return symbols


def get_universe(state):
    symbols = []

    try:
        symbols.extend(
            read_symbol_file(
                NASDAQ_URL,
                "Symbol"
            )
        )
    except Exception as error:
        print("NASDAQ 목록 오류:", error)

    try:
        symbols.extend(
            read_symbol_file(
                OTHER_URL,
                "ACT Symbol"
            )
        )
    except Exception as error:
        print("NYSE/AMEX 목록 오류:", error)

    if not symbols:
        raise RuntimeError("미국 종목 목록 조회 실패")

    symbols = sorted(
        set(symbols),
        key=lambda x: hashlib.sha256(
            x.encode("utf-8")
        ).hexdigest()
    )

    selected = symbols[:MAX_SCAN]

    held = [
        symbol
        for symbol, value in state.items()
        if isinstance(value, dict)
        and value.get("position")
    ]

    return list(
        dict.fromkeys(selected + held)
    )


def rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    avg_loss = loss.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    rs = avg_gain / avg_loss.replace(0, float("nan"))
    value = (100 - 100 / (1 + rs)).iloc[-1]

    return 50.0 if pd.isna(value) else float(value)


def atr(data, period=20):
    if len(data) < period + 1:
        return 0.0

    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)
    prev_close = close.shift(1)

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ],
        axis=1
    ).max(axis=1)

    value = tr.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean().iloc[-1]

    if pd.isna(value) or value <= 0:
        return 0.0

    return float(value)
def download_batch(symbols):
    return yf.download(
        tickers=symbols,
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
        timeout=90
    )


def extract_data(downloaded, symbol, count):
    if downloaded.empty:
        return pd.DataFrame()

    try:
        if count == 1:
            frame = downloaded.copy()

        elif isinstance(downloaded.columns, pd.MultiIndex):
            level0 = downloaded.columns.get_level_values(0)
            level1 = downloaded.columns.get_level_values(1)

            if symbol in level0:
                frame = downloaded[symbol].copy()

            elif symbol in level1:
                frame = downloaded.xs(
                    symbol,
                    axis=1,
                    level=1
                ).copy()

            else:
                return pd.DataFrame()

        else:
            return pd.DataFrame()

        columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(column in frame.columns for column in columns):
            return pd.DataFrame()

        return frame.dropna(subset=columns)

    except Exception:
        return pd.DataFrame()


def quote(symbol):
    response = requests.get(
        "https://finnhub.io/api/v1/quote",
        params={
            "symbol": symbol,
            "token": FINNHUB_API_KEY
        },
        timeout=20
    )

    response.raise_for_status()
    data = response.json()

    price = float(data.get("c", 0) or 0)

    if price <= 0:
        raise RuntimeError(
            f"{symbol} 현재가 조회 실패: {data}"
        )

    return {
        "price": price,
        "previous_close": float(
            data.get("pc", 0) or 0
        )
    }


def pre_analyze(symbol, data, state_item):
    if data.empty or len(data) < 70:
        return None

    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)
    daily_price = float(close.iloc[-1])
    position = state_item.get("position")

    if not position and not (
        MIN_PRICE <= daily_price <= MAX_PRICE
    ):
        return None

    prior = data.iloc[:-1]

    if len(prior) < 60:
        return None

    high20 = float(
        prior["High"].tail(20).max()
    )

    low10 = float(
        prior["Low"].tail(10).min()
    )

    avg_volume = float(
        prior["Volume"].tail(20).mean()
    )

    if avg_volume <= 0:
        return None

    volume_ratio = float(
        volume.iloc[-1]
    ) / avg_volume

    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())
    current_atr = atr(data, ATR_PERIOD)

    if current_atr <= 0:
        return None

    result = {
        "symbol": symbol,
        "high20": high20,
        "low10": low10,
        "trend": ma20 > ma60,
        "rsi": rsi(close),
        "atr": current_atr,
        "volume_ratio": volume_ratio
    }

    if position:
        return result

    if avg_volume < MIN_AVG_VOLUME:
        return None

    distance = (
        (high20 - daily_price)
        / high20
        * 100
    )

    if distance > READY_DISTANCE:
        return None

    return result


def score(stage, trend, rsi_value, volume_ratio):
    value = {
        "BUY1": 45,
        "BUY2": 50,
        "BUY3": 50,
        "BUY4": 50,
        "준비": 30,
        "ATR SELL": 40,
        "10D SELL": 40
    }.get(stage, 0)

    if trend:
        value += 20

    if 50 <= rsi_value <= 70:
        value += 15

    if stage in {"ATR SELL", "10D SELL"} and rsi_value < 45:
        value += 15

    if volume_ratio >= 2:
        value += 20
    elif volume_ratio >= 1:
        value += 10

    return min(value, 100)


def make_result(candidate, stage, price, new_position):
    return {
        "symbol": candidate["symbol"],
        "stage": stage,
        "price": price,
        "score": score(
            stage,
            candidate["trend"],
            candidate["rsi"],
            candidate["volume_ratio"]
        ),
        "volume_ratio": candidate["volume_ratio"],
        "new_position": new_position
    }


def finalize(candidate, state_item):
    live = quote(candidate["symbol"])
    price = live["price"]
    previous_close = live["previous_close"]
    current_atr = candidate["atr"]
    position = state_item.get("position")

    if position:
        units = int(position.get("units", 1))
        first_entry = float(
            position.get("first_entry", price)
        )
        last_entry = float(
            position.get("last_entry", first_entry)
        )
        old_stop = float(
            position.get("atr_stop", 0) or 0
        )

        calculated_stop = (
            last_entry - STOP_ATR * current_atr
        )

        stop_price = (
            max(old_stop, calculated_stop)
            if old_stop > 0
            else calculated_stop
        )

        position["atr"] = current_atr
        position["atr_stop"] = stop_price
        state_item["position"] = position

        if price <= stop_price:
            return make_result(
                candidate,
                "ATR SELL",
                price,
                None
            )

        if price < candidate["low10"]:
            return make_result(
                candidate,
                "10D SELL",
                price,
                None
            )

        next_price = (
            last_entry + ADD_ATR * current_atr
        )

        if units < MAX_UNITS and price >= next_price:
            new_units = units + 1
            stage = f"BUY{new_units}"

            new_stop = max(
                stop_price,
                price - STOP_ATR * current_atr
            )

            new_position = {
                "units": new_units,
                "first_entry": first_entry,
                "last_entry": price,
                "atr": current_atr,
                "atr_stop": new_stop
            }

            return make_result(
                candidate,
                stage,
                price,
                new_position
            )

        return None

    if not MIN_PRICE <= price <= MAX_PRICE:
        return None

    distance = (
        (candidate["high20"] - price)
        / candidate["high20"]
        * 100
    )

    buy1 = (
        previous_close <= candidate["high20"]
        and price > candidate["high20"]
        and candidate["trend"]
    )

    ready = (
        not buy1
        and 0 <= distance <= READY_DISTANCE
        and candidate["trend"]
    )

    if buy1:
        new_position = {
            "units": 1,
            "first_entry": price,
            "last_entry": price,
            "atr": current_atr,
            "atr_stop": price - STOP_ATR * current_atr
        }

        return make_result(
            candidate,
            "BUY1",
            price,
            new_position
        )

    if ready:
        return make_result(
            candidate,
            "준비",
            price,
            None
        )

    return None
def is_new_stage(result, state_item):
    return state_item.get("last_stage", "") != result["stage"]


def apply_result(result, state_item):
    state_item["last_stage"] = result["stage"]
    state_item["last_time"] = datetime.now(
        timezone.utc
    ).isoformat()

    if result["stage"] in {
        "ATR SELL",
        "10D SELL"
    }:
        state_item["position"] = None

    elif result["new_position"] is not None:
        state_item["position"] = result["new_position"]

    return state_item


def format_message(results):
    sections = [
        ("🚨 BUY1", "BUY1"),
        ("🚨 BUY2", "BUY2"),
        ("🚨 BUY3", "BUY3"),
        ("🚨 BUY4", "BUY4"),
        ("👀 준비", "준비"),
        ("⚠️ ATR SELL", "ATR SELL"),
        ("⚠️ 10D SELL", "10D SELL")
    ]

    now = datetime.now(
        ZoneInfo("Asia/Seoul")
    )

    lines = [
        "🇺🇸 US Stock Scanner",
        now.strftime("%Y-%m-%d %H:%M")
    ]

    for title, stage in sections:
        items = [
            item
            for item in results
            if item["stage"] == stage
        ]

        if not items:
            continue

        lines.extend(
            [
                "",
                f"{title} ({len(items)})"
            ]
        )

        lines.extend(
            f"{item['symbol']}  ${item['price']:.2f}"
            for item in items
        )

    return "\n".join(lines)


def main():
    print("US TURTLE SCANNER V5 START")

    state = load_state()
    symbols = get_universe(state)

    print("검색 종목 수:", len(symbols))

    candidates = []

    for start in range(
        0,
        len(symbols),
        BATCH_SIZE
    ):
        batch = symbols[
            start:start + BATCH_SIZE
        ]

        number = start // BATCH_SIZE + 1
        total = (
            len(symbols)
            + BATCH_SIZE
            - 1
        ) // BATCH_SIZE

        print(
            f"배치 {number}/{total} 다운로드"
        )

        try:
            downloaded = download_batch(batch)

            for symbol in batch:
                frame = extract_data(
                    downloaded,
                    symbol,
                    len(batch)
                )

                item = symbol_state(
                    state,
                    symbol
                )

                candidate = pre_analyze(
                    symbol,
                    frame,
                    item
                )

                if candidate:
                    candidates.append(candidate)

        except Exception as error:
            print("배치 오류:", error)

        time.sleep(2)

    print("실시간 확인 후보:", len(candidates))

    results = []

    for index, candidate in enumerate(
        candidates,
        start=1
    ):
        symbol = candidate["symbol"]
        item = symbol_state(state, symbol)

        try:
            print(
                f"[{index}/{len(candidates)}] "
                f"{symbol}"
            )

            result = finalize(
                candidate,
                item
            )

            state[symbol] = item

            if not result:
                continue

            if not is_new_stage(
                result,
                item
            ):
                print(
                    symbol,
                    result["stage"],
                    "중복 생략"
                )
                continue

            results.append(result)

        except Exception as error:
            print(
                symbol,
                "실시간 분석 오류:",
                error
            )

        time.sleep(1.1)

    ranks = {
        "BUY4": 7,
        "BUY3": 6,
        "BUY2": 5,
        "BUY1": 4,
        "준비": 3,
        "ATR SELL": 2,
        "10D SELL": 1
    }

    results.sort(
        key=lambda item: (
            ranks.get(item["stage"], 0),
            item["score"],
            item["volume_ratio"]
        ),
        reverse=True
    )

    results = results[:MAX_ALERTS]

    if results:
        for result in results:
            symbol = result["symbol"]
            item = symbol_state(state, symbol)

            state[symbol] = apply_result(
                result,
                item
            )

        send_message(
            format_message(results),
            results
        )

        print(
            "텔레그램 전송:",
            len(results)
        )

    else:
        print("새로운 신호 없음")

    save_state(state)

    print(
        "분석 완료:",
        len(results)
    )


if __name__ == "__main__":
    main()