import hashlib
import json
import os
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf


BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]

STATE_FILE = Path("signal_state.json")

# 검색 설정
MAX_SCAN = 1000
BATCH_SIZE = 80

MIN_PRICE = 5.0
MAX_PRICE = 100.0

MIN_AVG_VOLUME = 1_000_000
ALERT_COOLDOWN_HOURS = 12

# 20일 고점까지 2% 이내면 준비
READY_DISTANCE_PERCENT = 2.0

# 한 번 실행할 때 텔레그램 최대 전송 개수
MAX_ALERTS_PER_RUN = 15

NASDAQ_LIST_URL = (
    "https://www.nasdaqtrader.com/dynamic/"
    "SymDir/nasdaqlisted.txt"
)

OTHER_LIST_URL = (
    "https://www.nasdaqtrader.com/dynamic/"
    "SymDir/otherlisted.txt"
)


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


def clean_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if not symbol:
        return None

    if symbol in {
        "FILE CREATION TIME",
        "SYMBOL",
        "ACT SYMBOL"
    }:
        return None

    # yfinance 표기 방식
    symbol = symbol.replace(".", "-")

    blocked_chars = [
        "$",
        "^",
        "/",
        "\\",
        " "
    ]

    if any(char in symbol for char in blocked_chars):
        return None

    if len(symbol) > 8:
        return None

    return symbol


def bad_security_name(name):
    name = str(name).upper()

    blocked_words = [
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

    return any(word in name for word in blocked_words)


def get_nasdaq_symbols():
    response = requests.get(
        NASDAQ_LIST_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )

    response.raise_for_status()

    data = pd.read_csv(
        StringIO(response.text),
        sep="|"
    )

    symbols = []

    for _, row in data.iterrows():
        symbol = clean_symbol(
            row.get("Symbol", "")
        )

        if not symbol:
            continue

        if str(
            row.get("Test Issue", "N")
        ).upper() == "Y":
            continue

        if str(
            row.get("ETF", "N")
        ).upper() == "Y":
            continue

        if bad_security_name(
            row.get("Security Name", "")
        ):
            continue

        symbols.append(symbol)

    return symbols


def get_other_symbols():
    response = requests.get(
        OTHER_LIST_URL,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=30
    )

    response.raise_for_status()

    data = pd.read_csv(
        StringIO(response.text),
        sep="|"
    )

    symbols = []

    for _, row in data.iterrows():
        symbol = clean_symbol(
            row.get("ACT Symbol", "")
        )

        if not symbol:
            continue

        if str(
            row.get("Test Issue", "N")
        ).upper() == "Y":
            continue

        if str(
            row.get("ETF", "N")
        ).upper() == "Y":
            continue

        if bad_security_name(
            row.get("Security Name", "")
        ):
            continue

        symbols.append(symbol)

    return symbols


def symbol_sort_key(symbol):
    digest = hashlib.sha256(
        symbol.encode("utf-8")
    ).hexdigest()

    return digest


def get_stock_universe():
    symbols = []

    try:
        symbols.extend(
            get_nasdaq_symbols()
        )
    except Exception as error:
        print(
            "NASDAQ 종목 목록 오류:",
            error
        )

    try:
        symbols.extend(
            get_other_symbols()
        )
    except Exception as error:
        print(
            "NYSE/AMEX 종목 목록 오류:",
            error
        )

    symbols = sorted(
        set(symbols),
        key=symbol_sort_key
    )

    if not symbols:
        raise RuntimeError(
            "미국 종목 목록을 가져오지 못했습니다."
        )

    return symbols[:MAX_SCAN]


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

    rs = average_gain / average_loss.replace(
        0,
        float("nan")
    )

    rsi = 100 - (
        100 / (1 + rs)
    )

    value = rsi.iloc[-1]

    if pd.isna(value):
        return 50.0

    return float(value)


def download_batch(symbols):
    data = yf.download(
        tickers=symbols,
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        threads=True,
        progress=False,
        timeout=60
    )

    return data


def extract_symbol_data(
    downloaded,
    symbol,
    symbol_count
):
    if downloaded.empty:
        return pd.DataFrame()

    try:
        if symbol_count == 1:
            frame = downloaded.copy()

        elif isinstance(
            downloaded.columns,
            pd.MultiIndex
        ):
            first_level = (
                downloaded.columns
                .get_level_values(0)
            )

            if symbol in first_level:
                frame = downloaded[
                    symbol
                ].copy()
            else:
                return pd.DataFrame()

        else:
            return pd.DataFrame()

        required = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            column in frame.columns
            for column in required
        ):
            return pd.DataFrame()

        frame = frame.dropna(
            subset=required
        )

        return frame

    except Exception:
        return pd.DataFrame()


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

    current_price = float(
        data.get("c", 0) or 0
    )

    if current_price <= 0:
        raise RuntimeError(
            f"{symbol} 실시간 가격 없음: {data}"
        )

    return {
        "price": current_price,
        "change": float(
            data.get("d", 0) or 0
        ),
        "change_percent": float(
            data.get("dp", 0) or 0
        ),
        "previous_close": float(
            data.get("pc", 0) or 0
        )
    }


def pre_analyze_symbol(
    symbol,
    data
):
    if data.empty or len(data) < 65:
        return None

    close = data["Close"].astype(float)
    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    volume = data["Volume"].astype(float)

    daily_price = float(
        close.iloc[-1]
    )

    if (
        daily_price < MIN_PRICE
        or daily_price > MAX_PRICE
    ):
        return None

    ma20 = float(
        close.tail(20).mean()
    )

    ma60 = float(
        close.tail(60).mean()
    )

    prior_data = data.iloc[:-1]

    if len(prior_data) < 60:
        return None

    prior_20_high = float(
        prior_data["High"]
        .tail(20)
        .max()
    )

    prior_10_low = float(
        prior_data["Low"]
        .tail(10)
        .min()
    )

    average_volume_20 = float(
        prior_data["Volume"]
        .tail(20)
        .mean()
    )

    current_volume = float(
        volume.iloc[-1]
    )

    if average_volume_20 <= 0:
        return None

    if (
        average_volume_20
        < MIN_AVG_VOLUME
    ):
        return None

    volume_ratio = (
        current_volume
        / average_volume_20
    )

    rsi = calculate_rsi(close)

    distance = (
        (
            prior_20_high
            - daily_price
        )
        / prior_20_high
        * 100
    )

    trend_ok = ma20 > ma60

    near_high = (
        distance
        <= READY_DISTANCE_PERCENT
    )

    near_low = (
        daily_price
        <= prior_10_low * 1.02
    )

    # 일봉 기준으로 가능성 없는 종목은
    # Finnhub 실시간 조회 전에 제외
    if not near_high and not near_low:
        return None

    return {
        "symbol": symbol,
        "daily_price": daily_price,
        "prior_20_high": prior_20_high,
        "prior_10_low": prior_10_low,
        "ma20": ma20,
        "ma60": ma60,
        "trend_ok": trend_ok,
        "rsi": rsi,
        "volume_ratio": volume_ratio
    }


def finalize_signal(candidate):
    symbol = candidate["symbol"]

    quote = get_finnhub_quote(
        symbol
    )

    current_price = quote["price"]
    previous_close = quote[
        "previous_close"
    ]

    if (
        current_price < MIN_PRICE
        or current_price > MAX_PRICE
    ):
        return None

    prior_20_high = candidate[
        "prior_20_high"
    ]

    prior_10_low = candidate[
        "prior_10_low"
    ]

    trend_ok = candidate[
        "trend_ok"
    ]

    distance_to_high = (
        (
            prior_20_high
            - current_price
        )
        / prior_20_high
        * 100
    )

    buy1 = (
        previous_close
        <= prior_20_high
        and current_price
        > prior_20_high
        and trend_ok
    )

    ready = (
        not buy1
        and 0
        <= distance_to_high
        <= READY_DISTANCE_PERCENT
        and trend_ok
    )

    sell = (
        current_price
        < prior_10_low
    )

    if sell:
        stage = "SELL"

    elif buy1:
        stage = "BUY1"

    elif ready:
        stage = "준비"

    else:
        return None

    reasons = []

    if stage == "BUY1":
        reasons.append(
            "20일 고점 돌파"
        )

    elif stage == "준비":
        reasons.append(
            f"20일 고점까지 "
            f"{distance_to_high:.2f}%"
        )

    elif stage == "SELL":
        reasons.append(
            "10일 저점 이탈"
        )

    if trend_ok:
        reasons.append(
            "20일선 > 60일선"
        )

    else:
        reasons.append(
            "20일선 ≤ 60일선"
        )

    reasons.append(
        f"RSI {candidate['rsi']:.1f}"
    )

    # 내부 점수
    # 텔레그램에는 표시하지 않음
    internal_score = 0

    if stage == "BUY1":
        internal_score += 45

    elif stage == "준비":
        internal_score += 30

    elif stage == "SELL":
        internal_score += 40

    if trend_ok:
        internal_score += 20

    rsi = candidate["rsi"]

    if 50 <= rsi <= 70:
        internal_score += 15

    elif (
        stage == "SELL"
        and rsi < 45
    ):
        internal_score += 15

    volume_ratio = candidate[
        "volume_ratio"
    ]

    if volume_ratio >= 2:
        internal_score += 20

    elif volume_ratio >= 1:
        internal_score += 10

    internal_score = min(
        internal_score,
        100
    )

    return {
        "symbol": symbol,
        "stage": stage,
        "price": current_price,
        "change": quote["change"],
        "change_percent": quote[
            "change_percent"
        ],
        "prior_20_high": prior_20_high,
        "prior_10_low": prior_10_low,
        "distance_to_high": (
            distance_to_high
        ),
        "rsi": rsi,

        # 내부 계산용
        "volume_ratio": volume_ratio,
        "internal_score": (
            internal_score
        ),

        "reasons": reasons
    }


def can_send_alert(result, state):
    symbol = result["symbol"]
    stage = result["stage"]

    old = state.get(symbol)

    if not old:
        return True

    if old.get("stage") != stage:
        return True

    last_time_text = old.get(
        "time"
    )

    if not last_time_text:
        return True

    try:
        last_time = datetime.fromisoformat(
            last_time_text
        )

        now = datetime.now(
            timezone.utc
        )

        elapsed_hours = (
            now - last_time
        ).total_seconds() / 3600

        return (
            elapsed_hours
            >= ALERT_COOLDOWN_HOURS
        )

    except (
        TypeError,
        ValueError
    ):
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
        title = (
            "미국주식 BUY1 발생"
        )

    elif stage == "SELL":
        icon = "⚠️"
        title = (
            "미국주식 SELL 발생"
        )

    else:
        icon = "👀"
        title = (
            "미국주식 돌파 준비"
        )

    reason_text = "\n".join(
        f"• {reason}"
        for reason in result["reasons"]
    )

    return (
        f"{icon} {title}\n\n"
        f"종목: {result['symbol']}\n"
        f"현재가: "
        f"${result['price']:.2f}\n"
        f"등락률: "
        f"{result['change_percent']:+.2f}%"
        f"\n\n"
        f"{reason_text}\n\n"
        f"20일 고점: "
        f"${result['prior_20_high']:.2f}\n"
        f"10일 저점: "
        f"${result['prior_10_low']:.2f}"
    )


def scan_batch(
    symbols,
    batch_number,
    total_batches
):
    print(
        f"배치 {batch_number}/"
        f"{total_batches} 다운로드"
    )

    downloaded = download_batch(
        symbols
    )

    candidates = []

    for symbol in symbols:
        try:
            frame = extract_symbol_data(
                downloaded,
                symbol,
                len(symbols)
            )

            candidate = pre_analyze_symbol(
                symbol,
                frame
            )

            if candidate:
                candidates.append(
                    candidate
                )

        except Exception as error:
            print(
                symbol,
                "일봉 분석 오류:",
                error
            )

    return candidates


def main():
    print(
        "US STOCK SCANNER V2 START"
    )

    state = load_state()

    symbols = get_stock_universe()

    print(
        f"검색 종목 수: "
        f"{len(symbols)}개"
    )

    all_candidates = []

    total_batches = (
        len(symbols)
        + BATCH_SIZE
        - 1
    ) // BATCH_SIZE

    for start_index in range(
        0,
        len(symbols),
        BATCH_SIZE
    ):
        batch_symbols = symbols[
            start_index:
            start_index + BATCH_SIZE
        ]

        batch_number = (
            start_index // BATCH_SIZE
        ) + 1

        try:
            candidates = scan_batch(
                batch_symbols,
                batch_number,
                total_batches
            )

            all_candidates.extend(
                candidates
            )

        except Exception as error:
            print(
                "배치 다운로드 오류:",
                error
            )

        time.sleep(2)

    print(
        f"실시간 확인 후보: "
        f"{len(all_candidates)}개"
    )

    results = []

    for index, candidate in enumerate(
        all_candidates,
        start=1
    ):
        symbol = candidate["symbol"]

        try:
            print(
                f"[{index}/"
                f"{len(all_candidates)}] "
                f"{symbol} 실시간 확인"
            )

            result = finalize_signal(
                candidate
            )

            if result:
                results.append(
                    result
                )

        except Exception as error:
            print(
                symbol,
                "실시간 분석 오류:",
                error
            )

        # Finnhub 무료 API 과속 방지
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
            item[
                "internal_score"
            ],
            item[
                "volume_ratio"
            ]
        ),
        reverse=True
    )

    sent_count = 0

    for result in results:
        if (
            sent_count
            >= MAX_ALERTS_PER_RUN
        ):
            break

        if not can_send_alert(
            result,
            state
        ):
            print(
                result["symbol"],
                "중복 알림 생략"
            )
            continue

        message = format_alert(
            result
        )

        send_telegram_message(
            message
        )

        update_state(
            result,
            state
        )

        sent_count += 1

        time.sleep(1)

    save_state(state)

    print(
        f"분석 완료: "
        f"신호 {len(results)}개, "
        f"전송 {sent_count}개"
    )


if __name__ == "__main__":
    main()