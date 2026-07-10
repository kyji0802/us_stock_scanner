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


# =========================================================
# 환경변수
# =========================================================
BOT_TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]
FINNHUB_API_KEY = os.environ["FINNHUB_API_KEY"]


# =========================================================
# 파일
# =========================================================
STATE_FILE = Path("signal_state.json")


# =========================================================
# 검색 설정
# =========================================================
MAX_SCAN = 1000
BATCH_SIZE = 80

MIN_PRICE = 5.0
MAX_PRICE = 100.0
MIN_AVG_VOLUME = 1_000_000

# 이전 20일 고점까지 2% 이내면 준비
READY_DISTANCE_PERCENT = 2.0

# 한 번 실행할 때 최대 알림 종목 수
MAX_ALERTS_PER_RUN = 30

# 터틀 설정
ATR_PERIOD = 20
ADD_UNIT_ATR = 0.5
STOP_ATR = 2.0
MAX_UNITS = 4


NASDAQ_LIST_URL = (
    "https://www.nasdaqtrader.com/dynamic/"
    "SymDir/nasdaqlisted.txt"
)

OTHER_LIST_URL = (
    "https://www.nasdaqtrader.com/dynamic/"
    "SymDir/otherlisted.txt"
)


# =========================================================
# 상태 파일
# =========================================================
def load_state():
    if not STATE_FILE.exists():
        return {}

    try:
        with open(STATE_FILE, "r", encoding="utf-8") as file:
            state = json.load(file)

        if isinstance(state, dict):
            return state

    except (OSError, json.JSONDecodeError) as error:
        print("상태 파일 읽기 오류:", error)

    return {}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as file:
        json.dump(
            state,
            file,
            ensure_ascii=False,
            indent=2
        )


def get_symbol_state(state, symbol):
    symbol_state = state.get(symbol)

    if not isinstance(symbol_state, dict):
        symbol_state = {}

    symbol_state.setdefault("last_alert_stage", "")
    symbol_state.setdefault("last_alert_time", "")
    symbol_state.setdefault("position", None)

    return symbol_state


def save_symbol_state(state, symbol, symbol_state):
    state[symbol] = symbol_state


# =========================================================
# 텔레그램
# =========================================================
def send_telegram_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": CHAT_ID,
            "text": text,
            "disable_web_page_preview": True
        },
        timeout=30
    )

    print("Telegram:", response.status_code, response.text)
    response.raise_for_status()


# =========================================================
# 종목 목록
# =========================================================
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

    # BRK.B 형식을 yfinance용 BRK-B로 변경
    symbol = symbol.replace(".", "-")

    blocked_characters = ["$", "^", "/", "\\", " "]

    if any(character in symbol for character in blocked_characters):
        return None

    if len(symbol) > 8:
        return None

    return symbol


def bad_security_name(name):
    name = str(name).upper()

    blocked_words = [
        "WARRANT",
        "RIGHT",
        "RIGHTS",
        "UNIT",
        "UNITS",
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
        symbol = clean_symbol(row.get("Symbol", ""))

        if not symbol:
            continue

        if str(row.get("Test Issue", "N")).upper() == "Y":
            continue

        if str(row.get("ETF", "N")).upper() == "Y":
            continue

        if bad_security_name(row.get("Security Name", "")):
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
        symbol = clean_symbol(row.get("ACT Symbol", ""))

        if not symbol:
            continue

        if str(row.get("Test Issue", "N")).upper() == "Y":
            continue

        if str(row.get("ETF", "N")).upper() == "Y":
            continue

        if bad_security_name(row.get("Security Name", "")):
            continue

        symbols.append(symbol)

    return symbols


def symbol_sort_key(symbol):
    return hashlib.sha256(
        symbol.encode("utf-8")
    ).hexdigest()


def get_stock_universe(state):
    symbols = []

    try:
        symbols.extend(get_nasdaq_symbols())
    except Exception as error:
        print("NASDAQ 목록 오류:", error)

    try:
        symbols.extend(get_other_symbols())
    except Exception as error:
        print("NYSE/AMEX 목록 오류:", error)

    symbols = sorted(
        set(symbols),
        key=symbol_sort_key
    )

    if not symbols:
        raise RuntimeError(
            "미국 종목 목록을 가져오지 못했습니다."
        )

    selected_symbols = symbols[:MAX_SCAN]

    # 기존 BUY1~BUY4 상태 종목은 1,000개 목록에서 빠져도 계속 감시
    position_symbols = []

    for symbol, symbol_state in state.items():
        if (
            isinstance(symbol_state, dict)
            and symbol_state.get("position")
        ):
            position_symbols.append(symbol)

    return list(
        dict.fromkeys(
            selected_symbols + position_symbols
        )
    )


# =========================================================
# 보조 지표
# =========================================================
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

    rsi = 100 - (100 / (1 + rs))
    value = rsi.iloc[-1]

    if pd.isna(value):
        return 50.0

    return float(value)


def calculate_atr(data, period=20):
    if data.empty or len(data) < period + 1:
        return 0.0

    high = data["High"].astype(float)
    low = data["Low"].astype(float)
    close = data["Close"].astype(float)

    previous_close = close.shift(1)

    true_range = pd.concat(
        [
            high - low,
            (high - previous_close).abs(),
            (low - previous_close).abs()
        ],
        axis=1
    ).max(axis=1)

    # Wilder 방식 ATR
    atr_series = true_range.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False
    ).mean()

    atr_value = atr_series.iloc[-1]

    if pd.isna(atr_value) or atr_value <= 0:
        return 0.0

    return float(atr_value)


# =========================================================
# yfinance 일봉 다운로드
# =========================================================
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


def extract_symbol_data(downloaded, symbol, symbol_count):
    if downloaded.empty:
        return pd.DataFrame()

    try:
        if symbol_count == 1:
            frame = downloaded.copy()

        elif isinstance(downloaded.columns, pd.MultiIndex):
            first_level = downloaded.columns.get_level_values(0)
            second_level = downloaded.columns.get_level_values(1)

            if symbol in first_level:
                frame = downloaded[symbol].copy()

            elif symbol in second_level:
                frame = downloaded.xs(
                    symbol,
                    axis=1,
                    level=1
                ).copy()

            else:
                return pd.DataFrame()

        else:
            return pd.DataFrame()

        required_columns = [
            "Open",
            "High",
            "Low",
            "Close",
            "Volume"
        ]

        if not all(
            column in frame.columns
            for column in required_columns
        ):
            return pd.DataFrame()

        frame = frame.dropna(
            subset=required_columns
        )

        return frame

    except Exception as error:
        print(symbol, "데이터 추출 오류:", error)
        return pd.DataFrame()


# =========================================================
# Finnhub 실시간 가격
# =========================================================
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
        "change": float(data.get("d", 0) or 0),
        "change_percent": float(data.get("dp", 0) or 0),
        "previous_close": float(data.get("pc", 0) or 0)
    }


# =========================================================
# 일봉 1차 분석
# =========================================================
def pre_analyze_symbol(symbol, data, symbol_state):
    if data.empty or len(data) < 70:
        return None

    close = data["Close"].astype(float)
    volume = data["Volume"].astype(float)

    daily_price = float(close.iloc[-1])
    position = symbol_state.get("position")

    # 신규 종목에만 가격 제한 적용
    # 이미 BUY1 이상인 종목은 가격이 범위를 벗어나도 계속 감시
    if not position:
        if (
            daily_price < MIN_PRICE
            or daily_price > MAX_PRICE
        ):
            return None

    ma20 = float(close.tail(20).mean())
    ma60 = float(close.tail(60).mean())

    # 고점과 저점 계산에서는 현재 진행 중인 일봉 제외
    prior_data = data.iloc[:-1]

    if len(prior_data) < 60:
        return None

    prior_20_high = float(
        prior_data["High"].tail(20).max()
    )

    prior_10_low = float(
        prior_data["Low"].tail(10).min()
    )

    average_volume_20 = float(
        prior_data["Volume"].tail(20).mean()
    )

    current_volume = float(volume.iloc[-1])

    if average_volume_20 <= 0:
        return None

    volume_ratio = (
        current_volume / average_volume_20
    )

    rsi = calculate_rsi(close)

    # ATR은 현재까지 들어온 최신 일봉을 포함해 매번 새로 계산
    current_atr = calculate_atr(
        data,
        ATR_PERIOD
    )

    if current_atr <= 0:
        return None

    trend_ok = ma20 > ma60

    distance_to_high = (
        (prior_20_high - daily_price)
        / prior_20_high
        * 100
    )

    # 이미 진입 상태인 종목은 무조건 계속 확인
    if position:
        return {
            "symbol": symbol,
            "daily_price": daily_price,
            "prior_20_high": prior_20_high,
            "prior_10_low": prior_10_low,
            "ma20": ma20,
            "ma60": ma60,
            "trend_ok": trend_ok,
            "rsi": rsi,
            "atr": current_atr,
            "volume_ratio": volume_ratio
        }

    # 신규 진입 후보는 평균 거래량 조건 적용
    if average_volume_20 < MIN_AVG_VOLUME:
        return None

    near_high = (
        distance_to_high
        <= READY_DISTANCE_PERCENT
    )

    if not near_high:
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
        "atr": current_atr,
        "volume_ratio": volume_ratio
    }


# =========================================================
# 내부 점수
# 텔레그램에는 표시하지 않음
# =========================================================
def calculate_internal_score(
    stage,
    trend_ok,
    rsi,
    volume_ratio
):
    score = 0

    if stage == "BUY1":
        score += 45

    elif stage in {"BUY2", "BUY3", "BUY4"}:
        score += 50

    elif stage == "준비":
        score += 30

    elif stage in {"ATR SELL", "10D SELL"}:
        score += 40

    if trend_ok:
        score += 20

    if 50 <= rsi <= 70:
        score += 15

    elif (
        stage in {"ATR SELL", "10D SELL"}
        and rsi < 45
    ):
        score += 15

    if volume_ratio >= 2:
        score += 20

    elif volume_ratio >= 1:
        score += 10

    return min(score, 100)


# =========================================================
# 최종 신호 판정
# =========================================================
def finalize_signal(candidate, symbol_state):
    symbol = candidate["symbol"]
    quote = get_finnhub_quote(symbol)

    current_price = quote["price"]
    previous_close = quote["previous_close"]

    prior_20_high = candidate["prior_20_high"]
    prior_10_low = candidate["prior_10_low"]

    trend_ok = candidate["trend_ok"]
    rsi = candidate["rsi"]

    # 매 실행 시 새로 계산된 현재 ATR
    current_atr = candidate["atr"]

    volume_ratio = candidate["volume_ratio"]
    position = symbol_state.get("position")

    # =====================================================
    # 이미 BUY1 이상 진입한 상태
    # =====================================================
    if position:
        units = int(
            position.get("units", 1)
        )

        first_entry = float(
            position.get(
                "first_entry",
                current_price
            )
        )

        last_entry = float(
            position.get(
                "last_entry",
                first_entry
            )
        )

        old_stop = float(
            position.get("atr_stop", 0) or 0
        )

        # 최신 ATR 기준 손절가
        new_calculated_stop = (
            last_entry
            - STOP_ATR * current_atr
        )

        # 손절가는 아래로 내려가지 않음
        if old_stop > 0:
            atr_stop_price = max(
                old_stop,
                new_calculated_stop
            )
        else:
            atr_stop_price = (
                new_calculated_stop
            )

        # 최신 ATR과 손절가를 상태에 계속 반영
        position["atr"] = current_atr
        position["atr_stop"] = atr_stop_price
        symbol_state["position"] = position

        # ATR 손절
        if current_price <= atr_stop_price:
            stage = "ATR SELL"

            return {
                "symbol": symbol,
                "stage": stage,
                "price": current_price,
                "internal_score": calculate_internal_score(
                    stage,
                    trend_ok,
                    rsi,
                    volume_ratio
                ),
                "volume_ratio": volume_ratio,
                "new_position": None
            }

        # 10일 저점 이탈
        if current_price < prior_10_low:
            stage = "10D SELL"

            return {
                "symbol": symbol,
                "stage": stage,
                "price": current_price,
                "internal_score": calculate_internal_score(
                    stage,
                    trend_ok,
                    rsi,
                    volume_ratio
                ),
                "volume_ratio": volume_ratio,
                "new_position": None
            }

        # 최신 ATR을 이용한 다음 추가 진입 가격
        next_add_price = (
            last_entry
            + ADD_UNIT_ATR * current_atr
        )

        if (
            units < MAX_UNITS
            and current_price >= next_add_price
        ):
            new_units = units + 1
            stage = f"BUY{new_units}"

            # 추가 진입 후 최신 진입가 기준 새 손절가
            added_unit_stop = (
                current_price
                - STOP_ATR * current_atr
            )

            # 손절가는 기존보다 낮아지지 않음
            updated_stop = max(
                atr_stop_price,
                added_unit_stop
            )

            new_position = {
                "units": new_units,
                "first_entry": first_entry,
                "last_entry": current_price,
                "atr": current_atr,
                "atr_stop": updated_stop
            }

            return {
                "symbol": symbol,
                "stage": stage,
                "price": current_price,
                "internal_score": calculate_internal_score(
                    stage,
                    trend_ok,
                    rsi,
                    volume_ratio
                ),
                "volume_ratio": volume_ratio,
                "new_position": new_position
            }

        return None

    # =====================================================
    # 아직 진입하지 않은 신규 종목
    # =====================================================
    if (
        current_price < MIN_PRICE
        or current_price > MAX_PRICE
    ):
        return None

    distance_to_high = (
        (prior_20_high - current_price)
        / prior_20_high
        * 100
    )

    buy1 = (
        previous_close <= prior_20_high
        and current_price > prior_20_high
        and trend_ok
    )

    ready = (
        not buy1
        and 0 <= distance_to_high <= READY_DISTANCE_PERCENT
        and trend_ok
    )

    if buy1:
        stage = "BUY1"

        initial_stop = (
            current_price
            - STOP_ATR * current_atr
        )

        new_position = {
            "units": 1,
            "first_entry": current_price,
            "last_entry": current_price,
            "atr": current_atr,
            "atr_stop": initial_stop
        }

    elif ready:
        stage = "준비"
        new_position = None

    else:
        return None

    return {
        "symbol": symbol,
        "stage": stage,
        "price": current_price,
        "internal_score": calculate_internal_score(
            stage,
            trend_ok,
            rsi,
            volume_ratio
        ),
        "volume_ratio": volume_ratio,
        "new_position": new_position
    }


# =========================================================
# 중복 알림
# 같은 단계는 다시 보내지 않음
# =========================================================
def can_send_alert(result, symbol_state):
    old_stage = symbol_state.get(
        "last_alert_stage",
        ""
    )

    return old_stage != result["stage"]


def apply_result_to_state(result, symbol_state):
    symbol_state["last_alert_stage"] = (
        result["stage"]
    )

    symbol_state["last_alert_time"] = (
        datetime.now(timezone.utc).isoformat()
    )

    if result["stage"] in {
        "ATR SELL",
        "10D SELL"
    }:
        symbol_state["position"] = None

    elif result.get("new_position") is not None:
        symbol_state["position"] = (
            result["new_position"]
        )

    return symbol_state


# =========================================================
# 텔레그램 묶음 메시지
# =========================================================
def format_group_message(results):
    now_kst = datetime.now(
        ZoneInfo("Asia/Seoul")
    )

    sections = [
        ("🚨 BUY1", "BUY1"),
        ("🚨 BUY2", "BUY2"),
        ("🚨 BUY3", "BUY3"),
        ("🚨 BUY4", "BUY4"),
        ("👀 준비", "준비"),
        ("⚠️ ATR SELL", "ATR SELL"),
        ("⚠️ 10D SELL", "10D SELL")
    ]

    lines = [
        "🇺🇸 US Stock Scanner",
        now_kst.strftime("%Y-%m-%d %H:%M")
    ]

    for title, stage in sections:
        stage_items = [
            item
            for item in results
            if item["stage"] == stage
        ]

        if not stage_items:
            continue

        lines.append("")
        lines.append(
            f"{title} ({len(stage_items)})"
        )

        for item in stage_items:
            lines.append(
                f"{item['symbol']}  "
                f"${item['price']:.2f}"
            )

    return "\n".join(lines)


# =========================================================
# 배치 분석
# =========================================================
def scan_batch(symbols, downloaded, state):
    candidates = []

    for symbol in symbols:
        try:
            frame = extract_symbol_data(
                downloaded,
                symbol,
                len(symbols)
            )

            symbol_state = get_symbol_state(
                state,
                symbol
            )

            candidate = pre_analyze_symbol(
                symbol,
                frame,
                symbol_state
            )

            if candidate:
                candidates.append(candidate)

        except Exception as error:
            print(
                symbol,
                "일봉 분석 오류:",
                error
            )

    return candidates


# =========================================================
# 메인
# =========================================================
def main():
    print("US TURTLE SCANNER V4 START")

    state = load_state()
    symbols = get_stock_universe(state)

    print(f"검색 종목 수: {len(symbols)}개")

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

        print(
            f"배치 {batch_number}/"
            f"{total_batches} 다운로드"
        )

        try:
            downloaded = download_batch(
                batch_symbols
            )

            candidates = scan_batch(
                batch_symbols,
                downloaded,
                state
            )

            all_candidates.extend(candidates)

        except Exception as error:
            print("배치 오류:", error)

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

            symbol_state = get_symbol_state(
                state,
                symbol
            )

            result = finalize_signal(
                candidate,
                symbol_state
            )

            # ATR이나 손절가가 갱신된 상태도 저장
            save_symbol_state(
                state,
                symbol,
                symbol_state
            )

            if not result:
                continue

            if not can_send_alert(
                result,
                symbol_state
            ):
                print(
                    symbol,
                    result["stage"],
                    "중복 알림 생략"
                )
                continue

            results.append(result)

        except Exception as error:
            print(
                symbol,
                "실시간 분석 오류:",
                error
            )

        # Finnhub 무료 API 호출 제한 방지
        time.sleep(1.1)

    stage_rank = {
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
            stage_rank.get(
                item["stage"],
                0
            ),
            item["internal_score"],
            item["volume_ratio"]
        ),
        reverse=True
    )

    results = results[:MAX_ALERTS_PER_RUN]

    if results:
        for result in results:
            symbol = result["symbol"]

            symbol_state = get_symbol_state(
                state,
                symbol
            )

            symbol_state = apply_result_to_state(
                result,
                symbol_state
            )

            save_symbol_state(
                state,
                symbol,
                symbol_state
            )

        message = format_group_message(results)
        send_telegram_message(message)

        print(
            f"텔레그램 묶음 전송: "
            f"{len(results)}개"
        )

    else:
        print("새로운 신호 없음")

    # 알림이 없어도 최신 ATR과 손절가 저장
    save_state(state)

    print(
        f"분석 완료: "
        f"신규 신호 {len(results)}개"
    )


if __name__ == "__main__":
    main()