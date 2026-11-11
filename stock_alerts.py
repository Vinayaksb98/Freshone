
import os
import time
from datetime import datetime
from io import StringIO
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

IST = ZoneInfo("Asia/Kolkata")

CSV_URL = (
    "https://nsearchives.nseindia.com/"
    "content/indices/ind_nifty500list.csv"
)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://www.niftyindices.com/",
}

BATCH_SIZE = 40
DOWNLOAD_PERIOD = "6mo"
DOWNLOAD_INTERVAL = "1d"
DOWNLOAD_RETRIES = 2
TELEGRAM_MESSAGE_LIMIT = 3500


# ============================================================
# TELEGRAM
# ============================================================

def get_telegram_credentials():

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        raise RuntimeError(
            "Missing GitHub Secrets: "
            "TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID"
        )

    return token, chat_id


def send_message(message, token, chat_id):

    url = f"https://api.telegram.org/bot{token}/sendMessage"

    response = requests.post(
        url,
        data={
            "chat_id": chat_id,
            "text": message
        },
        timeout=25
    )

    print("Telegram HTTP status:", response.status_code)

    response.raise_for_status()

    result = response.json()

    if not result.get("ok"):
        raise RuntimeError(
            f"Telegram rejected message: {result}"
        )

    print("Telegram message sent successfully.")


def send_long_message(message, token, chat_id):

    lines = message.splitlines()

    chunks = []
    current_message = ""

    for line in lines:

        candidate = (
            line if not current_message
            else current_message + "\n" + line
        )

        if len(candidate) > TELEGRAM_MESSAGE_LIMIT:

            if current_message:
                chunks.append(current_message)

            current_message = line

        else:
            current_message = candidate

    if current_message:
        chunks.append(current_message)

    for chunk in chunks:
        send_message(chunk, token, chat_id)


# ============================================================
# DOWNLOAD NIFTY 500 STOCK LIST
# ============================================================

def get_nifty500_tickers():

    response = requests.get(
        CSV_URL,
        headers=HEADERS,
        timeout=30
    )

    response.raise_for_status()

    data = pd.read_csv(StringIO(response.text))

    symbol_col = next(
        (
            column for column in data.columns
            if str(column).strip().lower()
            in ("symbol", "ticker")
        ),
        None
    )

    if symbol_col is None:
        raise ValueError(
            "NIFTY 500 symbol column not found. "
            f"Received columns: {list(data.columns)}"
        )

    symbols = (
        data[symbol_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    tickers = list(dict.fromkeys(
        symbol if symbol.endswith(".NS")
        else symbol + ".NS"
        for symbol in symbols
        if symbol and symbol.lower() != "nan"
    ))

    if len(tickers) < 400:
        raise ValueError(
            f"NIFTY 500 list incomplete. "
            f"Only {len(tickers)} stocks received."
        )

    print(f"Loaded NIFTY 500 stocks: {len(tickers)}")

    return tickers


# ============================================================
# YAHOO FINANCE BATCH DOWNLOAD
# ============================================================

def download_batch(tickers):

    for attempt in range(DOWNLOAD_RETRIES + 1):

        try:

            print(
                f"Yahoo download attempt "
                f"{attempt + 1}/{DOWNLOAD_RETRIES + 1}"
            )

            data = yf.download(
                tickers=tickers,
                period=DOWNLOAD_PERIOD,
                interval=DOWNLOAD_INTERVAL,
                group_by="ticker",
                auto_adjust=False,
                actions=False,
                progress=False,
                threads=True,
                timeout=30
            )

            if data is not None and not data.empty:
                return data

            print("Yahoo returned empty data.")

        except Exception as exc:

            print(
                "Yahoo download error:",
                type(exc).__name__,
                str(exc)
            )

        if attempt < DOWNLOAD_RETRIES:
            time.sleep(3 * (attempt + 1))

    return pd.DataFrame()


# ============================================================
# EXTRACT INDIVIDUAL STOCK DATA FROM BATCH
# ============================================================

def extract_ticker_frame(downloaded, ticker):

    if downloaded is None or downloaded.empty:
        return pd.DataFrame()

    if not isinstance(downloaded.columns, pd.MultiIndex):

        # Flat columns are valid when Yahoo returns one ticker.
        if len(downloaded.columns) > 0:
            return downloaded.copy()

        return pd.DataFrame()

    for level in range(downloaded.columns.nlevels):

        values = downloaded.columns.get_level_values(level)

        if ticker in values:

            try:

                frame = downloaded.xs(
                    ticker,
                    axis=1,
                    level=level,
                    drop_level=True
                )

                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = (
                        frame.columns.get_level_values(0)
                    )

                return frame.copy()

            except (KeyError, ValueError):
                continue

    return pd.DataFrame()


# ============================================================
# DAILY STRATEGY CALCULATION
# ============================================================

def calculate_match(ticker, df):

    if df is None or df.empty:
        return None

    df = df.copy()

    df.columns = [
        str(column).strip().title()
        for column in df.columns
    ]

    required_columns = {
        "Close",
        "High",
        "Volume"
    }

    if not required_columns.issubset(df.columns):
        return None

    df = df.sort_index()

    df = df[
        ~df.index.duplicated(keep="last")
    ]

    close = pd.to_numeric(
        df["Close"],
        errors="coerce"
    ).dropna()

    high = pd.to_numeric(
        df["High"],
        errors="coerce"
    ).reindex(close.index)

    volume = pd.to_numeric(
        df["Volume"],
        errors="coerce"
    ).reindex(close.index)

    if len(close) < 30:
        return None

    # EMA 10 and EMA 20

    ema10 = close.ewm(
        span=10,
        adjust=False
    ).mean()

    ema20 = close.ewm(
        span=20,
        adjust=False
    ).mean()

    # RSI 14

    delta = close.diff()

    gain = delta.where(
        delta > 0, 0
    ).rolling(14).mean()

    loss = (
        -delta.where(delta < 0, 0)
    ).rolling(14).mean()

    rs = gain / loss

    rsi = 100 - (
        100 / (1 + rs)
    )

    rsi = rsi.mask(
        (loss == 0) & (gain > 0),
        100
    )

    rsi = rsi.mask(
        (loss == 0) & (gain == 0),
        50
    )

    # Original 10-day volume average calculation

    volume_average = volume.rolling(10).mean()

    # Latest daily candle values

    values = [
        close.iloc[-1],
        high.iloc[-1],
        ema10.iloc[-1],
        ema20.iloc[-1],
        rsi.iloc[-1],
        volume.iloc[-1],
        volume_average.iloc[-1]
    ]

    if not all(pd.notna(value) for value in values):
        return None

    c, h, e10, e20, r, v, va = map(
        float,
        values
    )

    if va <= 0:
        return None

    # ========================================================
    # BUY SIGNAL CONDITIONS
    # ========================================================

    condition_1 = e10 > e20

    condition_2 = 55 <= r <= 68

    condition_3 = v > 1.5 * va

    condition_4 = c > e20

    if (
        condition_1
        and condition_2
        and condition_3
        and condition_4
    ):

        candle_date = close.index[-1]

        if hasattr(candle_date, "strftime"):
            candle_date = candle_date.strftime("%d-%m-%Y")
        else:
            candle_date = str(candle_date)

        risk = ((c - e20) / c) * 100

        match = (
            f"Stock: {ticker.replace('.NS', '')}\n"
            f"Daily Candle: {candle_date}\n"
            f"CMP: ₹{c:.2f}\n"
            f"Buy Above: ₹{h + 0.50:.2f}\n"
            f"Stop-Loss (20 EMA): ₹{e20:.2f}\n"
            f"Target (6%): ₹{c * 1.06:.2f}\n"
            f"Risk: {risk:.2f}%\n"
            f"RSI: {r:.2f} | "
            f"Volume: {v / va:.2f}x"
        )

        return match

    return None


# ============================================================
# MAIN SCANNER
# ============================================================

def main():

    start_time = time.monotonic()

    now = datetime.now(IST)

    print("=" * 65)
    print("NIFTY 500 DAILY STOCK SCANNER")
    print("Start time:", now.isoformat())
    print("Timeframe: DAILY")
    print("Scheduled frequency: EVERY 15 MINUTES")
    print("=" * 65)

    token, chat_id = get_telegram_credentials()

    tickers = get_nifty500_tickers()

    batches = [
        tickers[i:i + BATCH_SIZE]
        for i in range(0, len(tickers), BATCH_SIZE)
    ]

    print("Total stocks:", len(tickers))
    print("Total batches:", len(batches))

    matches = []
    failed = 0
    scanned = 0

    for batch_number, batch in enumerate(
        batches,
        start=1
    ):

        print("\n" + "-" * 50)

        print(
            f"Processing batch "
            f"{batch_number}/{len(batches)}"
        )

        downloaded = download_batch(batch)

        scanned += len(batch)

        if downloaded.empty:

            failed += len(batch)

            print(
                f"Batch failed: {len(batch)} stocks"
            )

            continue

        for ticker in batch:

            try:

                df = extract_ticker_frame(
                    downloaded,
                    ticker
                )

                if df.empty:

                    failed += 1

                    print(
                        "No Yahoo data:",
                        ticker
                    )

                    continue

                match = calculate_match(
                    ticker,
                    df
                )

                if match:

                    matches.append(match)

                    print(
                        "BUY MATCH:",
                        ticker
                    )

            except Exception as exc:

                failed += 1

                print(
                    f"Stock error {ticker}: "
                    f"{type(exc).__name__}: {exc}"
                )

        print(
            f"Batch completed: "
            f"{batch_number}/{len(batches)}"
        )

    # ========================================================
    # TELEGRAM REPORT
    # ========================================================

    completed_at = datetime.now(IST)

    duration = (
        time.monotonic() - start_time
    ) / 60

    heading = (
        "NIFTY 500 DAILY STOCK SCAN\n"
        "--------------------------------\n"
        f"Started: {now:%d-%m-%Y %H:%M:%S} IST\n"
        f"Completed: {completed_at:%d-%m-%Y %H:%M:%S} IST\n"
        f"Duration: {duration:.1f} minutes\n"
        f"Total Stocks: {len(tickers)}\n"
        f"Processed: {scanned}\n"
        f"Matches: {len(matches)}\n"
        f"Data Errors: {failed}\n"
        "--------------------------------\n\n"
    )

    if matches:

        report = heading + "\n\n".join(matches)

    else:

        report = (
            heading
            + "No stocks matched your criteria."
        )

    send_long_message(
        report,
        token,
        chat_id
    )

    print("\n" + "=" * 65)

    print("SCAN COMPLETED")
    print("Matches:", len(matches))
    print("Data errors:", failed)
    print(f"Total duration: {duration:.2f} minutes")

    print("=" * 65)


if __name__ == "__main__":
    main()
