import os
import requests
import pandas as pd
import yfinance as yf
from io import StringIO
from datetime import datetime
from zoneinfo import ZoneInfo

TOKEN = os.environ["TELEGRAM_BOT_TOKEN"]
CHAT_ID = os.environ["TELEGRAM_CHAT_ID"]

CSV_URL ="https://nsearchives.nseindia.com/content/indices/ind_nifty500list.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.niftyindices.com/",
}


def send_message(message):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    response = requests.post(
        url,
        data={"chat_id": CHAT_ID, "text": message},
        timeout=20,
    )
    response.raise_for_status()
    if not response.json().get("ok"):
        raise RuntimeError("Telegram rejected the message")


def main():
    now = datetime.now(ZoneInfo("Asia/Kolkata"))
    print("Starting NIFTY 500 scan:", now.isoformat())

    response = requests.get(CSV_URL, headers=HEADERS, timeout=30)
    response.raise_for_status()
    data = pd.read_csv(StringIO(response.text))

    symbol_col = next(
        (c for c in data.columns if c.strip().lower() in ("symbol", "ticker")),
        None,
    )
    if symbol_col is None:
        raise ValueError("NIFTY 500 symbol column not found")

    symbols = data[symbol_col].dropna().astype(str).str.strip().str.upper()
    tickers = list(dict.fromkeys(
        s if s.endswith(".NS") else s + ".NS"
        for s in symbols if s
    ))

    if len(tickers) < 400:
        raise ValueError("NIFTY 500 list appears incomplete")

    matches = []
    failed = 0

    for number, ticker in enumerate(tickers, 1):
        print(f"Scanning {number}/{len(tickers)}: {ticker}")

        try:
            df = yf.download(
                ticker,
                period="6mo",
                interval="1d",
                progress=False,
                auto_adjust=False,
                threads=False,
            )

            if df.empty or len(df) < 30:
                failed += 1
                continue

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            if not {"Close", "High", "Volume"}.issubset(df.columns):
                failed += 1
                continue

            close = pd.to_numeric(df["Close"], errors="coerce").dropna()
            volume = pd.to_numeric(df["Volume"], errors="coerce").reindex(close.index)
            high = pd.to_numeric(df["High"], errors="coerce").reindex(close.index)

            if len(close) < 30:
                failed += 1
                continue

            ema10 = close.ewm(span=10, adjust=False).mean()
            ema20 = close.ewm(span=20, adjust=False).mean()

            delta = close.diff()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi = rsi.mask((loss == 0) & (gain > 0), 100)
            rsi = rsi.mask((loss == 0) & (gain == 0), 50)

            vol_avg = volume.rolling(10).mean()

            vals = [
                close.iloc[-1], ema10.iloc[-1], ema20.iloc[-1],
                rsi.iloc[-1], volume.iloc[-1], vol_avg.iloc[-1],
                high.iloc[-1],
            ]

            if not all(pd.notna(v) for v in vals) or float(vol_avg.iloc[-1]) <= 0:
                failed += 1
                continue

            c, e10, e20, r, v, va, h = map(float, vals)

            if e10 > e20 and 55 <= r <= 68 and v > 1.5 * va and c > e20:
                matches.append(
                    f"Stock: {ticker.replace('.NS', '')}\n"
                    f"CMP: ₹{c:.2f}\n"
                    f"Buy Above: ₹{h + 0.50:.2f}\n"
                    f"Stop-Loss (20 EMA): ₹{e20:.2f}\n"
                    f"Target (6%): ₹{c * 1.06:.2f}\n"
                    f"Risk: {((c - e20) / c) * 100:.2f}%\n"
                    f"RSI: {r:.2f} | Volume: {v / va:.2f}x"
                )

        except Exception as exc:
            failed += 1
            print(f"Failed {ticker}: {exc}")

    heading = (
        f"NIFTY 500 STOCK SCAN\n"
        f"Time: {now:%d-%m-%Y %H:%M} IST\n"
        f"Matches: {len(matches)}\n"
        f"Data errors: {failed}\n\n"
    )

    if not matches:
        send_message(heading + "No stocks matched your criteria.")
    else:
        # Keep each Telegram message below its text limit.
        message = heading
        for match in matches:
            block = match + "\n\n"
            if len(message) + len(block) > 3500:
                send_message(message)
                message = f"More shortlisted stocks:\n\n{block}"
            else:
                message += block
        if message.strip():
            send_message(message)

    print("Scan completed. Matches:", len(matches))


if __name__ == "__main__":
    main()
