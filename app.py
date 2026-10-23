import streamlit as st
import yfinance as yf
import pandas as pd
import requests

st.set_page_config(page_title="6% Swing Strategy Screener", layout="wide")
st.title("📈 6% Swing Trading Screener — NIFTY 500")
if st.button("Test Telegram Connection"):
    try:
        token = st.secrets["TELEGRAM_BOT_TOKEN"]
        chat_id = st.secrets["TELEGRAM_CHAT_ID"]

        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            data={
                "chat_id": chat_id,
                "text": "✅ Connected! Your stock scanner Telegram alerts are ready to test."
            },
            timeout=15,
        )
        response.raise_for_status()

        if response.json().get("ok"):
            st.success("Test message sent! Check Telegram.")
        else:
            st.error("Telegram did not accept the message.")
    except Exception as e:
        st.error(f"Telegram test failed: {e}")
st.markdown("""
**Active rules**
- 10 EMA > 20 EMA
- RSI between 55 and 68
- Latest volume > 1.5 × 10-day average volume
- Price > 20 EMA

The app downloads the NIFTY 500 constituent list and scans its NSE tickers.
""")

NIFTY500_CSV_URL = "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/126.0 Safari/537.36",
    "Referer": "https://www.niftyindices.com/",
}

@st.cache_data(ttl=86400, show_spinner=False)
def get_nifty500_tickers():
    response = requests.get(NIFTY500_CSV_URL, headers=HEADERS, timeout=20)
    response.raise_for_status()
    data = pd.read_csv(pd.io.common.StringIO(response.text))
    symbol_col = next((c for c in data.columns if c.strip().lower() in ("symbol", "ticker")), None)
    if symbol_col is None:
        raise ValueError("No Symbol column found in NIFTY 500 CSV.")
    symbols = data[symbol_col].dropna().astype(str).str.strip().str.upper()
    tickers = [s if s.endswith(".NS") else s + ".NS" for s in symbols if s]
    tickers = list(dict.fromkeys(tickers))
    if len(tickers) < 400:
        raise ValueError(f"Only {len(tickers)} constituents loaded; expected the NIFTY 500 universe.")
    return tickers

st.sidebar.header("⚙️ Telegram Configuration")
telegram_token = st.sidebar.text_input("Bot Token", type="password")
telegram_chat_id = st.sidebar.text_input("Chat ID")

def send_telegram_msg(token, chat_id, message):
    if not token or not chat_id:
        return False
    try:
        response = requests.post(
            f"https://api.telegram.org/bot{token}/sendMessage",
            json={"chat_id": chat_id, "text": message, "parse_mode": "Markdown"},
            timeout=10,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False

try:
    ticker_list = get_nifty500_tickers()
    st.success(f"NIFTY 500 universe loaded: {len(ticker_list)} stocks")
except Exception as exc:
    ticker_list = []
    st.error("Could not load the official NIFTY 500 list. Please retry later.")
    st.caption(f"Details: {exc}")

with st.expander("View NIFTY 500 tickers"):
    if ticker_list:
        st.write(", ".join(t.replace(".NS", "") for t in ticker_list))

if "scan_results" not in st.session_state:
    st.session_state.scan_results = None

if st.button("🚀 Run NIFTY 500 Market Scan", type="primary", disabled=not ticker_list):
    results, failed = [], 0
    bar, status = st.progress(0), st.empty()
    for i, ticker in enumerate(ticker_list):
        status.text(f"Scanning {ticker} ({i + 1}/{len(ticker_list)})...")
        bar.progress((i + 1) / len(ticker_list))
        try:
            df = yf.download(ticker, period="6mo", interval="1d", progress=False,
                             auto_adjust=False, threads=False)
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

            vals = [close.iloc[-1], ema10.iloc[-1], ema20.iloc[-1], rsi.iloc[-1],
                    volume.iloc[-1], vol_avg.iloc[-1], high.iloc[-1]]
            if not all(pd.notna(v) for v in vals) or float(vol_avg.iloc[-1]) <= 0:
                failed += 1
                continue

            c, e10, e20, r, v, va, h = map(float, vals)
            if e10 > e20 and 55 <= r <= 68 and v > 1.5 * va and c > e20:
                results.append({
                    "Ticker": ticker.replace(".NS", ""),
                    "CMP (₹)": round(c, 2),
                    "RSI": round(r, 2),
                    "Vol Ratio": f"{v / va:.2f}x",
                    "Buy Trigger": round(h + 0.50, 2),
                    "SL (20 EMA)": round(e20, 2),
                    "Target (6%)": round(c * 1.06, 2),
                    "Risk %": f"{((c - e20) / c) * 100:.2f}%"
                })
        except Exception:
            failed += 1
    bar.empty()
    status.empty()
    st.session_state.scan_results = results
    st.session_state.scan_failed = failed
    st.session_state.scan_count = len(ticker_list)

if st.session_state.scan_results is not None:
    results = st.session_state.scan_results
    st.caption(f"Universe: {st.session_state.get('scan_count', 0)} stocks · No usable data/errors: {st.session_state.get('scan_failed', 0)}")
    if results:
        st.success(f"🎯 Found {len(results)} qualifying setups!")
        st.dataframe(pd.DataFrame(results), use_container_width=True, hide_index=True)
        if st.button("📱 Push All Signals to Telegram"):
            if not telegram_token or not telegram_chat_id:
                st.error("Configure Telegram token and chat ID in the sidebar.")
            else:
                sent = 0
                for row in results:
                    message = (
                        f"🚨 *SWING TRADE ALERT (6% Target)* 🚨\n\n"
                        f"📌 *Stock:* `{row['Ticker']}`\n"
                        f"💰 *CMP:* ₹{row['CMP (₹)']}\n"
                        f"🎯 *Buy Above:* ₹{row['Buy Trigger']}\n"
                        f"🛑 *Stop-Loss:* ₹{row['SL (20 EMA)']} ({row['Risk %']})\n"
                        f"🚀 *6% Target:* ₹{row['Target (6%)']}\n"
                        f"📊 *RSI:* {row['RSI']} | *Volume:* {row['Vol Ratio']}"
                    )
                    sent += int(send_telegram_msg(telegram_token, telegram_chat_id, message))
                st.success(f"Sent {sent} of {len(results)} Telegram alerts.")
    else:
        st.warning("No stocks met the criteria today.")
