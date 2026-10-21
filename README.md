# NIFTY 500 6% Swing Screener

Upload `app.py`, `requirements.txt`, and `README.md` to the root of your GitHub repository. Deploy `app.py` using Streamlit Community Cloud.

The app fetches the official NIFTY 500 constituent CSV at runtime and caches it for 24 hours. It scans daily Yahoo Finance data for the four screening rules listed in the app. A full scan can take time; Yahoo Finance may throttle requests. Internet access is required.
