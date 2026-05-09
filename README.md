# Alpha Vantage Financial Dashboard

**Alpha Vantage API → Streamlit → TCN / LSTM forecasting + FinBERT-style sentiment**

An AI-powered financial dashboard built with Streamlit: conversational Q&A over equities, technical indicators, multi-layer sentiment (lexicons, WordNet-style expansion, FinBERT pipeline), and optional real-time data from [Alpha Vantage](https://www.alphavantage.co/).

Repository layout follows the same conventions as [Job Notifier](https://github.com/Hari-Prasath-M91/Job-Notifier): clear entrypoint, `requirements.txt`, `.env` support, documented setup, and a GPL-3.0 license.

## Architecture

```
Alpha Vantage API  →  pandas / ta  →  TCN & LSTM forecasts
        │                    │
        └──── NEWS / quotes ──┴──► Multi-layer sentiment + FinBERT
                                           │
                                    Streamlit dashboard (chat UI)
```

## Features

- **Streamlit UI** with sidebar configuration (API key, mock vs live data, model preference).
- **Forecasting**: Temporal Convolutional Network (TCN) and LSTM with training curves and metric comparisons.
- **Sentiment**: Enhanced pipeline plus FinBERT-only path; optional side-by-side comparison and classification metrics.
- **Mock mode** when no API key is set (demo-friendly).
- **Optional `.env`** for `ALPHA_VANTAGE_API_KEY` so keys are not typed every session.

## Setup

### 1. Alpha Vantage API key

1. Sign up at [alphavantage.co](https://www.alphavantage.co/support/#api-key).
2. Either paste the key in the app sidebar or put it in `.env` (see below).

Free tier limits apply (e.g. request rate); use **mock data** when experimenting.

### 2. Python environment

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

**PyTorch:** If you need a CUDA build, install `torch` from the [official PyTorch install page](https://pytorch.org/get-started/locally/) instead of relying on the generic wheel from pip.

### 3. Environment file (optional)

```bash
copy .env.example .env   # Windows
# cp .env.example .env    # macOS / Linux
```

Edit `.env`:

| Variable | Description |
|----------|-------------|
| `ALPHA_VANTAGE_API_KEY` | Your Alpha Vantage key (pre-fills the sidebar field on first load). |

## Run locally

From the project root:

```bash
python main.py
```

Or run Streamlit directly:

```bash
streamlit run dashboard.py
```

Open the URL shown in the terminal (default `http://localhost:8501`).

## Deploy (Streamlit Community Cloud)

1. Push this repo to GitHub (do **not** commit `.env`).
2. On [Streamlit Community Cloud](https://streamlit.io/cloud), connect the repo and set **Main file path** to `dashboard.py`.
3. Add secrets in the dashboard: `ALPHA_VANTAGE_API_KEY` = your key (and map it in app settings if you extend the app to read `st.secrets`).

## Project structure

```
alpha-vantage/
├── .env.example          # Template for environment variables
├── .gitignore
├── LICENSE               # GPL-3.0 (same family as Job Notifier reference)
├── README.md
├── alpha6.py             # Shim → same as python main.py (old repo filename)
├── config.py             # Tickers, sector map, financial sentiment lexicon
├── dashboard.py          # Streamlit app (core logic was formerly only alpha6.py)
├── main.py               # Entrypoint: runs Streamlit
└── requirements.txt
```

## Security

- Keep API keys in `.env` or Streamlit secrets — never commit them.
- This app is for **education and demonstration** only; outputs are not financial advice.

## License

GPL-3.0 — see [LICENSE](LICENSE).

## Acknowledgement

Structure and documentation style inspired by [Hari-Prasath-M91/Job-Notifier](https://github.com/Hari-Prasath-M91/Job-Notifier).
