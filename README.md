# Alpha Vantage Financial Dashboard

**Alpha Vantage API -> Streamlit -> TCN / LSTM forecasting + FinBERT-style sentiment**

An AI-powered financial dashboard built with Streamlit: conversational Q&A over equities, technical indicators, multi-layer sentiment analysis (lexicons, context checks, and transformer-based classification), and optional real-time data from Alpha Vantage.

## Architecture

```text
Alpha Vantage API  ->  pandas / ta  ->  TCN & LSTM forecasts
        |                    |
        +---- NEWS / quotes--+--> Multi-layer sentiment + FinBERT
                                           |
                                    Streamlit dashboard (chat UI)
```

## Features

- Streamlit UI with sidebar configuration (API key, mock vs live data, model preference).
- Forecasting with both TCN and LSTM, including metrics and visual comparison.
- Multi-layer sentiment analysis pipeline with financial context and classification reporting.
- Mock mode support when no API key is provided.
- Optional `.env` support for `ALPHA_VANTAGE_API_KEY`.

## Setup

### 1) Get Alpha Vantage API key

1. Sign up at [Alpha Vantage](https://www.alphavantage.co/).
2. Paste key in the app sidebar or store it in `.env`.

### 2) Create virtual environment and install dependencies

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS/Linux:
# source venv/bin/activate

pip install -r requirements.txt
```

### 3) Environment file (optional)

```bash
copy .env.example .env   # Windows
# cp .env.example .env   # macOS/Linux
```

Then set:

```env
ALPHA_VANTAGE_API_KEY=your_api_key_here
```

## Run locally

```bash
python main.py
```

Or run Streamlit directly:

```bash
streamlit run dashboard.py
```

## Project Structure

```text
project/
├── .env.example
├── .gitignore
├── README.md
├── alpha6.py             # Existing core dashboard implementation
├── dashboard.py          # Streamlit entry module (loads alpha6 app)
├── main.py               # Python entrypoint that launches Streamlit
├── config.py             # Shared configuration constants
├── requirements.txt
└── requirement.txt       # Legacy dependency file (kept for compatibility)
```

## Security

- Keep API keys in `.env` (never commit secrets).
- This project is for educational and demonstration use only.
- Outputs are not financial advice.
