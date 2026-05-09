"""Shared project configuration constants."""

DEFAULT_MODEL_OPTIONS = ["TCN-I", "LSTM"]

TOP_COMPANIES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "GOOGL": "Alphabet Inc.",
    "AMZN": "Amazon.com Inc.",
    "META": "Meta Platforms Inc.",
    "TSLA": "Tesla Inc.",
    "NVDA": "NVIDIA Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "JNJ": "Johnson & Johnson",
    "V": "Visa Inc.",
}

SECTOR_MAPPING = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "GOOGL": "Technology",
    "AMZN": "Consumer Cyclical",
    "META": "Communication Services",
    "TSLA": "Automotive",
    "NVDA": "Technology",
    "JPM": "Financial Services",
    "JNJ": "Healthcare",
    "V": "Financial Services",
}
