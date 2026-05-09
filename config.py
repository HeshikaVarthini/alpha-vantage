"""Shared constants: tickers, sectors, and financial sentiment lexicon."""

# Expanded list of companies for analysis
all_companies = {
    'AAPL': 'Apple Inc.',
    'MSFT': 'Microsoft Corporation',
    'GOOGL': 'Alphabet Inc.',
    'AMZN': 'Amazon.com Inc.',
    'META': 'Meta Platforms Inc.',
    'TSLA': 'Tesla Inc.',
    'NVDA': 'NVIDIA Corporation',
    'JPM': 'JPMorgan Chase & Co.',
    'JNJ': 'Johnson & Johnson',
    'V': 'Visa Inc.',
    'PG': 'Procter & Gamble Co.',
    'DIS': 'Walt Disney Co.',
    'BAC': 'Bank of America Corp',
    'XOM': 'Exxon Mobil Corp',
    'INTC': 'Intel Corporation',
    'CSCO': 'Cisco Systems Inc.',
    'PFE': 'Pfizer Inc.',
    'KO': 'Coca-Cola Co',
    'WMT': 'Walmart Inc.',
    'GS': 'Goldman Sachs Group Inc.'
}

# Sector mapping
sector_mapping = {
    'AAPL': 'Technology', 'MSFT': 'Technology', 'GOOGL': 'Technology',
    'AMZN': 'Consumer Cyclical', 'META': 'Communication Services',
    'TSLA': 'Automotive', 'NVDA': 'Technology', 'JPM': 'Financial Services',
    'JNJ': 'Healthcare', 'V': 'Financial Services', 'PG': 'Consumer Defensive',
    'DIS': 'Communication Services', 'BAC': 'Financial Services',
    'XOM': 'Energy', 'INTC': 'Technology', 'CSCO': 'Technology',
    'PFE': 'Healthcare', 'KO': 'Consumer Defensive', 'WMT': 'Consumer Defensive',
    'GS': 'Financial Services'
}

# Enhanced sentiment lexicon with financial terms
FINANCIAL_SENTIMENT_LEXICON = {
    # Positive terms
    'bullish': 0.8, 'rally': 0.7, 'surge': 0.8, 'soar': 0.9, 'jump': 0.6,
    'gain': 0.6, 'profit': 0.7, 'growth': 0.7, 'optimistic': 0.6, 'strong': 0.5,
    'beat': 0.7, 'outperform': 0.7, 'upgrade': 0.6, 'buy': 0.6,
    'record': 0.6, 'breakthrough': 0.7, 'dividend': 0.4, 'buyback': 0.5,

    # Negative terms
    'bearish': -0.8, 'plunge': -0.8, 'slump': -0.7, 'drop': -0.6, 'fall': -0.6,
    'loss': -0.7, 'decline': -0.6, 'weak': -0.5, 'pessimistic': -0.6, 'downgrade': -0.6,
    'sell': -0.6, 'underperform': -0.6, 'miss': -0.6, 'cut': -0.5, 'reduce': -0.4,
    'warning': -0.7, 'risk': -0.5, 'volatile': -0.4, 'uncertain': -0.4,

    # Neutral/contextual terms
    'hold': 0.0, 'neutral': 0.0, 'maintain': 0.0, 'stable': 0.1, 'flat': 0.0,
    'consolidate': 0.0, 'sideways': 0.0
}
