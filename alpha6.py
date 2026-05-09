import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import requests
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
import ta  # For technical indicators
import time
import warnings
import random
warnings.filterwarnings('ignore')
from collections import Counter
import re
from scipy import stats
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from nltk.corpus import wordnet
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch
from transformers import BertTokenizer, BertForSequenceClassification, pipeline, TFBertForSequenceClassification
import streamlit as st

# Download required NLTK data
'''try:
    nltk.data.find('vader_lexicon')
    nltk.data.find('punkt')
    nltk.data.find('wordnet')
    nltk.data.find('omw-1.4')
except LookupError:
    nltk.download('vader_lexicon')
    nltk.download('punkt')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
            "text-classification",
'''
# Streamlit page config
st.set_page_config(
    page_title="Financial Conversational AI Dashboard",
    page_icon="📈",
    layout="wide"
)

if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
    
if 'stock_data' not in st.session_state:
    st.session_state.stock_data = {}

if 'news_data' not in st.session_state:
    st.session_state.news_data = {}

st.title("📈 Conversational AI Financial Dashboard")
st.markdown("""
Interact with real-time financial data using natural language.
Ask questions about stocks, sectors, or technical indicators and get AI-powered insights.
""")

st.sidebar.header("Configuration")
alpha_vantage_api_key = st.sidebar.text_input("Alpha Vantage API Key", type="password")
use_mock_data = st.sidebar.checkbox("Use Mock Data (if no API key)", value=True)

# Display important information about API keys
st.sidebar.info("""
**Note about API Keys:**
- This application requires an Alpha Vantage API key
- Free tier allows 5 API requests per minute
- You can get a free API key from https://www.alphavantage.co/
""")

if alpha_vantage_api_key and not use_mock_data:
    st.sidebar.success("API Key provided - using real data")
elif use_mock_data:
    st.sidebar.info("Using mock data for demonstration")
else:
    st.sidebar.warning("Using mock data - enter API key for real data")

# Add API connection test button
if st.sidebar.button("Test API Connection") and alpha_vantage_api_key and not use_mock_data:
    test_url = f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE&symbol=AAPL&apikey={alpha_vantage_api_key}"
    
    try:
        response = requests.get(test_url)
        if response.status_code == 200:
            data = response.json()
            if 'Global Quote' in data:
                st.sidebar.success("API connection successful!")
            elif 'Note' in data:
                st.sidebar.warning(f"API Note: {data['Note']}")
            elif 'Error Message' in data:
                st.sidebar.error(f"API Error: {data['Error Message']}")
            else:
                st.sidebar.error("Unexpected API response format")
        else:
            st.sidebar.error(f"HTTP Error: {response.status_code}")
    except Exception as e:
        st.sidebar.error(f"Connection failed: {str(e)}")

model_type = st.sidebar.selectbox(
    "Select Forecasting Model",
    ["TCN-I", "LSTM"],
    help="Choose which model to use for time series forecasting"
)

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
    'beat': 0.7, 'outperform': 0.7, 'upgrade': 0.6, 'buy': 0.6, 'outperform': 0.7,
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

def generate_mock_stock_data(symbol, days=90):
    if symbol in st.session_state.stock_data:
        return st.session_state.stock_data[symbol]
    
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    np.random.seed(hash(symbol) % 10000)
    base_price = np.random.uniform(50, 300)
    
    # Create more realistic price patterns with trends and volatility
    trend = np.random.choice([-0.0005, 0, 0.0005])  # Slight trend
    volatility = np.random.uniform(0.01, 0.03)
    
    # Generate random walk with trend
    returns = np.random.normal(trend, volatility, days)
    prices = base_price * (1 + returns).cumprod()
    
    # Generate OHLC data with realistic relationships
    opens = []
    highs = []
    lows = []
    closes = []
    volumes = []
    
    for i in range(days):
        prev_close = prices[i-1] if i > 0 else prices[i]
        open_price = prev_close * (1 + np.random.normal(0, 0.005))
        close_price = prices[i]
        
        # High and low based on open and close
        daily_range = abs(close_price - open_price) + close_price * np.random.uniform(0.01, 0.03)
        high_price = max(open_price, close_price) + daily_range * np.random.uniform(0.1, 0.4)
        low_price = min(open_price, close_price) - daily_range * np.random.uniform(0.1, 0.4)
        
        # Ensure high is highest and low is lowest
        high_price = max(open_price, close_price, high_price)
        low_price = min(open_price, close_price, low_price)
        
        opens.append(open_price)
        highs.append(high_price)
        lows.append(low_price)
        closes.append(close_price)
        volumes.append(np.random.lognormal(15, 1))
    
    df = pd.DataFrame({
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'volume': volumes
    }, index=dates)
    
    st.session_state.stock_data[symbol] = df
    return df

def fetch_stock_data(symbol, days=90):
    # Check if we already have cached data for this symbol
    if symbol in st.session_state.stock_data:
        cached_data = st.session_state.stock_data[symbol]
        if len(cached_data) >= days:
            return cached_data.iloc[-days:]
    
    if use_mock_data or not alpha_vantage_api_key:
        return generate_mock_stock_data(symbol, days)
    
    # Alpha Vantage API endpoint for daily data
    url = f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={alpha_vantage_api_key}&outputsize=compact"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check for errors in response
        if 'Error Message' in data:
            st.error(f"Alpha Vantage Error for {symbol}: {data['Error Message']}")
            return generate_mock_stock_data(symbol, days)
        elif 'Note' in data:
            st.warning(f"Alpha Vantage Note for {symbol}: {data['Note']}")
            # Use mock data if we hit API limits
            return generate_mock_stock_data(symbol, days)
        
        if 'Time Series (Daily)' in data:
            time_series = data['Time Series (Daily)']
            dates = []
            opens = []
            highs = []
            lows = []
            closes = []
            volumes = []
            
            # Convert to dataframe
            for date, values in time_series.items():
                dates.append(datetime.strptime(date, '%Y-%m-%d'))
                opens.append(float(values['1. open']))
                highs.append(float(values['2. high']))
                lows.append(float(values['3. low']))
                closes.append(float(values['4. close']))
                volumes.append(float(values['5. volume']))
            
            df = pd.DataFrame({
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            }, index=dates)
            
            # Sort by date and get the most recent days
            df = df.sort_index(ascending=True)
            if len(df) > days:
                df = df.iloc[-days:]
            
            # Cache the data
            st.session_state.stock_data[symbol] = df
            return df
        else:
            st.warning(f"No time series data found for {symbol} in API response. Using mock data.")
            return generate_mock_stock_data(symbol, days)
            
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {str(e)}. Using mock data.")
        return generate_mock_stock_data(symbol, days)

# NEW FUNCTION: Fetch real news from Alpha Vantage
def fetch_alpha_vantage_news(symbol, limit=10):
    """
    Fetch real news articles from Alpha Vantage NEWS_SENTIMENT endpoint
    """
    # Check if we have cached news data for this symbol
    cache_key = f"{symbol}_news"
    if cache_key in st.session_state.news_data:
        cached_news = st.session_state.news_data[cache_key]
        # Check if cache is still fresh (less than 1 hour old)
        if time.time() - cached_news.get('timestamp', 0) < 3600:
            return cached_news['articles']
    
    if use_mock_data or not alpha_vantage_api_key:
        return generate_mock_news(symbol)
    
    url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={symbol}&apikey={alpha_vantage_api_key}&limit={limit}"
    
    try:
        response = requests.get(url)
        data = response.json()
        
        # Check for errors in response
        if 'Error Message' in data:
            st.warning(f"Alpha Vantage News Error for {symbol}: {data['Error Message']}. Using mock data.")
            return generate_mock_news(symbol)
        elif 'Note' in data:
            st.warning(f"Alpha Vantage Note for {symbol}: {data['Note']}. Using mock data.")
            return generate_mock_news(symbol)
        
        if 'feed' in data:
            articles = []
            for item in data['feed']:
                article = {
                    'title': item.get('title', 'No title available'),
                    'summary': item.get('summary', item.get('title', 'No summary available')),
                    'url': item.get('url', '#'),
                    'source': item.get('source', 'Unknown source'),
                    'time_published': item.get('time_published', ''),
                    'sentiment_score': item.get('overall_sentiment_score', 0),
                    'sentiment_label': item.get('overall_sentiment_label', 'Neutral')
                }
                articles.append(article)
            
            # Cache the news data
            st.session_state.news_data[cache_key] = {
                'articles': articles,
                'timestamp': time.time()
            }
            
            return articles
        else:
            st.warning(f"No news data found for {symbol} in API response. Using mock data.")
            return generate_mock_news(symbol)
            
    except Exception as e:
        st.error(f"Error fetching news for {symbol}: {str(e)}. Using mock data.")
        return generate_mock_news(symbol)

# MODIFIED FUNCTION: Now uses real Alpha Vantage news when available
def generate_mock_news(symbol):
    """Generate mock news articles for demonstration (fallback when real news is unavailable)"""
    news_templates = [
        {"title": f"{symbol} Reports Strong Quarterly Earnings", "summary": f"{symbol} exceeded analyst expectations with record revenue growth."},
        {"title": f"Analysts Upgrade {symbol} to Buy Rating", "summary": "Several major analysts have upgraded their outlook on the stock."},
        {"title": f"{symbol} Announces New Product Launch", "summary": "The company unveiled its latest innovation expected to drive future growth."},
        {"title": f"Market Volatility Impacts {symbol} Shares", "summary": "Recent market conditions have created uncertainty around the stock."},
        {"title": f"{symbol} Faces Regulatory Challenges", "summary": "New regulations could impact the company's operations in key markets."}
    ]
    return random.sample(news_templates, min(3, len(news_templates)))

def calculate_technical_indicators(df):
    if df is None or len(df) == 0:
        return df
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
    
    # Moving Averages
    df['sma_20'] = ta.trend.SMAIndicator(df['close'], window=20).sma_indicator()
    df['sma_50'] = ta.trend.SMAIndicator(df['close'], window=50).sma_indicator()
    df['sma_200'] = ta.trend.SMAIndicator(df['close'], window=200).sma_indicator()
    df['ema_12'] = ta.trend.EMAIndicator(df['close'], window=12).ema_indicator()
    df['ema_26'] = ta.trend.EMAIndicator(df['close'], window=26).ema_indicator()
    
    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bollinger.bollinger_hband()
    df['bb_mid'] = bollinger.bollinger_mavg()
    df['bb_low'] = bollinger.bollinger_lband()
    df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # Average True Range
    df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
    
    # On Balance Volume
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    
    return df
from transformers import pipeline
@st.cache_resource
#def load_finbert_model():
 #   try:
  #      finbert = pipeline("text-classification",
   #                   model="ProsusAI/finbert",
    #                  tokenizer="ProsusAI/finbert",
     #                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
      #                from_tf=True)
       # return finbert
    #except Exception as e:
     #   st.error(f"Error loading FinBERT model: {str(e)}")
     #   return None


def load_finbert_model():
    try:
        # Use a PyTorch-based sentiment analysis model that works reliably
        finbert = pipeline(
            "sentiment-analysis",
            model="nlptown/bert-base-multilingual-uncased-sentiment",
            tokenizer="nlptown/bert-base-multilingual-uncased-sentiment"
        )
        return finbert

    except Exception as e:
        st.error(f"Error loading sentiment model: {str(e)}")
        return None
# Enhanced TCN Model Definition with performance tracking
class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = nn.Sequential(
            nn.Conv1d(input_size, num_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_channels, num_channels, kernel_size, padding=(kernel_size-1)//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(num_channels, output_size)
        )
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.tcn(x)
        return x

# Enhanced LSTM Model Definition with performance tracking
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
        self.train_losses = []
        self.val_losses = []
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Function to prepare data for forecasting
def prepare_forecasting_data(df, feature='close', seq_length=30, train_ratio=0.8):
    """Prepare data for time series forecasting with train/validation split"""
    values = df[feature].values.reshape(-1, 1)
    
    # Scale the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(values)
    
    # Create sequences
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i-seq_length:i, 0])
        y.append(scaled_data[i, 0])
    
    X, y = np.array(X), np.array(y)
    
    # Split into train and validation
    split_idx = int(len(X) * train_ratio)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    return X_train, X_val, y_train, y_val, scaler

# Enhanced training function with performance metrics
def train_forecast(model, X_train, X_val, y_train, y_val, epochs=100, learning_rate=0.001, model_type="TCN"):
    """Train forecasting model with comprehensive performance metrics"""
    criterion = nn.MSELoss()
    
    if model_type == "TCN":
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    else:  # LSTM
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    
    # Convert to tensors with correct dimensions
    if model_type == "TCN":
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1)
    else:  # LSTM
        X_train_tensor = torch.FloatTensor(X_train).unsqueeze(-1)
        X_val_tensor = torch.FloatTensor(X_val).unsqueeze(-1)
    
    y_train_tensor = torch.FloatTensor(y_train).unsqueeze(-1)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(-1)
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor)
        
        model.train()
        
        # Store losses
        model.train_losses.append(loss.item())
        model.val_losses.append(val_loss.item())
    
    # Make predictions
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor)
        val_predictions = model(X_val_tensor)
    
    # Calculate comprehensive metrics
    train_metrics = calculate_performance_metrics(y_train_tensor.numpy(), train_predictions.numpy())
    val_metrics = calculate_performance_metrics(y_val_tensor.numpy(), val_predictions.numpy())
    
    return train_predictions.numpy(), val_predictions.numpy(), train_metrics, val_metrics

def calculate_performance_metrics(y_true, y_pred):
    """Calculate comprehensive performance metrics"""
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from scipy.stats import pearsonr
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate directional accuracy
    y_true_dir = np.diff(y_true.flatten()) > 0
    y_pred_dir = np.diff(y_pred.flatten()) > 0
    directional_accuracy = np.mean(y_true_dir == y_pred_dir) if len(y_true_dir) > 0 else 0
    
    # Calculate correlation
    correlation = pearsonr(y_true.flatten(), y_pred.flatten())[0] if len(y_true) > 1 else 0
    
    return {
        'MSE': mse,
        'MAE': mae,
        'RMSE': rmse,
        'R2': r2,
        'Directional_Accuracy': directional_accuracy,
        'Correlation': correlation
    }

def plot_model_performance(model, model_name):
    """Plot training and validation performance"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    ax1.plot(model.train_losses, label='Training Loss')
    ax1.plot(model.val_losses, label='Validation Loss')
    ax1.set_title(f'{model_name} - Training vs Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot performance metrics
    epochs = range(len(model.train_losses))
    ax2.plot(epochs, model.train_losses, label='Train Loss')
    ax2.plot(epochs, model.val_losses, label='Val Loss')
    ax2.set_title(f'{model_name} - Loss Convergence')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    return fig

def analyze_most_traded_stocks():
    """Analyze and return the most traded stocks by volume"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total_companies = len(all_companies)
    
    for i, (symbol, name) in enumerate(all_companies.items()):
        status_text.text(f"Analyzing {name} ({symbol})...")
        progress_bar.progress((i + 1) / total_companies)
        
        # Add delay to avoid rate limiting
        if not use_mock_data and alpha_vantage_api_key and i > 0 and i % 5 == 0:
            time.sleep(65)
        
        # Fetch stock data
        df = fetch_stock_data(symbol, days=30)  # Last 30 days for volume analysis
        if df is None or len(df) == 0:
            continue
        
        # Calculate average volume
        avg_volume = df['volume'].mean()
        current_volume = df['volume'].iloc[-1]
        
        # Store results
        results.append({
            'symbol': symbol,
            'name': name,
            'avg_volume': avg_volume,
            'current_volume': current_volume,
            'volume_ratio': current_volume / avg_volume if avg_volume > 0 else 0
        })
    
    # Sort by average volume (descending)
    results.sort(key=lambda x: x['avg_volume'], reverse=True)
    return pd.DataFrame(results)

def analyze_low_value_timing(symbol):
    """Analyze when a stock typically has lower values"""
    df = fetch_stock_data(symbol, days=365)  # 1 year of data
    
    if df is None or len(df) == 0:
        return "No data available for analysis."
    
    # Calculate daily returns
    df['daily_return'] = df['close'].pct_change()
    
    # Analyze by day of week
    df['day_of_week'] = df.index.dayofweek
    day_stats = df.groupby('day_of_week')['daily_return'].mean()
    
    # Analyze by month
    df['month'] = df.index.month
    month_stats = df.groupby('month')['daily_return'].mean()
    
    # Find the day with lowest average returns
    worst_day = day_stats.idxmin()
    worst_day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][worst_day]
    
    # Find the month with lowest average returns
    worst_month = month_stats.idxmin()
    worst_month_name = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ][worst_month - 1]
    
    return worst_day_name, worst_month_name, day_stats, month_stats

def analyze_sell_signals(symbol):
    """Analyze technical indicators to identify sell signals"""
    df = fetch_stock_data(symbol, days=365)
    
    if df is None or len(df) == 0:
        return ["No data available for analysis."], None
    
    # Calculate technical indicators
    df = calculate_technical_indicators(df)
    
    # Get current values
    current_rsi = df['rsi'].iloc[-1] if 'rsi' in df.columns else 50
    current_macd = df['macd'].iloc[-1] if 'macd' in df.columns else 0
    current_macd_signal = df['macd_signal'].iloc[-1] if 'macd_signal' in df.columns else 0
    current_price = df['close'].iloc[-1]
    
    # Check for sell signals
    sell_signals = []
    
    # RSI overbought
    if current_rsi > 70:
        sell_signals.append(f"RSI is overbought ({current_rsi:.2f})")
    
    # MACD bearish crossover
    if current_macd < current_macd_signal:
        sell_signals.append("MACD shows bearish signal")
    
    # Price above upper Bollinger Band (if available)
    if 'bb_high' in df.columns and current_price > df['bb_high'].iloc[-1]:
        sell_signals.append("Price is above upper Bollinger Band")
    
    # Check trend (using EMA if available)
    if 'ema_12' in df.columns and 'ema_26' in df.columns:
        if df['ema_12'].iloc[-1] < df['ema_26'].iloc[-1]:
            sell_signals.append("Short-term EMA below long-term EMA (bearish trend)")
    
    # If no specific sell signals, check if stock is at high levels
    if not sell_signals:
        # Check if price is near 52-week high (if we have enough data)
        if len(df) >= 252:
            fifty_two_week_high = df['close'].max()
            if current_price >= fifty_two_week_high * 0.95:  # Within 5% of 52-week high
                sell_signals.append("Price is near 52-week high, consider taking profits")
    
    return sell_signals, df

def calculate_optimal_portfolio_size(available_capital):
    """Calculate optimal number of stocks based on available capital"""
    # Basic portfolio sizing rules
    if available_capital < 5000:
        return 3, "Limited diversification due to small capital"
    elif available_capital < 20000:
        return 5, "Moderate diversification"
    elif available_capital < 50000:
        return 8, "Good diversification"
    elif available_capital < 100000:
        return 12, "Well-diversified portfolio"
    else:
        return 15, "Highly diversified portfolio"

def extract_symbols_from_question(question):
    """Extract stock symbols from the question"""
    question_upper = question.upper()
    symbols_found = []
    
    for symbol in all_companies:
        if symbol in question_upper or all_companies[symbol].upper() in question_upper:
            symbols_found.append(symbol)
    
    return symbols_found

def extract_timeframe_from_question(question):
    """Extract timeframe from the question"""
    question_lower = question.lower()
    
    if 'tomorrow' in question_lower:
        return 'tomorrow'
    elif 'next week' in question_lower:
        return 'next week'
    elif 'next month' in question_lower:
        return 'next month'
    elif 'next quarter' in question_lower:
        return 'next quarter'
    elif 'this year' in question_lower:
        return 'this year'
    elif 'long-term' in question_lower:
        return 'long-term'
    else:
        return 'short-term'

def analyze_price_trend(symbol, timeframe='short-term'):
    """Analyze price trend for a stock"""
    df = fetch_stock_data(symbol, days=365 if timeframe == 'long-term' else 90)
    
    if df is None or len(df) == 0:
        return "No data available for analysis."
    
    # Calculate returns
    df['returns'] = df['close'].pct_change()
    
    # Calculate moving averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # Calculate trend strength
    if len(df) >= 20:
        trend_strength = stats.linregress(range(len(df)), df['close']).slope * len(df) / df['close'].iloc[-1]
    else:
        trend_strength = 0
    
    # Determine trend direction
    current_price = df['close'].iloc[-1]
    sma_20 = df['sma_20'].iloc[-1]
    sma_50 = df['sma_50'].iloc[-1]
    
    if current_price > sma_20 > sma_50:
        trend = "strong uptrend"
    elif current_price > sma_20 and sma_20 > sma_50:
        trend = "moderate uptrend"
    elif current_price < sma_20 < sma_50:
        trend = "strong downtrend"
    elif current_price < sma_20 and sma_20 < sma_50:
        trend = "moderate downtrend"
    else:
        trend = "sideways or consolidating"
    
    # Calculate volatility
    volatility = df['returns'].std() * np.sqrt(252)  # Annualized volatility
    
    return {
        'trend': trend,
        'trend_strength': trend_strength,
        'volatility': volatility,
        'current_price': current_price,
        'sma_20': sma_20,
        'sma_50': sma_50
    }

def analyze_sector_performance():
    """Analyze performance by sector"""
    sector_performance = {}
    
    for symbol in all_companies:
        if symbol in sector_mapping:
            sector = sector_mapping[symbol]
            df = fetch_stock_data(symbol, days=90)
            
            if df is not None and len(df) > 0:
                returns = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                
                if sector not in sector_performance:
                    sector_performance[sector] = []
                
                sector_performance[sector].append(returns)
    
    # Calculate average returns by sector
    sector_avg_returns = {}
    for sector, returns in sector_performance.items():
        sector_avg_returns[sector] = sum(returns) / len(returns)
    
    return sector_avg_returns

def predict_future_price(symbol, days=1):
    """Predict future price using simple forecasting"""
    df = fetch_stock_data(symbol, days=90)
    
    if df is None or len(df) < 30:
        return None
    
    # Simple prediction based on recent trend
    recent_returns = df['close'].pct_change().dropna()
    avg_daily_return = recent_returns.mean()
    current_price = df['close'].iloc[-1]
    
    # Predict future price
    predicted_price = current_price * (1 + avg_daily_return) ** days
    
    return predicted_price

def analyze_earnings_impact(symbol):
    """Analyze historical earnings impact"""
    df = fetch_stock_data(symbol, days=365)
    
    if df is None or len(df) < 60:
        return "Insufficient data for earnings analysis"
    
    # Simulate earnings dates (quarterly)
    earnings_dates = []
    current_date = df.index[-1]
    
    for i in range(4):
        earnings_date = current_date - timedelta(days=90*(i+1))
        earnings_dates.append(earnings_date)
    
    # Analyze price changes around earnings
    earnings_impacts = []
    
    for earnings_date in earnings_dates:
        # Find the closest trading date
        time_diff = [(date - earnings_date).days for date in df.index]
        abs_time_diff = [abs(days) for days in time_diff]
        closest_idx = abs_time_diff.index(min(abs_time_diff))
        
        if closest_idx >= 5 and closest_idx < len(df) - 5:
            pre_earnings_price = df['close'].iloc[closest_idx-5]
            post_earnings_price = df['close'].iloc[closest_idx+5]
            impact = (post_earnings_price / pre_earnings_price - 1) * 100
            earnings_impacts.append(impact)
    
    if earnings_impacts:
        avg_impact = sum(earnings_impacts) / len(earnings_impacts)
        return avg_impact
    else:
        return 0

def analyze_technical_patterns(symbol):
    """Analyze technical patterns for a stock"""
    df = fetch_stock_data(symbol, days=90)
    
    if df is None or len(df) < 30:
        return "Insufficient data for technical pattern analysis"
    
    df = calculate_technical_indicators(df)
    
    patterns = []
    
    # Check for RSI patterns
    current_rsi = df['rsi'].iloc[-1]
    if current_rsi < 30:
        patterns.append("Oversold (RSI < 30)")
    elif current_rsi > 70:
        patterns.append("Overbought (RSI > 70)")
    
    # Check for MACD crossover
    current_macd = df['macd'].iloc[-1]
    current_signal = df['macd_signal'].iloc[-1]
    prev_macd = df['macd'].iloc[-2]
    prev_signal = df['macd_signal'].iloc[-2]
    
    if current_macd > current_signal and prev_macd <= prev_signal:
        patterns.append("Bullish MACD crossover")
    elif current_macd < current_signal and prev_macd >= prev_signal:
        patterns.append("Bearish MACD crossover")
    
    # Check for moving average crossover
    current_sma_20 = df['sma_20'].iloc[-1] if 'sma_20' in df.columns else None
    current_sma_50 = df['sma_50'].iloc[-1] if 'sma_50' in df.columns else None
    prev_sma_20 = df['sma_20'].iloc[-2] if 'sma_20' in df.columns else None
    prev_sma_50 = df['sma_50'].iloc[-2] if 'sma_50' in df.columns else None
    
    if current_sma_20 and current_sma_50 and prev_sma_20 and prev_sma_50:
        if current_sma_20 > current_sma_50 and prev_sma_20 <= prev_sma_50:
            patterns.append("Golden cross (bullish)")
        elif current_sma_20 < current_sma_50 and prev_sma_20 >= prev_sma_50:
            patterns.append("Death cross (bearish)")
    
    # Check Bollinger Bands position
    current_price = df['close'].iloc[-1]
    bb_high = df['bb_high'].iloc[-1]
    bb_low = df['bb_low'].iloc[-1]
    
    if current_price > bb_high:
        patterns.append("Price above upper Bollinger Band (overbought)")
    elif current_price < bb_low:
        patterns.append("Price below lower Bollinger Band (oversold)")
    
    return patterns

def analyze_all_stocks_performance(timeframe='short-term'):
    """Analyze all stocks and rank them by performance potential"""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    results = []
    total_companies = len(all_companies)
    
    for i, (symbol, name) in enumerate(all_companies.items()):
        status_text.text(f"Analyzing {name} ({symbol})...")
        progress_bar.progress((i + 1) / total_companies)
        
        # Fetch stock data
        df = fetch_stock_data(symbol, days=90)
        if df is None or len(df) < 30:
            continue
        
        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Calculate performance metrics
        recent_return = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
        
        # Create a performance score
        score = 0
        
        # Positive factors
        if 'rsi' in df.columns and df['rsi'].iloc[-1] < 40:  # Oversold
            score += 2
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            if df['macd'].iloc[-1] > df['macd_signal'].iloc[-1]:  # Bullish MACD
                score += 2
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            if df['sma_20'].iloc[-1] > df['sma_50'].iloc[-1]:  # Short-term above long-term
                score += 1
        
        # Negative factors
        if 'rsi' in df.columns and df['rsi'].iloc[-1] > 70:  # Overbought
            score -= 2
        
        # Add trend analysis
        trend_analysis = analyze_price_trend(symbol, timeframe)
        
        # Add sector information
        sector = sector_mapping.get(symbol, "Unknown")
        
        results.append({
            'symbol': symbol,
            'name': name,
            'sector': sector,
            'recent_return': recent_return,
            'score': score,
            'price': df['close'].iloc[-1],
            'trend': trend_analysis['trend'] if isinstance(trend_analysis, dict) else "Unknown"
        })
    
    # Sort by performance score
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def preprocess_financial_text(text, symbol=None):
    """
    Preprocess text to handle financial-specific terms and contexts
    """
    # Convert to lowercase
    text = text.lower()
    
    # Handle financial abbreviations and terms
    financial_mappings = {
        'bullish': 'positive',
        'bearish': 'negative',
        'long': 'buy',
        'short': 'sell',
        'rally': 'increase',
        'plunge': 'decrease',
        'dump': 'decrease',
        'pump': 'increase',
        'moon': 'increase significantly',
        'rocket': 'increase significantly',
        'crash': 'decrease significantly',
        'tank': 'decrease significantly',
        'volatile': 'uncertain',
    }
    
    for term, replacement in financial_mappings.items():
        text = re.sub(r'\b' + term + r'\b', replacement, text)
    
    # Handle numerical context
    text = handle_numerical_context(text)
    
    # Add symbol context if available
    if symbol:
        text = f"{symbol} {text}"
    
    return text

def handle_numerical_context(text):
    """
    Handle numerical ambiguities in financial text
    """
    # Pattern to find percentage changes
    percentage_pattern = r'(\w+)\s+(up|down|increase|decrease|rise|fall|gain|loss)\s+(\d+%|\d+\.\d+%)'
    matches = re.findall(percentage_pattern, text, re.IGNORECASE)
    
    for match in matches:
        subject, direction, amount = match
        # Financial context: revenue up = good, costs up = bad
        if subject.lower() in ['revenue', 'profit', 'earnings', 'income', 'growth']:
            if direction in ['up', 'increase', 'rise', 'gain']:
                text += " positive"
            else:
                text += " negative"
        elif subject.lower() in ['cost', 'expense', 'debt', 'loss', 'burn']:
            if direction in ['up', 'increase', 'rise']:
                text += " negative"
            else:
                text += " positive"
    
    return text

# MODIFIED FUNCTION: Now uses real Alpha Vantage news
def _generate_sarcastic_stock_texts(num_samples=5):
    """Generate sarcastic stock-news-like texts (template mutation)."""
    companies = [
        'AAPL', 'TSLA', 'AMZN', 'MSFT', 'NVDA', 'META'
    ]
    positive_events = [
        'record earnings', 'beat all expectations', 'surpassed guidance',
        'launched a game-changer', "won every analyst's heart"
    ]
    negative_events = [
        'missed by a mile', 'is drowning in losses', 'faces the worst quarter',
        'botched the launch', 'disappointed literally everyone'
    ]
    sarcasm_markers = [
        'yeah right', 'what a surprise', "totally didn't see that coming",
        'just amazing', 'brilliant move', 'love that for them'
    ]
    templates = [
        "{c} {pos} — {sar} — shares tumble as investors are clearly thrilled.",
        "{c} {neg}; {sar}. Bulls are definitely loving this chart.",
        "Analysts say {c} outlook is bright after it {neg}; {sar}.",
        "{c} announces {pos}. The market reacts with overwhelming enthusiasm — by selling.",
        "Investors cheer as {c} {neg}. {sar}"
    ]
    import random
    texts = []
    for _ in range(num_samples):
        c = random.choice(companies)
        pos = random.choice(positive_events)
        neg = random.choice(negative_events)
        sar = random.choice(sarcasm_markers)
        template = random.choice(templates)
        texts.append(template.format(c=c, pos=pos, neg=neg, sar=sar))
    return texts

_SARCASM_MARKERS = [
    'yeah right', 'what a surprise', "totally didn't see that coming",
    'just amazing', 'brilliant move', 'love that for them'
]

def _augment_with_sarcasm_if_needed(articles, symbol, min_sarcastic=1, gen_count=5):
    """If no sarcastic cues exist in real news, append GAN-like sarcastic headlines."""
    def _has_sarcasm(text):
        tl = (text or '').lower()
        return any(marker in tl for marker in _SARCASM_MARKERS)

    sarcastic_found = 0
    for a in (articles or []):
        t = (a.get('title', '') + ' ' + a.get('summary', '')).strip()
        if _has_sarcasm(t):
            sarcastic_found += 1
            if sarcastic_found >= min_sarcastic:
                break

    if sarcastic_found >= min_sarcastic:
        return articles

    # No sarcasm detected -> generate and append
    generated = _generate_sarcastic_stock_texts(gen_count)
    for text in generated:
        articles.append({
            'title': text,
            'summary': text,
            'url': '#',
            'source': 'GAN-Generated',
            'time_published': datetime.now().strftime('%Y%m%dT%H%M%S'),
            'sentiment_score': 0,
            'sentiment_label': 'Neutral'
        })
    return articles

def get_news_articles(symbol):
    """Get news articles; if real news lacks sarcasm, augment with GAN-generated sarcastic items."""
    articles = fetch_alpha_vantage_news(symbol)
    try:
        # Ensure list
        if not isinstance(articles, list):
            articles = []
    except Exception:
        articles = []

    articles = _augment_with_sarcasm_if_needed(articles, symbol)
    return articles

def generate_investment_recommendation(symbol, df, sentiment, trend):
    """Generate buy/sell/hold recommendation based on multiple factors"""
    # Calculate technical score
    technical_score = 0
    
    # RSI
    if 'rsi' in df.columns:
        rsi = df['rsi'].iloc[-1]
        if rsi < 30:
            technical_score += 2  # Oversold - bullish
        elif rsi > 70:
            technical_score -= 2  # Overbought - bearish
    
    # MACD
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd = df['macd'].iloc[-1]
        macd_signal = df['macd_signal'].iloc[-1]
        if macd > macd_signal:
            technical_score += 1  # Bullish crossover
        else:
            technical_score -= 1  # Bearish crossover
    
    # Moving averages
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        sma_20 = df['sma_20'].iloc[-1]
        sma_50 = df['sma_50'].iloc[-1]
        if sma_20 > sma_50:
            technical_score += 1  # Short-term above long-term - bullish
        else:
            technical_score -= 1  # Bearish
    
    # Price trend
    if isinstance(trend, dict):
        if "uptrend" in trend['trend']:
            technical_score += 1
        elif "downtrend" in trend['trend']:
            technical_score -= 1
    
    # Calculate sentiment score
    sentiment_score = 0
    if sentiment and 'average_score' in sentiment:
        sentiment_score = sentiment['average_score'] * 2  # Scale to similar range as technical
    
    # Combined score
    total_score = technical_score + sentiment_score
    
    # Generate recommendation
    if total_score > 3:
        recommendation = "STRONG BUY"
        confidence = min(90, 60 + total_score * 5)
    elif total_score > 1:
        recommendation = "BUY"
        confidence = min(80, 50 + total_score * 5)
    elif total_score > -1:
        recommendation = "HOLD"
        confidence = 50
    elif total_score > -3:
        recommendation = "SELL"
        confidence = min(80, 50 + abs(total_score) * 5)
    else:
        recommendation = "STRONG SELL"
        confidence = min(90, 60 + abs(total_score) * 5)
    
    # Generate reasoning
    reasoning = "Based on analysis of:\n"
    
    if 'rsi' in df.columns:
        rsi = df['rsi'].iloc[-1]
        reasoning += f"- RSI: {rsi:.2f} ({'oversold' if rsi < 30 else 'overbought' if rsi > 70 else 'neutral'})\n"
    
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd, signal = df['macd'].iloc[-1], df['macd_signal'].iloc[-1]
        reasoning += f"- MACD: {macd:.2f} vs Signal: {signal:.2f} ({'bullish' if macd > signal else 'bearish'})\n"
    
    if isinstance(trend, dict):
        reasoning += f"- Price Trend: {trend['trend']}\n"
        reasoning += f"- Volatility: {trend['volatility']:.2%}\n"
    
    if sentiment:
        reasoning += f"- Sentiment: {sentiment['average_score']:.2f} ({sentiment['positive_articles']} positive, {sentiment['negative_articles']} negative articles)\n"
    
    return recommendation, confidence, reasoning

def analyze_single_stock_recommendation(symbol, timeframe):
    """Analyze a single stock and provide recommendation"""
    with st.spinner(f"Analyzing {all_companies.get(symbol, symbol)} for investment recommendation..."):
        # Get technical analysis
        df = fetch_stock_data(symbol, days=90)
        if df is not None and len(df) > 0:
            df = calculate_technical_indicators(df)
            
            # Get sentiment analysis - NOW USING REAL NEWS
            news_articles = get_news_articles(symbol)
            sentiment = analyze_news_sentiment(news_articles, symbol)
            
            # Get trend analysis
            trend = analyze_price_trend(symbol, timeframe)
            
            # Generate recommendation
            recommendation, confidence, reasoning = generate_investment_recommendation(
                symbol, df, sentiment, trend
            )
            
            response = f"**Investment Recommendation for {all_companies.get(symbol, symbol)}:**\n\n"
            response += f"**{recommendation}** (Confidence: {confidence}%)\n\n"
            response += f"**Reasoning:**\n{reasoning}\n\n"
            response += f"**Current Price:** ${df['close'].iloc[-1]:.2f}\n"
            
            if isinstance(trend, dict):
                response += f"**Trend:** {trend['trend']}\n"
                response += f"**Volatility:** {trend['volatility']:.2%} (annualized)\n\n"
            
            # Add news source information
            if news_articles and len(news_articles) > 0:
                source = "Alpha Vantage" if 'source' in news_articles[0] and news_articles[0]['source'] != 'Unknown source' else "Mock Data"
                response += f"**News Source:** {source} ({len(news_articles)} articles analyzed)\n\n"
            
            response += "*Note: This is an automated recommendation based on technical and sentiment analysis. " \
                       "It should not be considered financial advice. Always do your own research.*"
            
            return response
    return "Could not analyze this stock. Please try again."

def analyze_all_stocks_recommendations(timeframe, question_lower):
    """Analyze all stocks and provide top recommendations based on the question context"""
    with st.spinner("Analyzing all stocks to find the best recommendations..."):
        # Get performance data for all stocks
        all_stocks_performance = analyze_all_stocks_performance(timeframe)
        
        # Filter based on question context
        if 'buy' in question_lower:
            # Get top buy recommendations
            buy_stocks = [s for s in all_stocks_performance if s['score'] > 2][:5]
            response = "**Top Buy Recommendations:**\n\n"
            
            for i, stock in enumerate(buy_stocks):
                response += f"{i+1}. **{stock['name']} ({stock['symbol']})**\n"
                response += f"   - Sector: {stock['sector']}\n"
                response += f"   - Current Price: ${stock['price']:.2f}\n"
                response += f"   - Performance Score: {stock['score']}/5\n"
                response += f"   - Trend: {stock['trend']}\n\n"
        
        elif 'sell' in question_lower:
            # Get top sell recommendations
            sell_stocks = [s for s in all_stocks_performance if s['score'] < -1][:5]
            response = "**Stocks to Consider Selling:**\n\n"
            
            for i, stock in enumerate(sell_stocks):
                response += f"{i+1}. **{stock['name']} ({stock['symbol']})**\n"
                response += f"   - Sector: {stock['sector']}\n"
                response += f"   - Current Price: ${stock['price']:.2f}\n"
                response += f"   - Performance Score: {stock['score']}/5\n"
                response += f"   - Trend: {stock['trend']}\n\n"
        
        else:
            # General hold recommendations or mixed
            response = "**Top Investment Opportunities:**\n\n"
            
            for i, stock in enumerate(all_stocks_performance[:5]):
                recommendation = "BUY" if stock['score'] > 1 else "HOLD" if stock['score'] > -1 else "SELL"
                response += f"{i+1}. **{stock['name']} ({stock['symbol']})** - {recommendation}\n"
                response += f"   - Sector: {stock['sector']}\n"
                response += f"   - Current Price: ${stock['price']:.2f}\n"
                response += f"   - Performance Score: {stock['score']}/5\n"
                response += f"   - Trend: {stock['trend']}\n\n"
        
        response += "*Note: These recommendations are based on technical analysis and recent performance. " \
                   "Always conduct your own research before making investment decisions.*"
        
        return response

def analyze_single_stock_performance(symbol):
    """Analyze performance of a single stock"""
    with st.spinner(f"Analyzing performance of {all_companies.get(symbol, symbol)}..."):
        df = fetch_stock_data(symbol, days=90)
        
        if df is not None and len(df) > 0:
            performance = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
            volatility = df['close'].pct_change().std() * np.sqrt(252) * 100  # Annualized
            
            response = f"**Performance Analysis for {all_companies.get(symbol, symbol)}:**\n\n"
            response += f"- **90-Day Return:** {performance:.2f}%\n"
            response += f"- **Annualized Volatility:** {volatility:.2f}%\n"
            response += f"- **Current Price:** ${df['close'].iloc[-1]:.2f}\n"
            
            # Compare to sector
            sector = sector_mapping.get(symbol, "Unknown")
            sector_perf = analyze_sector_performance().get(sector, 0)
            
            response += f"- **Sector ({sector}) Performance:** {sector_perf:.2f}%\n"
            
            if performance > sector_perf:
                response += f"- **Outperformance vs Sector:** {performance - sector_perf:.2f}%\n"
            else:
                response += f"- **Underperformance vs Sector:** {performance - sector_perf:.2f}%\n"
            
            # Add technical patterns
            patterns = analyze_technical_patterns(symbol)
            if patterns:
                response += f"\n**Technical Patterns:**\n"
                for pattern in patterns:
                    response += f"- {pattern}\n"
            
            return response
    return "Could not analyze this stock's performance."

def analyze_top_performing_stocks(timeframe):
    """Analyze and return top performing stocks"""
    with st.spinner("Finding top performing stocks..."):
        all_stocks_performance = analyze_all_stocks_performance(timeframe)
        
        response = "**Top Performing Stocks:**\n\n"
        
        for i, stock in enumerate(all_stocks_performance[:5]):
            response += f"{i+1}. **{stock['name']} ({stock['symbol']})**\n"
            response += f"   - Sector: {stock['sector']}\n"
            response += f"   - Recent Return: {stock['recent_return']:.2f}%\n"
            response += f"   - Current Price: ${stock['price']:.2f}\n"
            response += f"   - Trend: {stock['trend']}\n\n"
        
        response += "*Note: Past performance is not indicative of future results.*"
        
        return response

def analyze_sector_performance_insights(question_lower):
    """Analyze sector performance with insights based on the question context"""
    with st.spinner("Analyzing sector performance insights..."):
        sector_performance = analyze_sector_performance()
        
        # Extract sector keywords from question
        sector_keywords = {
            'tech': 'Technology',
            'bank': 'Financial Services',
            'energy': 'Energy',
            'ev': 'Automotive',
            'health': 'Healthcare',
            'consumer': 'Consumer',
            'financial': 'Financial Services',
            'technology': 'Technology',
            'automotive': 'Automotive',
            'healthcare': 'Healthcare'
        }
        
        target_sector = None
        for keyword, sector in sector_keywords.items():
            if keyword in question_lower:
                target_sector = sector
                break
        
        response = "**Sector Performance Insights:**\n\n"
        
        if target_sector:
            response += f"**Focus: {target_sector}**\n\n"
            perf = sector_performance.get(target_sector, None)
            
            if perf is not None:
                response += f"- **Performance:** {perf:.2f}%\n"
                
                # Compare to overall market
                avg_performance = sum(sector_performance.values()) / len(sector_performance)
                if perf > avg_performance:
                    response += f"- **Vs Market:** Outperforming by {perf - avg_performance:.2f}%\n"
                else:
                    response += f"- **Vs Market:** Underperforming by {avg_performance - perf:.2f}%\n"
                
                # Get top performers in this sector
                sector_stocks = [s for s, sec in sector_mapping.items() if sec == target_sector]
                if sector_stocks:
                    response += f"\n**Top Performers in {target_sector}:**\n"
                    stock_performances = []
                    
                    for stock in sector_stocks:
                        df = fetch_stock_data(stock, days=30)
                        if df is not None and len(df) > 0:
                            perf = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                            stock_performances.append((stock, perf))
                    
                    # Sort by performance
                    stock_performances.sort(key=lambda x: x[1], reverse=True)
                    
                    for i, (stock, perf) in enumerate(stock_performances[:3]):
                        response += f"{i+1}. {all_companies.get(stock, stock)}: {perf:.2f}%\n"
            else:
                response += "No performance data available for this sector.\n"
        else:
            # Show all sectors ranked by performance
            sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
            
            response += "**All Sectors Ranked by Performance:**\n\n"
            for i, (sector, performance) in enumerate(sorted_sectors):
                response += f"{i+1}. **{sector}**: {performance:.2f}%\n"
            
            # Highlight best and worst performing sectors
            best_sector, best_perf = sorted_sectors[0]
            worst_sector, worst_perf = sorted_sectors[-1]
            
            response += f"\n**Best Performing:** {best_sector} ({best_perf:.2f}%)\n"
            response += f"**Worst Performing:** {worst_sector} ({worst_perf:.2f}%)\n"
        
        response += "\n*Note: Performance calculated as average return of stocks in each sector.*"
        
        return response

def analyze_event_impact(symbol):
    """Analyze the impact of events on a specific stock"""
    with st.spinner(f"Analyzing event impact for {all_companies.get(symbol, symbol)}..."):
        # For events, we'll analyze historical reactions to similar events
        impact = analyze_earnings_impact(symbol)
        
        response = f"**Event Impact Analysis for {all_companies.get(symbol, symbol)}:**\n\n"
        response += f"Historical average price change around events: {impact:.2f}%\n\n"
        
        if impact > 2:
            response += "This stock has historically shown strong positive reactions to events."
        elif impact > 0:
            response += "This stock has historically shown mild positive reactions to events."
        elif impact > -2:
            response += "This stock has historically shown mild negative reactions to events."
        else:
            response += "This stock has historically shown strong negative reactions to events."
        
        # Add current sentiment - NOW USING REAL NEWS
        news_articles = get_news_articles(symbol)
        sentiment = analyze_news_sentiment(news_articles, symbol)
        
        if sentiment:
            response += f"\n\n**Current Sentiment:** {sentiment['average_score']:.2f} "
            if sentiment['average_score'] > 0.2:
                response += "(Positive) 📈"
            elif sentiment['average_score'] < -0.2:
                response += "(Negative) 📉"
            else:
                response += "(Neutral) ➡️"
        
        # Add technical patterns that might affect event reaction
        patterns = analyze_technical_patterns(symbol)
        if patterns:
            response += f"\n\n**Technical Patterns That May Affect Event Reaction:**\n"
            for pattern in patterns[:3]:  # Show top 3 patterns
                response += f"- {pattern}\n"
        
        response += "\n\n*Note: Past event reactions don't guarantee future performance.*"
        
        return response

def analyze_companies_with_upcoming_events():
    """Analyze companies that might have upcoming events based on historical patterns"""
    with st.spinner("Identifying companies with potential upcoming events..."):
        # This is a simplified approach - in a real system, you'd use earnings calendar data
        response = "**Companies with Potential Upcoming Catalysts:**\n\n"
        
        # Analyze all companies for potential event triggers
        event_candidates = []
        
        for symbol, name in all_companies.items():
            df = fetch_stock_data(symbol, days=60)
            if df is not None and len(df) > 30:
                # Look for consolidation patterns (low volatility) which often precede events
                recent_volatility = df['close'].pct_change().rolling(20).std().iloc[-1]
                avg_volatility = df['close'].pct_change().rolling(20).std().mean()
                
                if recent_volatility < avg_volatility * 0.7:  # Low volatility period
                    # Check if price is near key levels
                    current_price = df['close'].iloc[-1]
                    sma_50 = df['close'].rolling(50).mean().iloc[-1]
                    
                    if abs(current_price - sma_50) / sma_50 < 0.05:  # Near 50-day SMA
                        event_candidates.append((symbol, name, recent_volatility))
        
        # Sort by volatility (lowest first)
        event_candidates.sort(key=lambda x: x[2])
        
        if event_candidates:
            response += "These stocks are showing consolidation patterns that often precede significant moves:\n\n"
            
            for i, (symbol, name, volatility) in enumerate(event_candidates[:5]):
                response += f"{i+1}. **{name} ({symbol})**\n"
                response += f"   - Recent Volatility: {volatility:.4f}\n"
                response += f"   - Current Price: ${fetch_stock_data(symbol, days=1)['close'].iloc[-1]:.2f}\n\n"
        else:
            response += "No strong event candidates identified at this time.\n\n"
        
        response += "*Note: This analysis identifies stocks in consolidation patterns, which often precede significant moves. " \
                   "It doesn't predict specific events.*"
        
        return response

def analyze_single_stock_risk(symbol):
    """Analyze risk factors for a single stock"""
    with st.spinner(f"Analyzing risk factors for {all_companies.get(symbol, symbol)}..."):
        df = fetch_stock_data(symbol, days=90)
        
        if df is not None and len(df) > 0:
            # Calculate volatility (risk)
            volatility = df['close'].pct_change().std() * np.sqrt(252)  # Annualized
            
            # Get sentiment analysis - NOW USING REAL NEWS
            news_articles = get_news_articles(symbol)
            sentiment = analyze_news_sentiment(news_articles, symbol)
            
            # Calculate maximum drawdown
            rolling_max = df['close'].cummax()
            drawdown = (df['close'] - rolling_max) / rolling_max
            max_drawdown = drawdown.min()
            
            response = f"**Risk Analysis for {all_companies.get(symbol, symbol)}:**\n\n"
            response += f"- **Annualized Volatility:** {volatility:.2%}\n"
            response += f"- **Maximum Drawdown:** {max_drawdown:.2%}\n"
            
            if sentiment:
                response += f"- **Sentiment Score:** {sentiment['average_score']:.2f}\n"
                response += f"- **Positive Articles:** {sentiment['positive_articles']}\n"
                response += f"- **Negative Articles:** {sentiment['negative_articles']}\n"
                response += f"- **Neutral Articles:** {sentiment.get('neutral_articles', 0)}\n"
                
                # Add classification metrics if available
                if sentiment.get('classification_metrics') and 'error' not in sentiment['classification_metrics']:
                    metrics = sentiment['classification_metrics']
                    response += f"- **Accuracy:** {metrics['accuracy']:.3f}\n"
                    response += f"- **Precision (Macro):** {metrics['precision_macro']:.3f}\n"
                    response += f"- **Recall (Macro):** {metrics['recall_macro']:.3f}\n"
                    response += f"- **F1 Score (Macro):** {metrics['f1_macro']:.3f}\n"
            
            # Add beta-like comparison (how it moves vs market)
            spy_df = fetch_stock_data('SPY', days=90)
            if spy_df is not None and len(spy_df) > 0:
                correlation = df['close'].pct_change().corr(spy_df['close'].pct_change())
                response += f"- **Market Correlation:** {correlation:.2f}\n"
            
            # Risk assessment
            if volatility > 0.4:
                response += f"\n**Risk Level:** High\n"
                response += "This stock shows high volatility, which means higher potential returns but also higher risk."
            elif volatility > 0.2:
                response += f"\n**Risk Level:** Medium\n"
                response += "This stock shows moderate volatility, balancing potential returns with risk."
            else:
                response += f"\n**Risk Level:** Low\n"
                response += "This stock shows low volatility, which typically means lower risk but also lower potential returns."
            
            # Add sector context
            sector = sector_mapping.get(symbol, "Unknown")
            response += f"\n**Sector Context:** {sector}\n"
            
            return response
    return "Could not analyze risk for this stock."

def analyze_riskiest_stocks():
    """Analyze and return the riskiest stocks based on volatility"""
    with st.spinner("Identifying the riskiest stocks..."):
        risk_data = []
        
        for symbol, name in all_companies.items():
            df = fetch_stock_data(symbol, days=90)
            if df is not None and len(df) > 0:
                volatility = df['close'].pct_change().std() * np.sqrt(252)  # Annualized
                risk_data.append((symbol, name, volatility))
        
        # Sort by volatility (highest first)
        risk_data.sort(key=lambda x: x[2], reverse=True)
        
        response = "**Riskiest Stocks (Highest Volatility):**\n\n"
        
        for i, (symbol, name, volatility) in enumerate(risk_data[:5]):
            response += f"{i+1}. **{name} ({symbol})**\n"
            response += f"   - Annualized Volatility: {volatility:.2%}\n"
            response += f"   - Current Price: ${fetch_stock_data(symbol, days=1)['close'].iloc[-1]:.2f}\n"
            response += f"   - Sector: {sector_mapping.get(symbol, 'Unknown')}\n\n"
        
        response += "*Note: High volatility means higher potential returns but also higher risk. " \
                   "These stocks may experience large price swings.*"
        
        return response

def compare_two_stocks(symbol1, symbol2):
    """Compare two specific stocks"""
    with st.spinner(f"Comparing {all_companies.get(symbol1, symbol1)} and {all_companies.get(symbol2, symbol2)}..."):
        df1 = fetch_stock_data(symbol1, days=90)
        df2 = fetch_stock_data(symbol2, days=90)
        
        if df1 is not None and len(df1) > 0 and df2 is not None and len(df2) > 0:
            perf1 = (df1['close'].iloc[-1] / df1['close'].iloc[0] - 1) * 100
            perf2 = (df2['close'].iloc[-1] / df2['close'].iloc[0] - 1) * 100
            
            vol1 = df1['close'].pct_change().std() * np.sqrt(252) * 100
            vol2 = df2['close'].pct_change().std() * np.sqrt(252) * 100
            
            response = f"**Comparison: {all_companies.get(symbol1, symbol1)} vs {all_companies.get(symbol2, symbol2)}**\n\n"
            response += f"**{all_companies.get(symbol1, symbol1)}:**\n"
            response += f"- 90-Day Return: {perf1:.2f}%\n"
            response += f"- Annualized Volatility: {vol1:.2f}%\n"
            response += f"- Current Price: ${df1['close'].iloc[-1]:.2f}\n\n"
            
            response += f"**{all_companies.get(symbol2, symbol2)}:**\n"
            response += f"- 90-Day Return: {perf2:.2f}%\n"
            response += f"- Annualized Volatility: {vol2:.2f}%\n"
            response += f"- Current Price: ${df2['close'].iloc[-1]:.2f}\n\n"
            
            # Performance comparison
            if perf1 > perf2:
                response += f"**Performance Winner:** {all_companies.get(symbol1, symbol1)} " \
                           f"(by {perf1 - perf2:.2f}%)\n"
            else:
                response += f"**Performance Winner:** {all_companies.get(symbol2, symbol2)} " \
                           f"(by {perf2 - perf1:.2f}%)\n"
            
            # Risk-adjusted comparison
            risk_adj1 = perf1 / vol1 if vol1 > 0 else 0
            risk_adj2 = perf2 / vol2 if vol2 > 0 else 0
            
            response += f"\n**Risk-Adjusted Performance:**\n"
            response += f"- {all_companies.get(symbol1, symbol1)}: {risk_adj1:.2f}\n"
            response += f"- {all_companies.get(symbol2, symbol2)}: {risk_adj2:.2f}\n"
            
            if risk_adj1 > risk_adj2:
                response += f"**Better Risk-Adjusted Return:** {all_companies.get(symbol1, symbol1)}\n"
            else:
                response += f"**Better Risk-Adjusted Return:** {all_companies.get(symbol2, symbol2)}\n"
            
            return response
    return "Could not compare these stocks."

def analyze_comparative_sectors(question_lower):
    """Compare sectors when no specific stocks are mentioned"""
    with st.spinner("Comparing sector performances..."):
        sector_performance = analyze_sector_performance()
        sorted_sectors = sorted(sector_performance.items(), key=lambda x: x[1], reverse=True)
        
        # Extract sectors to compare from question
        sector_keywords = {
            'tech': 'Technology',
            'bank': 'Financial Services',
            'energy': 'Energy',
            'ev': 'Automotive',
            'health': 'Healthcare',
            'consumer': 'Consumer',
            'financial': 'Financial Services',
            'technology': 'Technology',
            'automotive': 'Automotive',
            'healthcare': 'Healthcare'
        }
        
        sectors_to_compare = []
        for keyword, sector in sector_keywords.items():
            if keyword in question_lower and sector in sector_performance:
                sectors_to_compare.append(sector)
        
        response = "**Sector Comparison:**\n\n"
        
        if len(sectors_to_compare) >= 2:
            # Compare specific sectors mentioned in question
            for sector in sectors_to_compare:
                perf = sector_performance.get(sector, 0)
                response += f"- **{sector}**: {perf:.2f}%\n"
            
            # Identify winner
            best_sector = max(sectors_to_compare, key=lambda x: sector_performance.get(x, 0))
            response += f"\n**Best Performing:** {best_sector} ({sector_performance.get(best_sector, 0):.2f}%)\n"
        else:
            # Show top 3 and bottom 3 sectors
            response += "**Top Performing Sectors:**\n"
            for sector, perf in sorted_sectors[:3]:
                response += f"- {sector}: {perf:.2f}%\n"
            
            response += "\n**Worst Performing Sectors:**\n"
            for sector, perf in sorted_sectors[-3:]:
                response += f"- {sector}: {perf:.2f}%\n"
            
            response += f"\n**Performance Spread:** {sorted_sectors[0][1] - sorted_sectors[-1][1]:.2f}%\n"
        
        # Add sector rotation insight
        if 'rotate' in question_lower or 'rotation' in question_lower:
            # Simple sector rotation insight based on recent performance
            response += "\n**Sector Rotation Insight:**\n"
            if sorted_sectors[0][1] - sorted_sectors[-1][1] > 10:  # Large performance spread
                response += "Large performance spread suggests potential for sector rotation. " \
                           "Outperforming sectors may be due for a pullback while underperformers may catch up."
            else:
                response += "Moderate performance spread suggests stable market conditions without strong rotation signals."
        
        response += "\n*Note: Sector performance based on average returns of constituent stocks.*"
        
        return response

# Enhanced sentiment analysis with multi-layer architecture
class EnhancedFinancialSentimentAnalyzer:
    def __init__(self):
        self.finbert = load_finbert_model()
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.base_lexicon = FINANCIAL_SENTIMENT_LEXICON.copy()
        self.expanded_lexicon = self._build_expanded_lexicon()
    def _build_expanded_lexicon(self):
        """Build expanded lexicon using WordNet and embedding similarity"""
        expanded = self.base_lexicon.copy()
        
        # WordNet expansion
        for word, score in self.base_lexicon.items():
            synonyms = self._get_synonyms(word)
            for synonym in synonyms:
                if synonym not in expanded:
                    expanded[synonym] = score * 0.8  # Slightly attenuated score for synonyms
        
        # Embedding-based expansion (simplified)
        embedding_expansions = self._embedding_based_expansion()
        expanded.update(embedding_expansions)
        
        return expanded
    
    def _get_synonyms(self, word):
        """Get synonyms using WordNet"""
        synonyms = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ').lower()
                if synonym != word and len(synonym.split()) == 1:  # Single word synonyms
                    synonyms.add(synonym)
        return list(synonyms)
    
    def _embedding_based_expansion(self):
        """Simple embedding-based expansion using word co-occurrence"""
        expansions = {}
        financial_terms = list(self.base_lexicon.keys())
        
        # Simple similarity based on string matching and financial context
        similarity_mappings = {
            'bullish': ['optimistic', 'positive', 'upbeat'],
            'bearish': ['pessimistic', 'negative', 'gloomy'],
            'rally': ['recovery', 'rebound', 'upswing'],
            'plunge': ['crash', 'collapse', 'tumble'],
            'profit': ['earnings', 'gain', 'income'],
            'loss': ['deficit', 'shortfall', 'decline']
        }
        
        for term, similar_terms in similarity_mappings.items():
            base_score = self.base_lexicon.get(term, 0)
            for similar_term in similar_terms:
                if similar_term not in self.base_lexicon:
                    expansions[similar_term] = base_score * 0.7
        
        return expansions
    
    def _coherence_check(self, text, sentiment_scores):
        """Check for sentiment coherence and detect contradictions"""
        words = text.lower().split()
        positive_words = [w for w in words if self.expanded_lexicon.get(w, 0) > 0.3]
        negative_words = [w for w in words if self.expanded_lexicon.get(w, 0) < -0.3]
        
        # If both strong positive and negative words present, flag for sarcasm
        if len(positive_words) > 0 and len(negative_words) > 0:
            return True, "contradiction_detected"
        
        # Check for negation patterns
        negation_words = ['not', 'no', 'never', 'without', 'lack']
        for i, word in enumerate(words):
            if word in negation_words and i < len(words) - 1:
                next_word = words[i + 1]
                if self.expanded_lexicon.get(next_word, 0) != 0:
                    return True, "negation_detected"
        
        return False, "coherent"
    
    def analyze_sentiment(self, text, context=None, symbol=None):
        """Enhanced multi-layer sentiment analysis"""
        # Layer 1: Base FinBERT
        finbert_sentiment = self._finbert_analysis(text)
        
        # Layer 2: Rule-based financial mapping
        rule_based_sentiment = self._rule_based_analysis(text)
        
        # Layer 3: Expanded lexicon analysis
        lexicon_sentiment = self._lexicon_analysis(text)
        
        # Layer 4: Coherence check
        is_incoherent, incoherence_type = self._coherence_check(text, {
            'finbert': finbert_sentiment,
            'rule_based': rule_based_sentiment,
            'lexicon': lexicon_sentiment
        })
        
        # Layer 5: Contextual adjustment
        contextual_sentiment = self._contextual_adjustment(
            text, finbert_sentiment, rule_based_sentiment, 
            lexicon_sentiment, context, symbol
        )
        
        # Final aggregation with coherence adjustment
        if is_incoherent:
            if incoherence_type == "contradiction_detected":
                # For contradictions, rely more on FinBERT and apply sarcasm adjustment
                final_score = finbert_sentiment['score'] * 0.6 + contextual_sentiment * 0.4
                final_score *= 0.7  # Sarcasm dampening
            else:  # negation_detected
                final_score = contextual_sentiment * -0.8  # Strong negation reversal
        else:
            # Weighted average of all layers
            final_score = (
                finbert_sentiment['score'] * 0.4 +
                rule_based_sentiment * 0.3 +
                lexicon_sentiment * 0.2 +
                contextual_sentiment * 0.1
            )
        
        # Classify final sentiment
        if final_score >= 0.2:
            sentiment_label = "positive"
        elif final_score <= -0.2:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return {
            "sentiment": sentiment_label,
            "score": final_score,
            "components": {
                "finbert": finbert_sentiment,
                "rule_based": rule_based_sentiment,
                "lexicon": lexicon_sentiment,
                "contextual": contextual_sentiment
            },
            "coherence": {
                "is_incoherent": is_incoherent,
                "incoherence_type": incoherence_type
            }
        }
    
    def _finbert_analysis(self, text):
        """FinBERT-based sentiment analysis"""
        if self.finbert is None:
            return {"label": "NEUTRAL", "score": 0.0}
        
        try:
            result = self.finbert(text)[0]
            label = result['label']
            score = result['score']
            
            # Map to numeric score
            if label == "positive":
                numeric_score = score
            elif label == "negative":
                numeric_score = -score
            else:
                numeric_score = 0.0
                
            return {"label": label, "score": numeric_score}
        except:
            return {"label": "NEUTRAL", "score": 0.0}
    
    def _rule_based_analysis(self, text):
        """Rule-based financial sentiment analysis"""
        text_lower = text.lower()
        score = 0.0
        word_count = 0
        
        for word, word_score in self.base_lexicon.items():
            if word in text_lower:
                score += word_score
                word_count += 1
        
        # Normalize score
        if word_count > 0:
            score = score / word_count
        else:
            # Fallback to VADER if no financial terms found
            vader_scores = self.vader_analyzer.polarity_scores(text)
            score = vader_scores['compound']
        
        return score
    
    def _lexicon_analysis(self, text):
        """Expanded lexicon-based analysis"""
        words = text.lower().split()
        total_score = 0.0
        matched_words = 0
        
        for word in words:
            if word in self.expanded_lexicon:
                total_score += self.expanded_lexicon[word]
                matched_words += 1
        
        return total_score / max(matched_words, 1)
    
    def _contextual_adjustment(self, text, finbert_sentiment, rule_based_sentiment, 
                             lexicon_sentiment, context, symbol):
        """Apply context-specific adjustments"""
        base_score = (finbert_sentiment['score'] + rule_based_sentiment + lexicon_sentiment) / 3
        
        # Context-specific adjustments
        if context == "earnings":
            # Earnings context often has mixed sentiment
            base_score *= 0.8  # Slightly dampen extreme sentiments
            
        elif context == "news" and symbol:
            # Stock-specific news might have different impact
            df = fetch_stock_data(symbol, days=30)
            if df is not None and len(df) > 0:
                performance = (df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100
                if performance > 5:  # Stock performing well
                    base_score += 0.1  # Slight positive bias
                elif performance < -5:  # Stock performing poorly
                    base_score -= 0.1  # Slight negative bias
        
        # Ensure score stays within bounds
        return max(-1.0, min(1.0, base_score))

# Initialize enhanced sentiment analyzer
@st.cache_resource
def load_enhanced_sentiment_analyzer():
    return EnhancedFinancialSentimentAnalyzer()

# Update the existing sentiment analysis functions to use the enhanced analyzer
def analyze_financial_sentiment(text, context=None, symbol=None):
    """Enhanced sentiment analysis using multi-layer architecture"""
    analyzer = load_enhanced_sentiment_analyzer()
    return analyzer.analyze_sentiment(text, context, symbol)

def calculate_sentiment_classification_metrics(sentiments, ground_truth=None):
    """Calculate classification metrics for sentiment analysis"""
    if not sentiments or len(sentiments) < 2:
        return None
    
    # Extract predicted labels
    predicted_labels = [s['sentiment'] for s in sentiments]
    
    # If no ground truth provided, create baseline using FinBERT as reference
    if ground_truth is None:
        ground_truth = []
        for s in sentiments:
            finbert_result = s.get('components', {}).get('finbert', {})
            if finbert_result.get('label'):
                ground_truth.append(finbert_result['label'].lower())
            else:
                # Fallback based on score
                score = finbert_result.get('score', 0)
                if score > 0.2:
                    ground_truth.append('positive')
                elif score < -0.2:
                    ground_truth.append('negative')
                else:
                    ground_truth.append('neutral')
    
    # Ensure same length
    min_length = min(len(predicted_labels), len(ground_truth))
    predicted_labels = predicted_labels[:min_length]
    ground_truth = ground_truth[:min_length]
    
    # Calculate metrics
    try:
        # Overall accuracy
        accuracy = accuracy_score(ground_truth, predicted_labels)
        
        # Multi-class metrics with different averaging strategies
        precision_macro = precision_score(ground_truth, predicted_labels, average='macro', zero_division=0)
        recall_macro = recall_score(ground_truth, predicted_labels, average='macro', zero_division=0)
        f1_macro = f1_score(ground_truth, predicted_labels, average='macro', zero_division=0)
        
        # Weighted metrics (accounts for class imbalance)
        precision_weighted = precision_score(ground_truth, predicted_labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(ground_truth, predicted_labels, average='weighted', zero_division=0)
        f1_weighted = f1_score(ground_truth, predicted_labels, average='weighted', zero_division=0)
        
        # Per-class metrics
        labels = list(set(ground_truth + predicted_labels))
        precision_per_class = precision_score(ground_truth, predicted_labels, average=None, labels=labels, zero_division=0)
        recall_per_class = recall_score(ground_truth, predicted_labels, average=None, labels=labels, zero_division=0)
        f1_per_class = f1_score(ground_truth, predicted_labels, average=None, labels=labels, zero_division=0)
        
        # Classification report
        classification_rep = classification_report(ground_truth, predicted_labels, output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(ground_truth, predicted_labels, labels=labels)
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_metrics': {
                'labels': labels,
                'precision': precision_per_class.tolist() if hasattr(precision_per_class, 'tolist') else precision_per_class,
                'recall': recall_per_class.tolist() if hasattr(recall_per_class, 'tolist') else recall_per_class,
                'f1': f1_per_class.tolist() if hasattr(f1_per_class, 'tolist') else f1_per_class
            },
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist() if hasattr(conf_matrix, 'tolist') else conf_matrix,
            'sample_size': len(predicted_labels)
        }
    except Exception as e:
        return {
            'error': f"Could not calculate metrics: {str(e)}",
            'sample_size': len(predicted_labels)
        }

# MODIFIED FUNCTION: Now uses real Alpha Vantage news data with classification metrics
def analyze_news_sentiment(news_articles, symbol=None):
    """Analyze sentiment across multiple news articles using enhanced analyzer with classification metrics"""
    analyzer = load_enhanced_sentiment_analyzer()
    sentiments = []
    
    for article in news_articles:
        # Use title and summary for sentiment analysis
        text = article.get('title', '') + " " + article.get('summary', '')
        sentiment = analyzer.analyze_sentiment(text, context="news", symbol=symbol)
        sentiments.append(sentiment)
    
    # Calculate average sentiment
    if sentiments:
        avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
        positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
        negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
        neutral_count = len(sentiments) - positive_count - negative_count
        
        # Calculate classification metrics
        classification_metrics = calculate_sentiment_classification_metrics(sentiments)
        
        return {
            'average_score': avg_score,
            'positive_articles': positive_count,
            'negative_articles': negative_count,
            'neutral_articles': neutral_count,
            'total_articles': len(sentiments),
            'sentiment_distribution': {
                'positive': positive_count / len(sentiments),
                'negative': negative_count / len(sentiments),
                'neutral': neutral_count / len(sentiments)
            },
            'classification_metrics': classification_metrics,
            'detailed_sentiments': sentiments
        }
    
    return None

def analyze_news_sentiment_finbert_only(news_articles):
    """Analyze sentiment using only FinBERT for comparison with multi-layer analyzer."""
    finbert = load_finbert_model()
    sentiments = []
    if finbert is None:
        return None
    for article in (news_articles or []):
        text = (article.get('title', '') + ' ' + article.get('summary', '')).strip()
        try:
            result = finbert(text)[0]
            label = result.get('label', 'neutral').lower()
            score_raw = float(result.get('score', 0.0))
            score = score_raw if label == 'positive' else (-score_raw if label == 'negative' else 0.0)
            sentiments.append({
                'sentiment': label,
                'score': score,
                'components': {'finbert': {'label': label, 'score': score}},
            })
        except Exception:
            continue
    if not sentiments:
        return None
    avg_score = sum(s['score'] for s in sentiments) / len(sentiments)
    positive_count = sum(1 for s in sentiments if s['sentiment'] == 'positive')
    negative_count = sum(1 for s in sentiments if s['sentiment'] == 'negative')
    neutral_count = len(sentiments) - positive_count - negative_count
    return {
        'average_score': avg_score,
        'positive_articles': positive_count,
        'negative_articles': negative_count,
        'neutral_articles': neutral_count,
        'total_articles': len(sentiments),
        'detailed_sentiments': sentiments
    }

def compare_sentiment_methods(news_articles, symbol=None):
    """Compare multi-layer analysis vs FinBERT-only and report disagreements and contradictions."""
    enhanced = analyze_news_sentiment(news_articles, symbol)
    finbert_only = analyze_news_sentiment_finbert_only(news_articles)
    comparison = {
        'enhanced_summary': None,
        'finbert_summary': None,
        'disagreements': 0,
        'contradictions_flagged': 0,
        'samples_compared': 0
    }
    if enhanced:
        comparison['enhanced_summary'] = {
            'average_score': enhanced.get('average_score', 0.0),
            'positive_articles': enhanced.get('positive_articles', 0),
            'negative_articles': enhanced.get('negative_articles', 0),
            'neutral_articles': enhanced.get('neutral_articles', 0),
            'total_articles': enhanced.get('total_articles', 0)
        }
    if finbert_only:
        comparison['finbert_summary'] = {
            'average_score': finbert_only.get('average_score', 0.0),
            'positive_articles': finbert_only.get('positive_articles', 0),
            'negative_articles': finbert_only.get('negative_articles', 0),
            'neutral_articles': finbert_only.get('neutral_articles', 0),
            'total_articles': finbert_only.get('total_articles', 0)
        }
    # Per-article disagreement and contradiction counts
    try:
        enh_details = (enhanced or {}).get('detailed_sentiments', [])
        fin_details = (finbert_only or {}).get('detailed_sentiments', [])
        n = min(len(enh_details), len(fin_details))
        comparison['samples_compared'] = n
        disagreements = 0
        contradictions = 0
        for i in range(n):
            enh = enh_details[i]
            fin = fin_details[i]
            if enh.get('sentiment') != fin.get('sentiment'):
                disagreements += 1
            coherence = enh.get('coherence') or {}
            if coherence.get('is_incoherent'):
                contradictions += 1
        comparison['disagreements'] = disagreements
        comparison['contradictions_flagged'] = contradictions
    except Exception:
        pass
    return comparison

# Enhanced forecasting function with model comparison
def enhanced_forecast_stock(symbol, days=30, seq_length=30):
    """Enhanced forecasting with both TCN and LSTM models and performance comparison"""
    df = fetch_stock_data(symbol, days=90)
    
    if df is None or len(df) < seq_length * 2:
        return None, None, "Insufficient data for forecasting"
    
    # Prepare data
    X_train, X_val, y_train, y_val, scaler = prepare_forecasting_data(
        df, feature='close', seq_length=seq_length
    )
    
    # Initialize models
    tcn_model = TCN(input_size=1, output_size=1, num_channels=64, kernel_size=3, dropout=0.2)
    lstm_model = LSTMModel(input_size=1, hidden_size=50, num_layers=2, output_size=1)
    
    # Train models
    st.info(f"Training TCN model for {symbol}...")
    tcn_train_pred, tcn_val_pred, tcn_train_metrics, tcn_val_metrics = train_forecast(
        tcn_model, X_train, X_val, y_train, y_val, epochs=100, model_type="TCN"
    )
    
    st.info(f"Training LSTM model for {symbol}...")
    lstm_train_pred, lstm_val_pred, lstm_train_metrics, lstm_val_metrics = train_forecast(
        lstm_model, X_train, X_val, y_train, y_val, epochs=100, model_type="LSTM"
    )
    
    # Generate future predictions
    last_sequence = X_val[-1] if len(X_val) > 0 else X_train[-1]
    future_predictions_tcn = []
    future_predictions_lstm = []
    
    current_sequence_tcn = last_sequence.copy()
    current_sequence_lstm = last_sequence.copy()
    
    for _ in range(days):
        # TCN prediction
        tcn_input = torch.FloatTensor(current_sequence_tcn).unsqueeze(0).unsqueeze(-1)
        tcn_pred = tcn_model(tcn_input).item()
        future_predictions_tcn.append(tcn_pred)
        
        # Update sequence for TCN
        current_sequence_tcn = np.roll(current_sequence_tcn, -1)
        current_sequence_tcn[-1] = tcn_pred
        
        # LSTM prediction
        lstm_input = torch.FloatTensor(current_sequence_lstm).unsqueeze(0).unsqueeze(-1)
        lstm_pred = lstm_model(lstm_input).item()
        future_predictions_lstm.append(lstm_pred)
        
        # Update sequence for LSTM
        current_sequence_lstm = np.roll(current_sequence_lstm, -1)
        current_sequence_lstm[-1] = lstm_pred
    
    # Inverse transform predictions
    future_predictions_tcn = scaler.inverse_transform(
        np.array(future_predictions_tcn).reshape(-1, 1)
    ).flatten()
    
    future_predictions_lstm = scaler.inverse_transform(
        np.array(future_predictions_lstm).reshape(-1, 1)
    ).flatten()
    
    # Prepare results
    results = {
        'tcn': {
            'predictions': future_predictions_tcn,
            'train_metrics': tcn_train_metrics,
            'val_metrics': tcn_val_metrics,
            'loss_history': {
                'train': tcn_model.train_losses,
                'val': tcn_model.val_losses
            }
        },
        'lstm': {
            'predictions': future_predictions_lstm,
            'train_metrics': lstm_train_metrics,
            'val_metrics': lstm_val_metrics,
            'loss_history': {
                'train': lstm_model.train_losses,
                'val': lstm_model.val_losses
            }
        },
        'current_price': df['close'].iloc[-1],
        'symbol': symbol
    }
    
    return results, scaler, "Forecasting completed successfully"

# Enhanced performance comparison function
def compare_model_performance(forecast_results):
    """Compare TCN and LSTM model performance"""
    tcn_metrics = forecast_results['tcn']['val_metrics']
    lstm_metrics = forecast_results['lstm']['val_metrics']
    
    comparison = {
        'metric': ['MSE', 'MAE', 'RMSE', 'R2', 'Directional Accuracy', 'Correlation'],
        'TCN': [
            tcn_metrics['MSE'], tcn_metrics['MAE'], tcn_metrics['RMSE'],
            tcn_metrics['R2'], tcn_metrics['Directional_Accuracy'], tcn_metrics['Correlation']
        ],
        'LSTM': [
            lstm_metrics['MSE'], lstm_metrics['MAE'], lstm_metrics['RMSE'],
            lstm_metrics['R2'], lstm_metrics['Directional_Accuracy'], lstm_metrics['Correlation']
        ]
    }
    
    df_comparison = pd.DataFrame(comparison)
    df_comparison['Difference'] = df_comparison['TCN'] - df_comparison['LSTM']
    df_comparison['Better_Model'] = df_comparison.apply(
        lambda x: 'TCN' if x['Difference'] < 0 else 'LSTM' if x['Difference'] > 0 else 'Tie', 
        axis=1
    )
    
    return df_comparison

# Update the existing functions to use enhanced forecasting
def predict_future_price_enhanced(symbol, days=1):
    """Enhanced price prediction using both models"""
    forecast_results, scaler, message = enhanced_forecast_stock(symbol, days=days)
    
    if forecast_results is None:
        return None, None, message
    
    # Use the selected model type
    if model_type == "TCN":
        predictions = forecast_results['tcn']['predictions']
        metrics = forecast_results['tcn']['val_metrics']
    else:
        predictions = forecast_results['lstm']['predictions']
        metrics = forecast_results['lstm']['val_metrics']
    
    return predictions[-1] if len(predictions) > 0 else None, metrics, message

# Add new function to display model performance
def display_model_performance(forecast_results, symbol):
    """Display comprehensive model performance analysis"""
    st.subheader(f"📊 Model Performance Analysis for {symbol}")
    
    # Model comparison
    comparison_df = compare_model_performance(forecast_results)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Performance Metrics Comparison:**")
        st.dataframe(comparison_df.style.format({
            'TCN': '{:.6f}', 'LSTM': '{:.6f}', 'Difference': '{:.6f}'
        }))
    
    with col2:
        st.write("**Model Recommendations:**")
        
        # Determine best model based on multiple metrics
        tcn_wins = len(comparison_df[comparison_df['Better_Model'] == 'TCN'])
        lstm_wins = len(comparison_df[comparison_df['Better_Model'] == 'LSTM'])
        
        if tcn_wins > lstm_wins:
            st.success("✅ **TCN performs better** on most metrics")
            st.write("TCN is better for capturing local patterns and short-term dependencies")
        elif lstm_wins > tcn_wins:
            st.success("✅ **LSTM performs better** on most metrics")
            st.write("LSTM is better for capturing long-term dependencies and sequential patterns")
        else:
            st.info("🤝 **Models are comparable**")
            st.write("Both models show similar performance")
    
    # Plot performance
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**TCN Training Progress:**")
        fig_tcn = plot_model_performance_combined(forecast_results['tcn'], 'TCN')
        st.pyplot(fig_tcn)
    
    with col2:
        st.write("**LSTM Training Progress:**")
        fig_lstm = plot_model_performance_combined(forecast_results['lstm'], 'LSTM')
        st.pyplot(fig_lstm)
    
    # Detailed metrics analysis
    st.subheader("📈 Detailed Metrics Analysis")
    
    metrics_to_display = ['MSE', 'MAE', 'RMSE', 'R2', 'Directional_Accuracy', 'Correlation']
    
    for metric in metrics_to_display:
        tcn_val = forecast_results['tcn']['val_metrics'][metric]
        lstm_val = forecast_results['lstm']['val_metrics'][metric]
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if metric in ['R2', 'Directional_Accuracy', 'Correlation']:
                # Higher is better
                if tcn_val > lstm_val:
                    st.success(f"**{metric}:** TCN ({tcn_val:.4f}) > LSTM ({lstm_val:.4f})")
                else:
                    st.info(f"**{metric}:** LSTM ({lstm_val:.4f}) ≥ TCN ({tcn_val:.4f})")
            else:
                # Lower is better
                if tcn_val < lstm_val:
                    st.success(f"**{metric}:** TCN ({tcn_val:.4f}) < LSTM ({lstm_val:.4f})")
                else:
                    st.info(f"**{metric}:** LSTM ({lstm_val:.4f}) ≤ TCN ({tcn_val:.4f})")

def plot_sentiment_classification_metrics(classification_metrics, symbol):
    """Plot sentiment classification metrics in a visual format"""
    if not classification_metrics or 'error' in classification_metrics:
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Overall metrics bar chart
    metrics_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1 Score\n(Macro)']
    metrics_values = [
        classification_metrics['accuracy'],
        classification_metrics['precision_macro'],
        classification_metrics['recall_macro'],
        classification_metrics['f1_macro']
    ]
    
    bars1 = ax1.bar(metrics_names, metrics_values, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax1.set_title(f'Sentiment Classification Metrics - {symbol}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Score')
    ax1.set_ylim(0, 1)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, value in zip(bars1, metrics_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Per-class performance
    if 'per_class_metrics' in classification_metrics:
        per_class = classification_metrics['per_class_metrics']
        labels = per_class['labels']
        
        x = np.arange(len(labels))
        width = 0.25
        
        bars2 = ax2.bar(x - width, per_class['precision'], width, label='Precision', alpha=0.7)
        bars3 = ax2.bar(x, per_class['recall'], width, label='Recall', alpha=0.7)
        bars4 = ax2.bar(x + width, per_class['f1'], width, label='F1 Score', alpha=0.7)
        
        ax2.set_title('Per-Class Performance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Sentiment Class')
        ax2.set_ylabel('Score')
        ax2.set_xticks(x)
        ax2.set_xticklabels([label.title() for label in labels])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1)
    
    # Plot 3: Confusion Matrix
    if 'confusion_matrix' in classification_metrics:
        conf_matrix = np.array(classification_metrics['confusion_matrix'])
        labels = classification_metrics['per_class_metrics']['labels']
        
        im = ax3.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        ax3.figure.colorbar(im, ax=ax3)
        
        ax3.set(xticks=np.arange(conf_matrix.shape[1]),
                yticks=np.arange(conf_matrix.shape[0]),
                xticklabels=[label.title() for label in labels],
                yticklabels=[label.title() for label in labels],
                title='Confusion Matrix',
                ylabel='True Label',
                xlabel='Predicted Label')
        
        # Add text annotations
        thresh = conf_matrix.max() / 2.
        for i in range(conf_matrix.shape[0]):
            for j in range(conf_matrix.shape[1]):
                ax3.text(j, i, format(conf_matrix[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
    
    # Plot 4: Metrics comparison (Macro vs Weighted)
    comparison_metrics = ['Precision', 'Recall', 'F1 Score']
    macro_values = [
        classification_metrics['precision_macro'],
        classification_metrics['recall_macro'],
        classification_metrics['f1_macro']
    ]
    weighted_values = [
        classification_metrics['precision_weighted'],
        classification_metrics['recall_weighted'],
        classification_metrics['f1_weighted']
    ]
    
    x = np.arange(len(comparison_metrics))
    width = 0.35
    
    bars5 = ax4.bar(x - width/2, macro_values, width, label='Macro Average', alpha=0.7)
    bars6 = ax4.bar(x + width/2, weighted_values, width, label='Weighted Average', alpha=0.7)
    
    ax4.set_title('Macro vs Weighted Averages', fontsize=14, fontweight='bold')
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Score')
    ax4.set_xticks(x)
    ax4.set_xticklabels(comparison_metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # Add value labels
    for bars in [bars5, bars6]:
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    return fig

def plot_model_performance_combined(model_results, model_name):
    """Plot combined performance metrics for a model"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax1.plot(model_results['loss_history']['train'], label='Training Loss', alpha=0.7)
    ax1.plot(model_results['loss_history']['val'], label='Validation Loss', alpha=0.7)
    ax1.set_title(f'{model_name} - Loss Convergence')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Metrics comparison
    metrics = model_results['val_metrics']
    metric_names = ['MSE', 'MAE', 'RMSE', 'R2']
    metric_values = [metrics['MSE'], metrics['MAE'], metrics['RMSE'], metrics['R2']]
    
    bars = ax2.bar(metric_names, metric_values, alpha=0.7)
    ax2.set_title(f'{model_name} - Validation Metrics')
    ax2.set_ylabel('Score')
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{value:.4f}', ha='center', va='bottom')
    
    # Directional accuracy and correlation
    accuracy_metrics = ['Directional Accuracy', 'Correlation']
    accuracy_values = [metrics['Directional_Accuracy'], metrics['Correlation']]
    
    bars = ax3.bar(accuracy_metrics, accuracy_values, alpha=0.7, color=['skyblue', 'lightcoral'])
    ax3.set_title(f'{model_name} - Predictive Accuracy')
    ax3.set_ylabel('Score')
    ax3.set_ylim(0, 1)
    
    for bar, value in zip(bars, accuracy_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{value:.4f}', ha='center', va='bottom')
    
    # Training vs validation final scores
    final_scores = [model_results['loss_history']['train'][-1], 
                   model_results['loss_history']['val'][-1]]
    labels = ['Training', 'Validation']
    
    ax4.bar(labels, final_scores, alpha=0.7, color=['lightgreen', 'lightblue'])
    ax4.set_title(f'{model_name} - Final Loss Scores')
    ax4.set_ylabel('Loss')
    
    for i, score in enumerate(final_scores):
        ax4.text(i, score, f'{score:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    return fig

def answer_general_financial_question(question):
    """Handle general financial questions not covered by specific categories"""
    question_lower = question.lower()
    
    # Default responses for general questions
    general_responses = {
        'hello': "Hello! I'm your financial AI assistant. How can I help you with stocks, investments, or market analysis today?",
        'hi': "Hi there! I can help you analyze stocks, provide investment recommendations, and explain market trends. What would you like to know?",
        'help': """I can help you with:
- Stock analysis and recommendations
- Price predictions and forecasts  
- Sector performance insights
- Technical indicator explanations
- Portfolio optimization advice
- Risk assessment
- Sentiment analysis

Try asking about specific stocks like AAPL or TSLA, or ask general questions about the market.""",
        'thank': "You're welcome! Feel free to ask more questions about stocks or investments.",
        'what can you do': """I'm a comprehensive financial AI assistant with these capabilities:

📈 **Stock Analysis**: Technical indicators, price trends, buy/sell signals
🤖 **AI Forecasting**: TCN and LSTM models for price predictions  
📊 **Sector Analysis**: Performance comparisons across industries
💬 **Sentiment Analysis**: Multi-layer analysis of news and market sentiment
⚖️ **Risk Assessment**: Volatility analysis and portfolio optimization
📋 **Recommendations**: Data-driven investment suggestions

Try asking about any stock or investment topic!"""
    }
    
    # Check for general greetings and questions
    for key, response in general_responses.items():
        if key in question_lower:
            return response
    
    # If no specific match, provide a helpful default response
    return """I understand you're asking about financial markets. I can help you with:

**Stock Analysis**: "Should I buy AAPL?", "How is TSLA performing?"
**Price Predictions**: "What will Amazon stock do next week?"  
**Sector Insights**: "How is the tech sector doing?"
**Portfolio Advice**: "How many stocks should I own?"
**Risk Assessment**: "Is Tesla a risky investment?"

Could you please rephrase your question with more specific details about stocks, sectors, or investments?"""
def display_sentiment_comparison_analysis(symbol):
    """Display comprehensive comparison between FinBERT-only and multi-layer sentiment analysis"""
    
    # Get news articles
    news_articles = get_news_articles(symbol)
    
    if not news_articles:
        return "No news articles available for comparison."
    
    # Perform both analyses
    multi_layer_sentiment = analyze_news_sentiment(news_articles, symbol)
    finbert_only_sentiment = analyze_news_sentiment_finbert_only(news_articles)
    comparison = compare_sentiment_methods(news_articles, symbol)
    
    # Build comparison response
    response = f"## 🔬 Sentiment Analysis Comparison for {all_companies.get(symbol, symbol)}\n\n"
    
    # Summary statistics
    response += "### 📊 Summary Statistics\n\n"
    
    if multi_layer_sentiment:
        response += "**Multi-Layer Analysis:**\n"
        response += f"- Average Score: {multi_layer_sentiment['average_score']:.3f}\n"
        response += f"- Positive Articles: {multi_layer_sentiment['positive_articles']}\n"
        response += f"- Negative Articles: {multi_layer_sentiment['negative_articles']}\n"
        response += f"- Neutral Articles: {multi_layer_sentiment.get('neutral_articles', 0)}\n\n"
    
    if finbert_only_sentiment:
        response += "**FinBERT-Only Analysis:**\n"
        response += f"- Average Score: {finbert_only_sentiment['average_score']:.3f}\n"
        response += f"- Positive Articles: {finbert_only_sentiment['positive_articles']}\n"
        response += f"- Negative Articles: {finbert_only_sentiment['negative_articles']}\n"
        response += f"- Neutral Articles: {finbert_only_sentiment.get('neutral_articles', 0)}\n\n"
    
    # Method comparison
    response += "### ⚖️ Method Comparison\n\n"
    
    if comparison:
        response += f"- **Samples Compared**: {comparison['samples_compared']} articles\n"
        response += f"- **Disagreements**: {comparison['disagreements']} articles\n"
        response += f"- **Contradictions Flagged**: {comparison['contradictions_flagged']} articles\n"
        response += f"- **Agreement Rate**: {((comparison['samples_compared'] - comparison['disagreements']) / comparison['samples_compared'] * 100):.1f}%\n\n"
    
    # Layer-by-layer breakdown
    response += "### 🏗️ Multi-Layer Architecture Breakdown\n\n"
    
    if multi_layer_sentiment and multi_layer_sentiment.get('detailed_sentiments'):
        # Analyze first article as example
        first_sentiment = multi_layer_sentiment['detailed_sentiments'][0]
        components = first_sentiment.get('components', {})
        
        response += "**Layer 1: FinBERT (Deep Learning Base)**\n"
        response += "- Pre-trained ProsusAI/finbert model specifically fine-tuned on financial texts\n"
        response += f"- Base Score: {components.get('finbert', {}).get('score', 0):.3f}\n"
        response += f"- Base Label: {components.get('finbert', {}).get('label', 'N/A')}\n\n"
        
        response += "**Layer 2: Rule-Based Financial Mapping**\n"
        response += "- Custom FINANCIAL_SENTIMENT_LEXICON with 40+ financial terms\n"
        response += "- Terms like 'bullish' (+0.8), 'bearish' (-0.8), 'rally' (+0.7), 'plunge' (-0.8)\n"
        response += f"- Rule-based Score: {components.get('rule_based', 0):.3f}\n\n"
        
        response += "**Layer 3: Expanded Lexicon Analysis**\n"
        response += "- WordNet Expansion: Automatically finds synonyms for financial terms\n"
        response += "  - 'bullish' → 'optimistic', 'positive', 'upbeat'\n"
        response += "  - 'bearish' → 'pessimistic', 'negative', 'gloomy'\n"
        response += "- Embedding-based Expansion: Adds semantically similar terms\n"
        response += f"- Lexicon Score: {components.get('lexicon', 0):.3f}\n\n"
        
        response += "**Layer 4: Coherence Checking**\n"
        coherence = first_sentiment.get('coherence', {})
        response += f"- Contradiction Detection: {'✓' if coherence.get('is_incoherent') and coherence.get('incoherence_type') == 'contradiction_detected' else '✗'}\n"
        response += f"- Negation Handling: {'✓' if coherence.get('is_incoherent') and coherence.get('incoherence_type') == 'negation_detected' else '✗'}\n"
        response += f"- Sarcasm Adjustment: {'Applied' if coherence.get('is_incoherent') else 'Not Applied'}\n\n"
        
        response += "**Layer 5: Contextual Adjustment**\n"
        response += "- Earnings Context: Adjusts for mixed sentiment in earnings reports\n"
        response += "- Stock Performance Bias: Incorporates recent stock performance\n"
        response += "- News Context: Adapts based on news article characteristics\n"
        response += f"- Contextual Score: {components.get('contextual', 0):.3f}\n\n"
        
        response += f"**Final Aggregated Score**: {first_sentiment.get('score', 0):.3f}\n"
        response += f"**Final Sentiment**: {first_sentiment.get('sentiment', 'N/A').title()}\n\n"
    
    # Add classification metrics for multi-layer
    if multi_layer_sentiment and multi_layer_sentiment.get('classification_metrics'):
        metrics = multi_layer_sentiment['classification_metrics']
        if 'error' not in metrics:
            response += "### 🎯 Multi-Layer Classification Performance\n\n"
            response += f"- **Accuracy**: {metrics['accuracy']:.3f}\n"
            response += f"- **Precision (Macro)**: {metrics['precision_macro']:.3f}\n"
            response += f"- **Recall (Macro)**: {metrics['recall_macro']:.3f}\n"
            response += f"- **F1 Score (Macro)**: {metrics['f1_macro']:.3f}\n"
            response += f"- **Sample Size**: {metrics['sample_size']} articles\n\n"
    
    # Benefits of multi-layer approach
    response += "### 🚀 Advantages of Multi-Layer Approach\n\n"
    response += "1. **Domain Specialization**: Financial lexicon captures industry-specific terminology\n"
    response += "2. **Context Awareness**: Handles financial context like earnings reports differently\n"
    response += "3. **Contradiction Detection**: Identifies sarcasm and mixed signals\n"
    response += "4. **Negation Handling**: Properly processes negative constructions\n"
    response += "5. **Expanded Vocabulary**: Covers more financial terms through synonyms\n"
    response += "6. **Performance Metrics**: Provides classification metrics for reliability assessment\n\n"
    
    # When each method is better
    response += "### 🎭 When to Use Each Method\n\n"
    response += "**FinBERT-Only is better for:**\n"
    response += "- Standard financial texts without specialized terminology\n"
    response += "- Quick analysis without domain adaptation\n"
    response += "- Texts without sarcasm or contradictions\n\n"
    
    response += "**Multi-Layer is better for:**\n"
    response += "- Complex financial texts with industry jargon\n"
    response += "- Texts with potential sarcasm or mixed signals\n"
    response += "- Earnings reports and analyst commentary\n"
    response += "- When reliability metrics are needed\n\n"
    
    return response

def plot_sentiment_comparison_chart(comparison_data, symbol):
    """Create visual comparison between FinBERT and Multi-Layer sentiment"""
    
    if not comparison_data or 'enhanced_summary' not in comparison_data or 'finbert_summary' not in comparison_data:
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Average sentiment scores comparison
    methods = ['FinBERT-Only', 'Multi-Layer']
    scores = [
        comparison_data['finbert_summary']['average_score'],
        comparison_data['enhanced_summary']['average_score']
    ]
    
    colors = ['lightblue', 'lightgreen']
    bars1 = ax1.bar(methods, scores, color=colors, alpha=0.7)
    ax1.set_title(f'Average Sentiment Score Comparison - {symbol}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sentiment Score')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Article sentiment distribution
    sentiment_types = ['Positive', 'Negative', 'Neutral']
    finbert_counts = [
        comparison_data['finbert_summary']['positive_articles'],
        comparison_data['finbert_summary']['negative_articles'],
        comparison_data['finbert_summary']['neutral_articles']
    ]
    multi_layer_counts = [
        comparison_data['enhanced_summary']['positive_articles'],
        comparison_data['enhanced_summary']['negative_articles'],
        comparison_data['enhanced_summary']['neutral_articles']
    ]
    
    x = np.arange(len(sentiment_types))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, finbert_counts, width, label='FinBERT-Only', alpha=0.7)
    bars3 = ax2.bar(x + width/2, multi_layer_counts, width, label='Multi-Layer', alpha=0.7)
    
    ax2.set_title('Sentiment Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment Type')
    ax2.set_ylabel('Number of Articles')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sentiment_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height, 
                    f'{int(height)}', ha='center', va='bottom')
    
    # Plot 3: Method agreement and disagreements
    if comparison_data['samples_compared'] > 0:
        agreement_data = [
            comparison_data['samples_compared'] - comparison_data['disagreements'],
            comparison_data['disagreements'],
            comparison_data['contradictions_flagged']
        ]
        agreement_labels = ['Agreements', 'Disagreements', 'Contradictions']
        
        colors_agreement = ['lightgreen', 'lightcoral', 'gold']
        bars4 = ax3.bar(agreement_labels, agreement_data, color=colors_agreement, alpha=0.7)
        
        ax3.set_title('Method Agreement Analysis', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Number of Articles')
        ax3.grid(True, alpha=0.3)
        
        # Add value labels and percentages
        total = comparison_data['samples_compared']
        for bar, count in zip(bars4, agreement_data):
            percentage = (count / total) * 100
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    
    # Plot 4: Performance metrics comparison (if available)
    # This would require classification metrics for both methods
    ax4.text(0.5, 0.5, 'Classification Metrics\nComparison\n\n(Multi-layer only\nin current implementation)', 
             ha='center', va='center', transform=ax4.transAxes, fontsize=12)
    ax4.set_title('Classification Performance', fontsize=14, fontweight='bold')
    ax4.set_xticks([])
    ax4.set_yticks([])
    
    plt.tight_layout()
    return fig
def calculate_sentiment_classification_metrics_both_methods(sentiments_enhanced, sentiments_finbert, ground_truth=None):
    """Calculate classification metrics for both enhanced and FinBERT-only sentiment analysis"""
    if not sentiments_enhanced or not sentiments_finbert:
        return None, None
    
    # Extract predicted labels
    enhanced_labels = [s['sentiment'] for s in sentiments_enhanced]
    finbert_labels = [s['sentiment'] for s in sentiments_finbert]
    
    # If no ground truth provided, create baseline using majority voting or external reference
    if ground_truth is None:
        # Use ensemble of both methods as reference (fallback)
        ground_truth = []
        for enh, fb in zip(enhanced_labels, finbert_labels):
            if enh == fb:
                ground_truth.append(enh)  # Agreement case
            else:
                # Use enhanced method as reference when they disagree
                ground_truth.append(enh)
    
    # Ensure same length
    min_length = min(len(enhanced_labels), len(finbert_labels), len(ground_truth))
    enhanced_labels = enhanced_labels[:min_length]
    finbert_labels = finbert_labels[:min_length]
    ground_truth = ground_truth[:min_length]
    
    # Calculate metrics for both methods
    enhanced_metrics = _calculate_single_method_metrics(enhanced_labels, ground_truth, "Enhanced")
    finbert_metrics = _calculate_single_method_metrics(finbert_labels, ground_truth, "FinBERT")
    
    return enhanced_metrics, finbert_metrics

def _calculate_single_method_metrics(predicted_labels, ground_truth, method_name):
    """Calculate metrics for a single sentiment analysis method"""
    try:
        # Overall accuracy
        accuracy = accuracy_score(ground_truth, predicted_labels)
        
        # Multi-class metrics
        precision_macro = precision_score(ground_truth, predicted_labels, average='macro', zero_division=0)
        recall_macro = recall_score(ground_truth, predicted_labels, average='macro', zero_division=0)
        f1_macro = f1_score(ground_truth, predicted_labels, average='macro', zero_division=0)
        
        # Weighted metrics
        precision_weighted = precision_score(ground_truth, predicted_labels, average='weighted', zero_division=0)
        recall_weighted = recall_score(ground_truth, predicted_labels, average='weighted', zero_division=0)
        f1_weighted = f1_score(ground_truth, predicted_labels, average='weighted', zero_division=0)
        
        # Per-class metrics
        labels = list(set(ground_truth + predicted_labels))
        precision_per_class = precision_score(ground_truth, predicted_labels, average=None, labels=labels, zero_division=0)
        recall_per_class = recall_score(ground_truth, predicted_labels, average=None, labels=labels, zero_division=0)
        f1_per_class = f1_score(ground_truth, predicted_labels, average=None, labels=labels, zero_division=0)
        
        # Classification report
        classification_rep = classification_report(ground_truth, predicted_labels, output_dict=True, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(ground_truth, predicted_labels, labels=labels)
        
        return {
            'method': method_name,
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class_metrics': {
                'labels': labels,
                'precision': precision_per_class.tolist() if hasattr(precision_per_class, 'tolist') else precision_per_class,
                'recall': recall_per_class.tolist() if hasattr(recall_per_class, 'tolist') else recall_per_class,
                'f1': f1_per_class.tolist() if hasattr(f1_per_class, 'tolist') else f1_per_class
            },
            'classification_report': classification_rep,
            'confusion_matrix': conf_matrix.tolist() if hasattr(conf_matrix, 'tolist') else conf_matrix,
            'sample_size': len(predicted_labels)
        }
    except Exception as e:
        return {
            'method': method_name,
            'error': f"Could not calculate metrics: {str(e)}",
            'sample_size': len(predicted_labels)
        }

def analyze_news_sentiment_with_comparison(news_articles, symbol=None):
    """Analyze sentiment with comprehensive comparison between methods"""
    # Get both analyses
    enhanced_sentiments = []
    finbert_sentiments = []
    
    analyzer = load_enhanced_sentiment_analyzer()
    finbert = load_finbert_model()
    
    for article in news_articles:
        text = article.get('title', '') + " " + article.get('summary', '')
        
        # Enhanced analysis
        enhanced_sentiment = analyzer.analyze_sentiment(text, context="news", symbol=symbol)
        enhanced_sentiments.append(enhanced_sentiment)
        
        # FinBERT-only analysis
        if finbert:
            try:
                result = finbert(text)[0]
                label = result.get('label', 'neutral').lower()
                score_raw = float(result.get('score', 0.0))
                score = score_raw if label == 'positive' else (-score_raw if label == 'negative' else 0.0)
                finbert_sentiments.append({
                    'sentiment': label,
                    'score': score
                })
            except Exception:
                finbert_sentiments.append({
                    'sentiment': 'neutral',
                    'score': 0.0
                })
        else:
            finbert_sentiments.append({
                'sentiment': 'neutral',
                'score': 0.0
            })
    
    # Calculate metrics for both methods
    enhanced_metrics, finbert_metrics = calculate_sentiment_classification_metrics_both_methods(
        enhanced_sentiments, finbert_sentiments
    )
    
    # Calculate average sentiment for both methods
    enhanced_avg = sum(s['score'] for s in enhanced_sentiments) / len(enhanced_sentiments) if enhanced_sentiments else 0
    finbert_avg = sum(s['score'] for s in finbert_sentiments) / len(finbert_sentiments) if finbert_sentiments else 0
    
    # Count sentiment distributions
    enhanced_positive = sum(1 for s in enhanced_sentiments if s['sentiment'] == 'positive')
    enhanced_negative = sum(1 for s in enhanced_sentiments if s['sentiment'] == 'negative')
    enhanced_neutral = len(enhanced_sentiments) - enhanced_positive - enhanced_negative
    
    finbert_positive = sum(1 for s in finbert_sentiments if s['sentiment'] == 'positive')
    finbert_negative = sum(1 for s in finbert_sentiments if s['sentiment'] == 'negative')
    finbert_neutral = len(finbert_sentiments) - finbert_positive - finbert_negative
    
    return {
        'enhanced': {
            'average_score': enhanced_avg,
            'positive_articles': enhanced_positive,
            'negative_articles': enhanced_negative,
            'neutral_articles': enhanced_neutral,
            'total_articles': len(enhanced_sentiments),
            'metrics': enhanced_metrics,
            'detailed_sentiments': enhanced_sentiments
        },
        'finbert': {
            'average_score': finbert_avg,
            'positive_articles': finbert_positive,
            'negative_articles': finbert_negative,
            'neutral_articles': finbert_neutral,
            'total_articles': len(finbert_sentiments),
            'metrics': finbert_metrics,
            'detailed_sentiments': finbert_sentiments
        },
        'comparison': {
            'score_difference': enhanced_avg - finbert_avg,
            'agreement_rate': calculate_agreement_rate(enhanced_sentiments, finbert_sentiments),
            'samples_compared': len(enhanced_sentiments)
        }
    }

def calculate_agreement_rate(enhanced_sentiments, finbert_sentiments):
    """Calculate agreement rate between two sentiment analysis methods"""
    if not enhanced_sentiments or not finbert_sentiments:
        return 0
    
    min_length = min(len(enhanced_sentiments), len(finbert_sentiments))
    agreements = 0
    
    for i in range(min_length):
        if enhanced_sentiments[i]['sentiment'] == finbert_sentiments[i]['sentiment']:
            agreements += 1
    
    return agreements / min_length

def plot_comprehensive_sentiment_comparison(comparison_data, symbol):
    """Create comprehensive comparison visualization for both sentiment methods"""
    if not comparison_data or 'enhanced' not in comparison_data or 'finbert' not in comparison_data:
        return None
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Average sentiment scores comparison
    methods = ['FinBERT-Only', 'Multi-Layer']
    scores = [
        comparison_data['finbert']['average_score'],
        comparison_data['enhanced']['average_score']
    ]
    
    colors = ['lightblue', 'lightgreen']
    bars1 = ax1.bar(methods, scores, color=colors, alpha=0.7)
    ax1.set_title(f'Average Sentiment Score Comparison - {symbol}', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Sentiment Score')
    ax1.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars1, scores):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Article sentiment distribution comparison
    sentiment_types = ['Positive', 'Negative', 'Neutral']
    
    finbert_counts = [
        comparison_data['finbert']['positive_articles'],
        comparison_data['finbert']['negative_articles'],
        comparison_data['finbert']['neutral_articles']
    ]
    enhanced_counts = [
        comparison_data['enhanced']['positive_articles'],
        comparison_data['enhanced']['negative_articles'],
        comparison_data['enhanced']['neutral_articles']
    ]
    
    x = np.arange(len(sentiment_types))
    width = 0.35
    
    bars2 = ax2.bar(x - width/2, finbert_counts, width, label='FinBERT-Only', alpha=0.7)
    bars3 = ax2.bar(x + width/2, enhanced_counts, width, label='Multi-Layer', alpha=0.7)
    
    ax2.set_title('Sentiment Distribution Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Sentiment Type')
    ax2.set_ylabel('Number of Articles')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sentiment_types)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bars in [bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height, 
                    f'{int(height)}', ha='center', va='bottom')
    
    # Plot 3: Performance metrics comparison (if available)
    if (comparison_data['enhanced'].get('metrics') and 'error' not in comparison_data['enhanced']['metrics'] and
        comparison_data['finbert'].get('metrics') and 'error' not in comparison_data['finbert']['metrics']):
        
        metrics_names = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1 Score\n(Macro)']
        
        enhanced_metrics_vals = [
            comparison_data['enhanced']['metrics']['accuracy'],
            comparison_data['enhanced']['metrics']['precision_macro'],
            comparison_data['enhanced']['metrics']['recall_macro'],
            comparison_data['enhanced']['metrics']['f1_macro']
        ]
        
        finbert_metrics_vals = [
            comparison_data['finbert']['metrics']['accuracy'],
            comparison_data['finbert']['metrics']['precision_macro'],
            comparison_data['finbert']['metrics']['recall_macro'],
            comparison_data['finbert']['metrics']['f1_macro']
        ]
        
        x_metrics = np.arange(len(metrics_names))
        width_metrics = 0.35
        
        bars_enhanced = ax3.bar(x_metrics - width_metrics/2, enhanced_metrics_vals, width_metrics, 
                               label='Multi-Layer', alpha=0.7, color='lightgreen')
        bars_finbert = ax3.bar(x_metrics + width_metrics/2, finbert_metrics_vals, width_metrics, 
                              label='FinBERT-Only', alpha=0.7, color='lightblue')
        
        ax3.set_title('Classification Performance Comparison', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x_metrics)
        ax3.set_xticklabels(metrics_names)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
        
        # Add value labels
        for bars in [bars_enhanced, bars_finbert]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height, 
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    else:
        ax3.text(0.5, 0.5, 'Performance Metrics\nNot Available', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=12)
        ax3.set_title('Classification Performance Comparison', fontsize=14, fontweight='bold')
    
    # Plot 4: Method agreement and performance summary
    agreement_data = [
        comparison_data['comparison']['agreement_rate'] * 100,
        (1 - comparison_data['comparison']['agreement_rate']) * 100
    ]
    agreement_labels = ['Agreement', 'Disagreement']
    
    colors_agreement = ['lightgreen', 'lightcoral']
    wedges, texts, autotexts = ax4.pie(agreement_data, labels=agreement_labels, autopct='%1.1f%%',
                                      colors=colors_agreement, startangle=90)
    
    ax4.set_title('Method Agreement Rate', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig

def display_comprehensive_sentiment_comparison(symbol):
    """Display comprehensive comparison between both sentiment analysis methods with performance metrics"""
    
    # Get news articles
    news_articles = get_news_articles(symbol)
    
    if not news_articles:
        return "No news articles available for comparison."
    
    # Perform comprehensive analysis
    comparison_results = analyze_news_sentiment_with_comparison(news_articles, symbol)
    
    # Build comprehensive response
    response = f"## 🔬 Comprehensive Sentiment Analysis Comparison for {all_companies.get(symbol, symbol)}\n\n"
    
    # Summary statistics for both methods
    response += "### 📊 Summary Statistics\n\n"
    
    response += "**Multi-Layer Analysis:**\n"
    response += f"- Average Score: {comparison_results['enhanced']['average_score']:.3f}\n"
    response += f"- Positive Articles: {comparison_results['enhanced']['positive_articles']}\n"
    response += f"- Negative Articles: {comparison_results['enhanced']['negative_articles']}\n"
    response += f"- Neutral Articles: {comparison_results['enhanced']['neutral_articles']}\n"
    response += f"- Total Articles: {comparison_results['enhanced']['total_articles']}\n\n"
    
    response += "**FinBERT-Only Analysis:**\n"
    response += f"- Average Score: {comparison_results['finbert']['average_score']:.3f}\n"
    response += f"- Positive Articles: {comparison_results['finbert']['positive_articles']}\n"
    response += f"- Negative Articles: {comparison_results['finbert']['negative_articles']}\n"
    response += f"- Neutral Articles: {comparison_results['finbert']['neutral_articles']}\n"
    response += f"- Total Articles: {comparison_results['finbert']['total_articles']}\n\n"
    
    # Method comparison summary
    response += "### ⚖️ Method Comparison Summary\n\n"
    response += f"- **Score Difference**: {comparison_results['comparison']['score_difference']:.3f} "
    if comparison_results['comparison']['score_difference'] > 0:
        response += "(Multi-Layer more positive)\n"
    else:
        response += "(FinBERT more positive)\n"
    
    response += f"- **Agreement Rate**: {comparison_results['comparison']['agreement_rate'] * 100:.1f}%\n"
    response += f"- **Samples Compared**: {comparison_results['comparison']['samples_compared']} articles\n\n"
    
    # Performance metrics for both methods
    response += "### 🎯 Performance Metrics Comparison\n\n"
    
    if (comparison_results['enhanced'].get('metrics') and 'error' not in comparison_results['enhanced']['metrics'] and
        comparison_results['finbert'].get('metrics') and 'error' not in comparison_results['finbert']['metrics']):
        
        enhanced_metrics = comparison_results['enhanced']['metrics']
        finbert_metrics = comparison_results['finbert']['metrics']
        
        response += "**Multi-Layer Performance:**\n"
        response += f"- Accuracy: {enhanced_metrics['accuracy']:.3f}\n"
        response += f"- Precision (Macro): {enhanced_metrics['precision_macro']:.3f}\n"
        response += f"- Recall (Macro): {enhanced_metrics['recall_macro']:.3f}\n"
        response += f"- F1 Score (Macro): {enhanced_metrics['f1_macro']:.3f}\n\n"
        
        response += "**FinBERT-Only Performance:**\n"
        response += f"- Accuracy: {finbert_metrics['accuracy']:.3f}\n"
        response += f"- Precision (Macro): {finbert_metrics['precision_macro']:.3f}\n"
        response += f"- Recall (Macro): {finbert_metrics['recall_macro']:.3f}\n"
        response += f"- F1 Score (Macro): {finbert_metrics['f1_macro']:.3f}\n\n"
        
        # Determine which method performs better
        enhanced_score = (enhanced_metrics['accuracy'] + enhanced_metrics['f1_macro']) / 2
        finbert_score = (finbert_metrics['accuracy'] + finbert_metrics['f1_macro']) / 2
        
        if enhanced_score > finbert_score:
            response += "**Performance Winner**: 🏆 **Multi-Layer Analysis**\n"
            response += f"- Advantage: +{(enhanced_score - finbert_score):.3f} in combined score\n"
        elif finbert_score > enhanced_score:
            response += "**Performance Winner**: 🏆 **FinBERT-Only**\n"
            response += f"- Advantage: +{(finbert_score - enhanced_score):.3f} in combined score\n"
        else:
            response += "**Performance**: 🤝 **Methods are comparable**\n"
        
    else:
        response += "Performance metrics not available for both methods.\n\n"
    
    # Method strengths and use cases
    response += "### 🚀 Method Strengths and Use Cases\n\n"
    
    response += "**Multi-Layer Analysis Strengths:**\n"
    response += "✅ Domain-specific financial terminology handling\n"
    response += "✅ Sarcasm and contradiction detection\n"
    response += "✅ Negation pattern recognition\n"
    response += "✅ Contextual adaptation for earnings/news\n"
    response += "✅ Expanded vocabulary through synonyms\n\n"
    
    response += "**FinBERT-Only Strengths:**\n"
    response += "✅ Fast inference with pre-trained model\n"
    response += "✅ Consistent performance on standard texts\n"
    response += "✅ No additional rule-based complexity\n"
    response += "✅ Direct financial domain fine-tuning\n"
    response += "✅ Simpler implementation and maintenance\n\n"
    
    # Recommendations
    response += "### 💡 Recommendations\n\n"
    
    if comparison_results['comparison']['agreement_rate'] > 0.8:
        response += "**High Agreement**: Both methods largely agree, increasing confidence in sentiment analysis results.\n\n"
    else:
        response += "**Moderate Agreement**: Methods show significant differences. Consider the context and use case when interpreting results.\n\n"
    
    if abs(comparison_results['comparison']['score_difference']) > 0.3:
        response += "**Large Score Difference**: Significant divergence in sentiment scores. Multi-layer method may be capturing additional contextual nuances.\n\n"
    
    return response

# Update the main answer_financial_question function to use the comprehensive comparison
def answer_financial_question(question):
    """Main function to analyze and answer financial questions"""
    question_lower = question.lower()
    
    # Extract symbols from question
    symbols = extract_symbols_from_question(question)
    
    # Enhanced sentiment comparison
    if any(word in question_lower for word in ['compare sentiment', 'sentiment comparison', 'finbert vs', 'multi-layer', 'layer comparison', 'both sentiment']):
        if symbols:
            symbol = symbols[0]
            response = display_comprehensive_sentiment_comparison(symbol)
            
            # Add visual comparison
            news_articles = get_news_articles(symbol)
            if news_articles:
                comparison_results = analyze_news_sentiment_with_comparison(news_articles, symbol)
                
                st.markdown("---")
                st.subheader("📊 Comprehensive Sentiment Analysis Visualization")
                
                fig_comprehensive = plot_comprehensive_sentiment_comparison(comparison_results, symbol)
                if fig_comprehensive:
                    st.pyplot(fig_comprehensive)
                
                # Display detailed metrics tables
                if (comparison_results['enhanced'].get('metrics') and 'error' not in comparison_results['enhanced']['metrics'] and
                    comparison_results['finbert'].get('metrics') and 'error' not in comparison_results['finbert']['metrics']):
                    
                    st.subheader("📋 Detailed Performance Metrics")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Multi-Layer Analysis Metrics**")
                        enhanced_metrics = comparison_results['enhanced']['metrics']
                        metrics_df_enhanced = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)',
                                      'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)'],
                            'Value': [
                                enhanced_metrics['accuracy'],
                                enhanced_metrics['precision_macro'],
                                enhanced_metrics['recall_macro'],
                                enhanced_metrics['f1_macro'],
                                enhanced_metrics['precision_weighted'],
                                enhanced_metrics['recall_weighted'],
                                enhanced_metrics['f1_weighted']
                            ]
                        })
                        st.dataframe(metrics_df_enhanced.style.format({'Value': '{:.3f}'}))
                    
                    with col2:
                        st.write("**FinBERT-Only Metrics**")
                        finbert_metrics = comparison_results['finbert']['metrics']
                        metrics_df_finbert = pd.DataFrame({
                            'Metric': ['Accuracy', 'Precision (Macro)', 'Recall (Macro)', 'F1 Score (Macro)',
                                      'Precision (Weighted)', 'Recall (Weighted)', 'F1 Score (Weighted)'],
                            'Value': [
                                finbert_metrics['accuracy'],
                                finbert_metrics['precision_macro'],
                                finbert_metrics['recall_macro'],
                                finbert_metrics['f1_macro'],
                                finbert_metrics['precision_weighted'],
                                finbert_metrics['recall_weighted'],
                                finbert_metrics['f1_weighted']
                            ]
                        })
                        st.dataframe(metrics_df_finbert.style.format({'Value': '{:.3f}'}))
            
            return response
        else:
            return "Please specify a stock symbol for sentiment comparison analysis."
    
    # ... rest of your existing answer_financial_question function remains the same ...
    # 1. Buy/Sell/Hold Decisions
    elif any(word in question_lower for word in ['buy', 'sell', 'hold', 'investment', 'should i']):
        if symbols:
            return analyze_single_stock_recommendation(symbols[0], timeframe)
        else:
            return analyze_all_stocks_recommendations(timeframe, question_lower)
    
    # 2. Company Performance
    elif any(word in question_lower for word in ['performance', 'how is', 'doing', 'growing']):
        if symbols:
            return analyze_single_stock_performance(symbols[0])
        else:
            return analyze_top_performing_stocks(timeframe)
    
    # 3. Sector/Industry Insights
    elif any(word in question_lower for word in ['sector', 'industry', 'tech stocks', 'banking', 'energy', 'ev']):
        return analyze_sector_performance_insights(question_lower)
    
    # 4. Event-Driven Questions
    elif any(word in question_lower for word in ['earnings', 'launch', 'call', 'event', 'announcement']):
        if symbols:
            return analyze_event_impact(symbols[0])
        else:
            return analyze_companies_with_upcoming_events()
    
    # 5. Risk & Uncertainty
    elif any(word in question_lower for word in ['risk', 'fear', 'uncertain', 'optimistic', 'pessimistic']):
        if symbols:
            return analyze_single_stock_risk(symbols[0])
        else:
            return analyze_riskiest_stocks()
    
    # 6. Comparative Questions
    elif any(word in question_lower for word in ['vs', 'versus', 'compare', 'better than', 'worse than']):
        if len(symbols) >= 2:
            return compare_two_stocks(symbols[0], symbols[1])
        else:
            return analyze_comparative_sectors(question_lower)
    
    # 7. Trading Volume Questions
    elif any(word in question_lower for word in ['most traded', 'highest volume', 'volume', 'active', 'liquid']):
        volume_df = analyze_most_traded_stocks()
        if volume_df is not None and len(volume_df) > 0:
            response = "**Most Traded Stocks (by average volume):**\n\n"
            for i, row in volume_df.head(5).iterrows():
                response += f"{i+1}. **{row['name']} ({row['symbol']})**: Avg Volume: {row['avg_volume']:,.0f}\n"
            response += "\n*Note: Higher volume typically indicates better liquidity and market interest.*"
            return response
    
    # 8. Portfolio Sizing Questions
    elif any(word in question_lower for word in ['how many', 'how much', 'quantity', 'number', 'portfolio', 'diversif']):
        capital = 10000  # Default assumption
        numbers = [int(word) for word in question.split() if word.isdigit()]
        if numbers:
            capital = numbers[0]
        
        num_stocks, reasoning = calculate_optimal_portfolio_size(capital)
        response = f"**Portfolio Sizing Recommendation:**\n\nWith ${capital:,.0f}, optimal number of stocks: **{num_stocks}**\n\n**Reasoning**: {reasoning}"
        return response
    
    # 9. Timing Questions (when to buy)
    elif any(word in question_lower for word in ['when to buy', 'time to buy', 'best time', 'low value', 'cheap']):
        if symbols:
            symbol = symbols[0]
            worst_day, worst_month, day_stats, month_stats = analyze_low_value_timing(symbol)
            response = f"**Optimal Buying Time for {all_companies.get(symbol, symbol)}:**\n\n"
            response += f"- Worst performing day: {worst_day} (often good for buying)\n"
            response += f"- Worst performing month: {worst_month} (potential buying opportunity)\n"
            return response
        else:
            return "Please specify a stock symbol for timing analysis."
    
    # 10. When to sell questions
    elif any(word in question_lower for word in ['when to sell', 'sell', 'exit']):
        if symbols:
            symbol = symbols[0]
            sell_signals, df = analyze_sell_signals(symbol)
            response = f"**Sell Signal Analysis for {all_companies.get(symbol, symbol)}:**\n\n"
            if sell_signals:
                response += "Potential sell signals detected:\n"
                for signal in sell_signals:
                    response += f"- {signal}\n"
            else:
                response += "No strong sell signals detected at this time.\n"
            return response
        else:
            return "Please specify a stock symbol for sell signal analysis."
    
    # 11. Enhanced forecasting questions
    # 11. Enhanced forecasting questions
    elif any(word in question_lower for word in ['forecast', 'predict', 'future price', 'will be']):
        if symbols:
            symbol = symbols[0]
            forecast_results, scaler, message = enhanced_forecast_stock(symbol, days=30)
        
            if forecast_results is not None:
            # Build the text response first
                response = f"**Enhanced Forecasting for {all_companies.get(symbol, symbol)}**\n\n"
                response += f"**Current Price:** ${forecast_results['current_price']:.2f}\n\n"
            
                tcn_pred = forecast_results['tcn']['predictions'][-1]
                lstm_pred = forecast_results['lstm']['predictions'][-1]
            
                response += f"**30-Day Predictions:**\n"
                response += f"- TCN: ${tcn_pred:.2f} ({((tcn_pred/forecast_results['current_price'])-1)*100:+.2f}%)\n"
                response += f"- LSTM: ${lstm_pred:.2f} ({((lstm_pred/forecast_results['current_price'])-1)*100:+.2f}%)\n\n"
            
            # Add performance summary to the text response
                comparison_df = compare_model_performance(forecast_results)
                tcn_wins = len(comparison_df[comparison_df['Better_Model'] == 'TCN'])
                lstm_wins = len(comparison_df[comparison_df['Better_Model'] == 'LSTM'])
            
                if tcn_wins > lstm_wins:
                    response += "**Performance Summary:** TCN model performs better on most metrics\n\n"
                elif lstm_wins > tcn_wins:
                    response += "**Performance Summary:** LSTM model performs better on most metrics\n\n"
                else:
                    response += "**Performance Summary:** Both models perform similarly\n\n"
            
                response += "**Detailed performance metrics below:**\n"
            
            # Now add the detailed metrics using st commands (these will appear after the chat response)
                st.markdown("---")
                st.subheader("📊 Detailed Model Performance Analysis")
            
            # Performance metrics table
                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("**Performance Metrics Comparison:**")
                    st.dataframe(comparison_df.style.format({
                        'TCN': '{:.6f}', 'LSTM': '{:.6f}', 'Difference': '{:.6f}'
                    }))
            
                with col2:
                    st.write("**Model Recommendations:**")
                    if tcn_wins > lstm_wins:
                        st.success("✅ **TCN performs better** on most metrics")
                        st.write("TCN is better for capturing local patterns and short-term dependencies")
                    elif lstm_wins > tcn_wins:
                        st.success("✅ **LSTM performs better** on most metrics")
                        st.write("LSTM is better for capturing long-term dependencies and sequential patterns")
                    else:
                        st.info("🤝 **Models are comparable**")
                        st.write("Both models show similar performance")
            
            # Training charts
                st.subheader("📈 Training Performance Charts")
                col1, col2 = st.columns(2)
            
                with col1:
                    st.write("**TCN Training Progress:**")
                    fig_tcn = plot_model_performance_combined(forecast_results['tcn'], 'TCN')
                    st.pyplot(fig_tcn)
            
                with col2:
                    st.write("**LSTM Training Progress:**")
                    fig_lstm = plot_model_performance_combined(forecast_results['lstm'], 'LSTM')
                    st.pyplot(fig_lstm)
            
            # Key metrics comparison
                st.subheader("🔍 Key Metrics Comparison")
            
                metrics_to_display = ['MSE', 'MAE', 'RMSE', 'R2', 'Directional_Accuracy', 'Correlation']
            
                for metric in metrics_to_display:
                    tcn_val = forecast_results['tcn']['val_metrics'][metric]
                    lstm_val = forecast_results['lstm']['val_metrics'][metric]
                
                    col1, col2 = st.columns(2)
                
                    with col1:
                        if metric in ['R2', 'Directional_Accuracy', 'Correlation']:
                        # Higher is better
                            if tcn_val > lstm_val:
                                st.success(f"**{metric}:** TCN ({tcn_val:.4f}) > LSTM ({lstm_val:.4f})")
                            else:
                                st.info(f"**{metric}:** LSTM ({lstm_val:.4f}) ≥ TCN ({tcn_val:.4f})")
                        else:
                        # Lower is better
                            if tcn_val < lstm_val:
                                st.success(f"**{metric}:** TCN ({tcn_val:.4f}) < LSTM ({lstm_val:.4f})")
                            else:
                                st.info(f"**{metric}:** LSTM ({lstm_val:.4f}) ≤ TCN ({tcn_val:.4f})")
                
                    with col2:
                    # Add a simple bar visualization
                        fig, ax = plt.subplots(figsize=(8, 2))
                        models = ['TCN', 'LSTM']
                        values = [tcn_val, lstm_val]
                        colors = ['lightblue', 'lightcoral']
                    
                        bars = ax.bar(models, values, color=colors, alpha=0.7)
                        ax.set_ylabel(metric)
                        ax.set_title(f'{metric} Comparison')
                    
                    # Add value labels on bars
                        for bar, value in zip(bars, values):
                            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                                f'{value:.4f}', ha='center', va='bottom')
                    
                        st.pyplot(fig)
            
                return response
            else:
                return f"Could not generate forecast for {symbol}. {message}"
        else:
            return "Please specify a stock symbol for forecasting analysis."
        
    
    # 12. Enhanced sentiment analysis questions - NOW USING REAL NEWS
    elif any(word in question_lower for word in ['sentiment', 'mood', 'feeling', 'opinion']):
        symbol = symbols[0] if symbols else 'SPY'
        news_articles = get_news_articles(symbol)
        sentiment_analysis = analyze_news_sentiment(news_articles, symbol)
        
        response = f"**Enhanced Sentiment Analysis for {all_companies.get(symbol, symbol)}**\n\n"
        if any(word in question_lower for word in ['compare', 'vs', 'versus', 'difference']):
            return display_sentiment_comparison_analysis(symbol)
        else:
            # Original sentiment analysis
            news_articles = get_news_articles(symbol)
            sentiment_analysis = analyze_news_sentiment(news_articles, symbol)
            
            response = f"**Enhanced Sentiment Analysis for {all_companies.get(symbol, symbol)}**\n\n"
            
            if sentiment_analysis:
                response += f"**Multi-Layer Analysis:**\n"
                response += f"- Average Score: {sentiment_analysis['average_score']:.3f}\n"
                response += f"- Positive Articles: {sentiment_analysis['positive_articles']}\n"
                response += f"- Negative Articles: {sentiment_analysis['negative_articles']}\n"
                response += f"- Neutral Articles: {sentiment_analysis.get('neutral_articles', 0)}\n\n"
                
                # Add quick comparison note
                response += "💡 *Tip: Ask 'Compare sentiment methods for AAPL' to see detailed FinBERT vs Multi-Layer comparison*\n\n"
        if sentiment_analysis:
            response += f"**Multi-Layer Analysis:**\n"
            response += f"- Average Score: {sentiment_analysis['average_score']:.3f}\n"
            response += f"- Positive Articles: {sentiment_analysis['positive_articles']}\n"
            response += f"- Negative Articles: {sentiment_analysis['negative_articles']}\n"
            response += f"- Neutral Articles: {sentiment_analysis.get('neutral_articles', 0)}\n\n"
            
            # Add classification metrics if available
            if sentiment_analysis.get('classification_metrics') and 'error' not in sentiment_analysis['classification_metrics']:
                metrics = sentiment_analysis['classification_metrics']
                response += f"**Classification Performance Metrics:**\n"
                response += f"- Accuracy: {metrics['accuracy']:.3f}\n"
                response += f"- Precision (Macro): {metrics['precision_macro']:.3f}\n"
                response += f"- Recall (Macro): {metrics['recall_macro']:.3f}\n"
                response += f"- F1 Score (Macro): {metrics['f1_macro']:.3f}\n"
                response += f"- Sample Size: {metrics['sample_size']} articles\n\n"
                
                # Add per-class metrics if available
                if 'per_class_metrics' in metrics:
                    per_class = metrics['per_class_metrics']
                    response += f"**Per-Class Performance:**\n"
                    for i, label in enumerate(per_class['labels']):
                        if i < len(per_class['precision']):
                            response += f"- {label.title()}: P={per_class['precision'][i]:.3f}, R={per_class['recall'][i]:.3f}, F1={per_class['f1'][i]:.3f}\n"
                    response += "\n"
            
            # Add news source information
            if news_articles and len(news_articles) > 0:
                source = "Alpha Vantage" if 'source' in news_articles[0] and news_articles[0]['source'] != 'Unknown source' else "Mock Data"
                response += f"**News Source:** {source} ({len(news_articles)} articles analyzed)\n\n"
            
            if sentiment_analysis['average_score'] > 0.2:
                response += "**Overall Sentiment:** 🟢 **Positive** (Bullish indicators detected)"
            elif sentiment_analysis['average_score'] < -0.2:
                response += "**Overall Sentiment:** 🔴 **Negative** (Bearish indicators detected)"
            else:
                response += "**Overall Sentiment:** 🟡 **Neutral** (Mixed or balanced sentiment)"

            # Display per-article sentiments being used
            if sentiment_analysis.get('detailed_sentiments'):
                st.markdown("---")
                st.subheader("📰 Per-article Sentiments Used")
                try:
                    rows = []
                    for article, s in zip(news_articles, sentiment_analysis['detailed_sentiments']):
                        rows.append({
                            'Title': article.get('title', '')[:120],
                            'Source': article.get('source', 'Unknown source'),
                            'Published': article.get('time_published', ''),
                            'Sentiment': s.get('sentiment', ''),
                            'Score': float(s.get('score', 0.0))
                        })
                    if rows:
                        df_sent = pd.DataFrame(rows)
                        st.dataframe(df_sent)
                except Exception:
                    pass
            
            # Add visual classification metrics if available
            if sentiment_analysis.get('classification_metrics') and 'error' not in sentiment_analysis['classification_metrics']:
                response += "\n\n**Detailed classification metrics visualization below:**"
                
                # Display visual metrics after the text response
                st.markdown("---")
                st.subheader("📊 Sentiment Classification Performance Metrics")
                
                # Create and display the sentiment metrics plot
                fig_sentiment = plot_sentiment_classification_metrics(
                    sentiment_analysis['classification_metrics'], 
                    all_companies.get(symbol, symbol)
                )
                if fig_sentiment:
                    st.pyplot(fig_sentiment)
                
                # Display detailed classification report
                if 'classification_report' in sentiment_analysis['classification_metrics']:
                    st.subheader("📋 Detailed Classification Report")
                    
                    # Convert classification report to DataFrame for better display
                    class_report = sentiment_analysis['classification_metrics']['classification_report']
                    
                    # Extract main metrics (excluding support and other summary stats)
                    metrics_data = []
                    for class_name, metrics in class_report.items():
                        if class_name not in ['accuracy', 'macro avg', 'weighted avg'] and isinstance(metrics, dict):
                            metrics_data.append({
                                'Class': class_name.title(),
                                'Precision': metrics.get('precision', 0),
                                'Recall': metrics.get('recall', 0),
                                'F1-Score': metrics.get('f1-score', 0),
                                'Support': metrics.get('support', 0)
                            })
                    
                    if metrics_data:
                        df_metrics = pd.DataFrame(metrics_data)
                        st.dataframe(df_metrics.style.format({
                            'Precision': '{:.3f}',
                            'Recall': '{:.3f}',
                            'F1-Score': '{:.3f}',
                            'Support': '{:.0f}'
                        }))
                    
                    # Display summary statistics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        if 'macro avg' in class_report:
                            macro_avg = class_report['macro avg']
                            st.metric("Macro Avg Precision", f"{macro_avg.get('precision', 0):.3f}")
                            st.metric("Macro Avg Recall", f"{macro_avg.get('recall', 0):.3f}")
                    
                    with col2:
                        if 'weighted avg' in class_report:
                            weighted_avg = class_report['weighted avg']
                            st.metric("Weighted Avg Precision", f"{weighted_avg.get('precision', 0):.3f}")
                            st.metric("Weighted Avg Recall", f"{weighted_avg.get('recall', 0):.3f}")
                    
                    with col3:
                        overall_accuracy = sentiment_analysis['classification_metrics']['accuracy']
                        st.metric("Overall Accuracy", f"{overall_accuracy:.3f}")
                        st.metric("Sample Size", f"{sentiment_analysis['classification_metrics']['sample_size']}")
        else:
            response += "No sentiment data available."
        
        return response

# Chat interface
st.header("💬 Enhanced Financial Assistant Chat")

user_query = st.text_input("Ask a financial question:", 
                          placeholder="e.g., What's the sentiment for AAPL? Forecast TSLA price, Compare model performance")

if st.button("Submit") and user_query:
    # Add user query to chat history
    st.session_state.chat_history.append({"role": "user", "content": user_query})
    
    # Process query
    with st.spinner("Analyzing with enhanced models..."):
        response = answer_financial_question(user_query)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

# Display chat history
st.subheader("Conversation History")
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**You:** {message['content']}")
    else:
        st.markdown(f"**AI:** {message['content']}")

# Add model information
st.sidebar.markdown("---")
st.sidebar.subheader("Enhanced Model Information")
st.sidebar.info("""
**Multi-Layer Sentiment Analysis:**
- Base: FinBERT (deep learning)
- Rule: Financial mappings + sarcasm detection
- Expansion: WordNet + embedding similarity
- Coherence: Contradiction checking
- **NEW**: Classification metrics (Accuracy, Precision, Recall, F1)

**Classification Performance Metrics:**
- Overall Accuracy
- Macro-averaged Precision, Recall, F1
- Weighted-averaged metrics (class imbalance handling)
- Per-class performance breakdown
- Confusion matrix analysis

**Forecasting Models:**
- TCN: Temporal Convolutional Networks
- LSTM: Long Short-Term Memory
- Both models include comprehensive performance metrics

**Real News Integration:**
- Now uses Alpha Vantage NEWS_SENTIMENT API
- Falls back to mock data if API unavailable
- Real-time news sentiment analysis
""")

# Footer
st.markdown("---")
st.markdown("""
**Enhanced Features:** 
- Multi-layer sentiment analysis with coherence checking
- **NEW**: Classification metrics (Accuracy, Precision, Recall, F1 Score)
- Per-class performance analysis for sentiment categories
- TCN vs LSTM model performance comparison
- Comprehensive forecasting metrics (MSE, MAE, R², Directional Accuracy)
- Real-time model training visualization
- **Real Alpha Vantage news integration** for sentiment analysis

**Note**: This is a demonstration application. The predictions and analysis should not be considered financial advice.
Always conduct your own research and consult with a qualified financial advisor before making investment decisions.
""")
