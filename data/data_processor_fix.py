"""
Fixed version of data processor with no relative imports.
"""
import logging
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Class to download and preprocess market data."""
    
    def __init__(self, tickers, start_date, end_date):
        """
        Initialize the data processor.
        
        Args:
            tickers (list): List of ticker symbols to download
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.price_data = None
        self.volume_data = None
        self.ohlc_data = None
        self.features = None
        self.normalized_features = None
        
    def download_data(self):
        """
        Download data using yfinance.
        
        Returns:
            pd.DataFrame: DataFrame containing adjusted close prices
        """
        logger.info(f"Downloading data for {len(self.tickers)} tickers from {self.start_date} to {self.end_date}")
        
        try:
            # Download data
            self.data = yf.download(
                self.tickers, 
                start=self.start_date, 
                end=self.end_date,
                auto_adjust=True  # Use adjusted prices
            )
            
            logger.info(f"Downloaded data shape: {self.data.shape}")
            
            # Check if we got any data
            if self.data.empty:
                logger.error("No data was downloaded")
                raise ValueError("No data was downloaded. Please check ticker symbols and date range.")
            
            # Check the structure of the data
            logger.info(f"Data columns: {self.data.columns.levels[0] if isinstance(self.data.columns, pd.MultiIndex) else self.data.columns}")
            
            # Extract price and volume data
            if isinstance(self.data.columns, pd.MultiIndex):
                # In case of multiple tickers, data has MultiIndex columns
                self.price_data = self.data['Close'].copy()  # Use Close instead of Adj Close with auto_adjust=True
                self.volume_data = self.data['Volume'].copy()
                self.ohlc_data = self.data[['Open', 'High', 'Low', 'Close']].copy()
            else:
                # In case of a single ticker, data has regular columns
                self.price_data = pd.DataFrame(self.data['Close'].copy(), columns=[self.tickers[0]])
                self.volume_data = pd.DataFrame(self.data['Volume'].copy(), columns=[self.tickers[0]])
                
                # Create a MultiIndex DataFrame for OHLC data
                ohlc_data = {}
                for col in ['Open', 'High', 'Low', 'Close']:
                    ohlc_data[col] = {self.tickers[0]: self.data[col].copy()}
                
                self.ohlc_data = pd.concat({k: pd.DataFrame(v) for k, v in ohlc_data.items()}, axis=1)
            
            logger.info(f"Processed data for {len(self.tickers)} tickers")
            
            return self.price_data
            
        except Exception as e:
            logger.error(f"Error downloading data: {str(e)}")
            raise
    
    def calculate_technical_indicators(self):
        """
        Calculate technical indicators for each asset.
        
        Returns:
            pd.DataFrame: DataFrame containing technical indicators
        """
        if self.price_data is None or self.volume_data is None:
            logger.error("Price or volume data not available. Call download_data() first.")
            raise ValueError("Price or volume data not available")
        
        logger.info("Calculating technical indicators")
        features = {}
        
        for ticker in self.tickers:
            try:
                # Get price and volume data for this ticker
                price = self.price_data[ticker]
                volume = self.volume_data[ticker]
                
                # Calculate indicators
                # 1. Returns
                returns = price.pct_change().fillna(0)
                
                # 2. Rolling statistics
                returns_5d = price.pct_change(5).fillna(0)
                returns_20d = price.pct_change(20).fillna(0)
                volatility_20d = returns.rolling(20).std().fillna(0)
                
                # 3. Moving averages
                ma_5 = price.rolling(5).mean().fillna(method='bfill') / price - 1
                ma_20 = price.rolling(20).mean().fillna(method='bfill') / price - 1
                ma_50 = price.rolling(50).mean().fillna(method='bfill') / price - 1
                
                # 4. RSI (simplified)
                delta = price.diff().fillna(0)
                gain = (delta.where(delta > 0, 0)).rolling(14).mean().fillna(0)
                loss = (-delta.where(delta < 0, 0)).rolling(14).mean().fillna(0)
                rs = gain / (loss + 1e-8)
                rsi = 100 - (100 / (1 + rs))
                
                # 5. Volume features
                volume_ma_5 = volume.rolling(5).mean().fillna(method='bfill')
                volume_ratio = volume / (volume_ma_5 + 1e-8)
                
                # Store features for this ticker
                features[f'{ticker}_returns_1d'] = returns
                features[f'{ticker}_returns_5d'] = returns_5d
                features[f'{ticker}_returns_20d'] = returns_20d
                features[f'{ticker}_volatility_20d'] = volatility_20d
                features[f'{ticker}_ma_5'] = ma_5
                features[f'{ticker}_ma_20'] = ma_20
                features[f'{ticker}_ma_50'] = ma_50
                features[f'{ticker}_rsi'] = rsi / 100  # Scale to [0, 1]
                features[f'{ticker}_volume_ratio'] = volume_ratio
                
                logger.info(f"Calculated technical indicators for {ticker}")
                
            except Exception as e:
                logger.error(f"Error calculating technical indicators for {ticker}: {str(e)}")
                # Continue with other tickers
        
        # Create features DataFrame
        self.features = pd.DataFrame(features)
        
        # Add market-wide features (using S&P 500 as proxy if available)
        if 'SPY' in self.tickers:
            market_returns = self.price_data['SPY'].pct_change().fillna(0)
            market_volatility = market_returns.rolling(20).std().fillna(0)
            
            for ticker in self.tickers:
                # Skip SPY itself
                if ticker == 'SPY':
                    continue
                
                # Add market features
                self.features[f'{ticker}_market_returns'] = market_returns
                self.features[f'{ticker}_market_volatility'] = market_volatility
        
        logger.info(f"Created features DataFrame with shape {self.features.shape}")
        return self.features