"""
Feature engineering module for calculating technical and fundamental indicators.
"""
import pandas as pd
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

def calculate_returns(prices, windows=[1, 5, 20]):
    """
    Calculate returns over different time windows.
    
    Args:
        prices (pd.DataFrame): Price data
        windows (list): List of windows to calculate returns for
        
    Returns:
        dict: Dictionary of return series
    """
    returns = {}
    for window in windows:
        window_returns = prices.pct_change(window).fillna(0)
        returns[f'returns_{window}d'] = window_returns
    return returns

def calculate_volatility(returns, windows=[5, 20, 60]):
    """
    Calculate rolling volatility over different windows.
    
    Args:
        returns (pd.Series): Return data
        windows (list): List of windows to calculate volatility for
        
    Returns:
        dict: Dictionary of volatility series
    """
    volatility = {}
    for window in windows:
        vol = returns.rolling(window).std().fillna(0)
        volatility[f'volatility_{window}d'] = vol
    return volatility

def calculate_moving_averages(prices, windows=[5, 20, 50, 200]):
    """
    Calculate moving averages over different windows.
    
    Args:
        prices (pd.DataFrame): Price data
        windows (list): List of windows to calculate MAs for
        
    Returns:
        dict: Dictionary of MA series
    """
    mas = {}
    for window in windows:
        ma = prices.rolling(window).mean().fillna(method='bfill')
        mas[f'ma_{window}'] = ma
        
        # Calculate normalized MA (MA / price - 1)
        normalized_ma = ma.divide(prices) - 1
        mas[f'ma_{window}_normalized'] = normalized_ma
    return mas

def calculate_rsi(prices, window=14):
    """
    Calculate Relative Strength Index.
    
    Args:
        prices (pd.DataFrame): Price data
        window (int): RSI window
        
    Returns:
        pd.Series: RSI values
    """
    delta = prices.diff().fillna(0)
    gain = (delta.where(delta > 0, 0)).rolling(window).mean().fillna(0)
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean().fillna(0)
    rs = gain / (loss + 1e-8)  # Adding small epsilon to avoid division by zero
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).
    
    Args:
        prices (pd.DataFrame): Price data
        fast (int): Fast EMA window
        slow (int): Slow EMA window
        signal (int): Signal line window
        
    Returns:
        tuple: (MACD line, Signal line, Histogram)
    """
    fast_ema = prices.ewm(span=fast, adjust=False).mean()
    slow_ema = prices.ewm(span=slow, adjust=False).mean()
    macd_line = fast_ema - slow_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return {
        'macd_line': macd_line,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    }

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """
    Calculate Bollinger Bands.
    
    Args:
        prices (pd.DataFrame): Price data
        window (int): Window for moving average
        num_std (float): Number of standard deviations
        
    Returns:
        tuple: (Middle Band, Upper Band, Lower Band)
    """
    middle_band = prices.rolling(window).mean()
    rolling_std = prices.rolling(window).std()
    
    upper_band = middle_band + (rolling_std * num_std)
    lower_band = middle_band - (rolling_std * num_std)
    
    # Calculate BB width and %B
    bb_width = (upper_band - lower_band) / middle_band
    percent_b = (prices - lower_band) / (upper_band - lower_band)
    
    return {
        'bb_middle': middle_band,
        'bb_upper': upper_band,
        'bb_lower': lower_band,
        'bb_width': bb_width,
        'bb_percent_b': percent_b
    }

def calculate_volume_indicators(prices, volume, windows=[5, 20]):
    """
    Calculate volume-based indicators.
    
    Args:
        prices (pd.DataFrame): Price data
        volume (pd.DataFrame): Volume data
        windows (list): List of windows for moving averages
        
    Returns:
        dict: Dictionary of volume indicators
    """
    volume_indicators = {}
    
    # Volume Moving Averages
    for window in windows:
        vol_ma = volume.rolling(window).mean().fillna(method='bfill')
        volume_indicators[f'volume_ma_{window}'] = vol_ma
        
        # Volume ratio (current volume / volume MA)
        volume_ratio = volume / (vol_ma + 1e-8)  # Adding small epsilon to avoid division by zero
        volume_indicators[f'volume_ratio_{window}'] = volume_ratio
    
    # On-Balance Volume (OBV)
    obv = pd.DataFrame(0, index=prices.index, columns=prices.columns)
    for ticker in prices.columns:
        price_change = prices[ticker].diff().fillna(0)
        obv[ticker] = (volume[ticker] * np.sign(price_change)).cumsum()
    
    volume_indicators['obv'] = obv
    
    # Money Flow Index (MFI)
    typical_price = pd.DataFrame(index=prices.index, columns=prices.columns)
    money_flow = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    for ticker in prices.columns:
        # Get OHLC data for this ticker
        high = prices[ticker].rolling(2).max()
        low = prices[ticker].rolling(2).min()
        close = prices[ticker]
        
        # Calculate typical price (high + low + close) / 3
        typical_price[ticker] = (high + low + close) / 3
        
        # Calculate money flow
        money_flow[ticker] = typical_price[ticker] * volume[ticker]
    
    # Calculate MFI for 14-day period
    period = 14
    mfi = pd.DataFrame(index=prices.index, columns=prices.columns)
    
    for ticker in prices.columns:
        positive_flow = []
        negative_flow = []
        
        # Get the typical price change
        tp_diff = typical_price[ticker].diff()
        
        # Calculate positive and negative money flow
        for i in range(1, len(typical_price)):
            if tp_diff.iloc[i] > 0:
                positive_flow.append(money_flow[ticker].iloc[i])
                negative_flow.append(0)
            elif tp_diff.iloc[i] < 0:
                positive_flow.append(0)
                negative_flow.append(money_flow[ticker].iloc[i])
            else:
                positive_flow.append(0)
                negative_flow.append(0)
        
        # Convert to series with the same index as the original data
        positive_series = pd.Series(positive_flow, index=tp_diff.index[1:])
        negative_series = pd.Series(negative_flow, index=tp_diff.index[1:])
        
        # Calculate the money flow ratio
        positive_mf = positive_series.rolling(window=period).sum()
        negative_mf = negative_series.rolling(window=period).sum()
        
        # Calculate money flow index
        mfi_series = 100 - (100 / (1 + positive_mf / (negative_mf + 1e-8)))
        
        # Assign to the dataframe
        mfi[ticker] = mfi_series
    
    volume_indicators['mfi'] = mfi
    
    return volume_indicators

def calculate_all_features(price_data, volume_data=None, ohlc_data=None):
    """
    Calculate all technical indicators.
    
    Args:
        price_data (pd.DataFrame): Price data
        volume_data (pd.DataFrame): Volume data
        ohlc_data (pd.DataFrame): OHLC data
        
    Returns:
        pd.DataFrame: DataFrame containing all technical indicators
    """
    logger.info("Calculating all technical features")
    all_features = {}
    
    # Calculate returns
    returns_dict = calculate_returns(price_data)
    all_features.update(returns_dict)
    
    # Calculate volatility
    volatility_dict = calculate_volatility(returns_dict['returns_1d'])
    all_features.update(volatility_dict)
    
    # Calculate moving averages
    ma_dict = calculate_moving_averages(price_data)
    all_features.update(ma_dict)
    
    # Calculate RSI
    all_features['rsi'] = calculate_rsi(price_data)
    
    # Calculate MACD
    macd_dict = calculate_macd(price_data)
    all_features.update(macd_dict)
    
    # Calculate Bollinger Bands
    bb_dict = calculate_bollinger_bands(price_data)
    all_features.update(bb_dict)
    
    # Calculate volume indicators if volume data is available
    if volume_data is not None:
        volume_dict = calculate_volume_indicators(price_data, volume_data)
        all_features.update(volume_dict)
    
    # Create a DataFrame from all features
    feature_df = pd.concat(all_features, axis=1)
    
    return feature_df