"""
Performance metrics calculation for trading strategies.
"""
import numpy as np
import pandas as pd
from scipy import stats
import logging

# Configure logging
logger = logging.getLogger(__name__)

def calculate_returns(portfolio_values):
    """
    Calculate returns from portfolio values.
    
    Args:
        portfolio_values (list): List of portfolio values
        
    Returns:
        np.array: Array of returns
    """
    if portfolio_values is None or len(portfolio_values) < 2:
        return np.array([])
        
    # Convert to numpy array if necessary
    portfolio_array = np.array(portfolio_values)
    
    # Calculate returns as percentage change
    returns = portfolio_array[1:] / portfolio_array[:-1] - 1
    
    return returns

def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate annualized Sharpe ratio.
    
    Args:
        returns (np.array): Array of returns
        risk_free_rate (float): Daily risk-free rate
        periods_per_year (int): Number of periods per year (252 for daily data)
        
    Returns:
        float: Sharpe ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate excess returns over risk-free rate
    excess_returns = returns - risk_free_rate
    
    # Calculate annualized Sharpe ratio
    sharpe_ratio = np.mean(excess_returns) / (np.std(excess_returns, ddof=1) + 1e-8) * np.sqrt(periods_per_year)
    
    return sharpe_ratio

def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate annualized Sortino ratio.
    
    Args:
        returns (np.array): Array of returns
        risk_free_rate (float): Daily risk-free rate
        periods_per_year (int): Number of periods per year (252 for daily data)
        
    Returns:
        float: Sortino ratio
    """
    if len(returns) < 2:
        return 0.0
    
    # Calculate excess returns over risk-free rate
    excess_returns = returns - risk_free_rate
    
    # Calculate downside returns (negative returns only)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) == 0:
        return float('inf')  # No downside risk
    
    # Calculate annualized Sortino ratio
    sortino_ratio = np.mean(excess_returns) / (np.std(downside_returns, ddof=1) + 1e-8) * np.sqrt(periods_per_year)
    
    return sortino_ratio

def calculate_max_drawdown(portfolio_values):
    """
    Calculate maximum drawdown.
    
    Args:
        portfolio_values (list): List of portfolio values
        
    Returns:
        tuple: (Maximum drawdown, start index, end index)
    """
    # Fix the problematic conditional by checking explicitly if it's None or empty
    if portfolio_values is None or len(portfolio_values) < 2:
        return 0.0, 0, 0
    
    # Convert to numpy array if necessary
    portfolio_array = np.array(portfolio_values)
    
    # Calculate running maximum
    running_max = np.maximum.accumulate(portfolio_array)
    
    # Calculate drawdowns
    drawdowns = (portfolio_array - running_max) / running_max
    
    # Find maximum drawdown and its indices
    max_drawdown = np.min(drawdowns)
    end_idx = np.argmin(drawdowns)
    
    # Find start of the drawdown period
    start_idx = np.argmax(portfolio_array[:end_idx+1])
    
    return abs(max_drawdown), start_idx, end_idx

def calculate_calmar_ratio(returns, portfolio_values, periods_per_year=252):
    """
    Calculate Calmar ratio (annualized return / maximum drawdown).
    
    Args:
        returns (np.array): Array of returns
        portfolio_values (list): List of portfolio values
        periods_per_year (int): Number of periods per year (252 for daily data)
        
    Returns:
        float: Calmar ratio
    """
    if len(returns) < 2 or len(portfolio_values) < 2:
        return 0.0
    
    # Calculate annualized return
    annualized_return = np.mean(returns) * periods_per_year
    
    # Calculate maximum drawdown
    max_drawdown, _, _ = calculate_max_drawdown(portfolio_values)
    
    if max_drawdown == 0:
        return float('inf')  # No drawdown
    
    # Calculate Calmar ratio
    calmar_ratio = annualized_return / max_drawdown
    
    return calmar_ratio

def calculate_omega_ratio(returns, threshold=0.0, periods_per_year=252):
    """
    Calculate Omega ratio.
    
    Args:
        returns (np.array): Array of returns
        threshold (float): Minimum acceptable return
        periods_per_year (int): Number of periods per year (252 for daily data)
        
    Returns:
        float: Omega ratio
    """
    if len(returns) < 2:
        return 1.0
    
    # Separate returns into gains and losses relative to threshold
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    
    if len(losses) == 0 or np.sum(losses) == 0:
        return float('inf')  # No losses
    
    # Calculate Omega ratio
    omega_ratio = np.sum(gains) / np.sum(losses)
    
    return omega_ratio

def calculate_information_ratio(returns, benchmark_returns):
    """
    Calculate Information ratio (active return / tracking error).
    
    Args:
        returns (np.array): Array of strategy returns
        benchmark_returns (np.array): Array of benchmark returns
        
    Returns:
        float: Information ratio
    """
    if len(returns) < 2 or len(benchmark_returns) < 2 or len(returns) != len(benchmark_returns):
        logger.warning("Cannot calculate information ratio: insufficient or mismatched data")
        return 0.0
    
    # Calculate active returns
    active_returns = returns - benchmark_returns
    
    # Calculate tracking error
    tracking_error = np.std(active_returns, ddof=1)
    
    if tracking_error == 0:
        return 0.0  # No tracking error
    
    # Calculate Information ratio
    information_ratio = np.mean(active_returns) / tracking_error
    
    return information_ratio

def calculate_win_rate(returns):
    """
    Calculate win rate (percentage of positive returns).
    
    Args:
        returns (np.array): Array of returns
        
    Returns:
        float: Win rate
    """
    if len(returns) == 0:
        return 0.0
    
    # Calculate number of positive returns
    positive_returns = np.sum(returns > 0)
    
    # Calculate win rate
    win_rate = positive_returns / len(returns)
    
    return win_rate

def calculate_average_gain_loss_ratio(returns):
    """
    Calculate average gain/loss ratio.
    
    Args:
        returns (np.array): Array of returns
        
    Returns:
        float: Average gain/loss ratio
    """
    if len(returns) == 0:
        return 0.0
    
    # Separate gains and losses
    gains = returns[returns > 0]
    losses = returns[returns < 0]
    
    if len(gains) == 0 or len(losses) == 0:
        return 0.0 if len(gains) == 0 else float('inf')
    
    # Calculate average gain and loss
    avg_gain = np.mean(gains)
    avg_loss = abs(np.mean(losses))
    
    # Calculate gain/loss ratio
    gain_loss_ratio = avg_gain / avg_loss
    
    return gain_loss_ratio

def calculate_performance_metrics(returns=None, portfolio_values=None, benchmark_returns=None, risk_free_rate=0.0, periods_per_year=252):
    """
    Calculate comprehensive performance metrics for a trading strategy.
    
    Args:
        returns (list): List of returns (optional if portfolio_values is provided)
        portfolio_values (list): List of portfolio values (optional if returns is provided)
        benchmark_returns (list): List of benchmark returns (optional)
        risk_free_rate (float): Daily risk-free rate
        periods_per_year (int): Number of periods per year (252 for daily data)
        
    Returns:
        dict: Dictionary of performance metrics
    """
    # Ensure we have returns
    if returns is None and portfolio_values is not None:
        returns = calculate_returns(portfolio_values)
    elif returns is not None:
        returns = np.array(returns)
    else:
        logger.error("Either returns or portfolio_values must be provided")
        return {}
    
    # Ensure we have portfolio values
    if portfolio_values is None and returns is not None:
        # Calculate portfolio values assuming initial value of 1.0
        portfolio_values = np.cumprod(1 + np.concatenate([[0], returns]))
    elif portfolio_values is not None:
        portfolio_values = np.array(portfolio_values)
    
    # Calculate metrics
    metrics = {}
    
    # Basic statistics
    metrics['total_return'] = portfolio_values[-1] / portfolio_values[0] - 1
    metrics['annualized_return'] = (1 + metrics['total_return']) ** (periods_per_year / len(returns)) - 1
    metrics['volatility'] = np.std(returns, ddof=1) * np.sqrt(periods_per_year)
    
    # Ratios
    metrics['sharpe_ratio'] = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    metrics['sortino_ratio'] = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    
    # Drawdowns
    metrics['max_drawdown'], metrics['max_drawdown_start'], metrics['max_drawdown_end'] = calculate_max_drawdown(portfolio_values)
    metrics['calmar_ratio'] = calculate_calmar_ratio(returns, portfolio_values, periods_per_year)
    
    # Win/loss statistics
    metrics['win_rate'] = calculate_win_rate(returns)
    metrics['gain_loss_ratio'] = calculate_average_gain_loss_ratio(returns)
    metrics['omega_ratio'] = calculate_omega_ratio(returns, risk_free_rate)
    
    # Benchmark-related metrics (if benchmark provided)
    if benchmark_returns is not None:
        benchmark_returns = np.array(benchmark_returns)
        metrics['information_ratio'] = calculate_information_ratio(returns, benchmark_returns)
        metrics['beta'] = np.cov(returns, benchmark_returns)[0, 1] / np.var(benchmark_returns)
        metrics['alpha'] = metrics['annualized_return'] - risk_free_rate - metrics['beta'] * (np.mean(benchmark_returns) * periods_per_year - risk_free_rate)
        metrics['r_squared'] = np.corrcoef(returns, benchmark_returns)[0, 1] ** 2
    
    # Other metrics
    metrics['skewness'] = stats.skew(returns)
    metrics['kurtosis'] = stats.kurtosis(returns)
    metrics['var_95'] = np.percentile(returns, 5)
    metrics['cvar_95'] = np.mean(returns[returns <= metrics['var_95']])
    
    return metrics

def generate_performance_summary(metrics, include_benchmark=False):
    """
    Generate a formatted performance summary from metrics.
    
    Args:
        metrics (dict): Dictionary of performance metrics
        include_benchmark (bool): Whether to include benchmark-related metrics
        
    Returns:
        str: Formatted performance summary
    """
    summary = "Performance Summary\n"
    summary += "===================\n\n"
    
    # Return metrics
    summary += "Return Metrics:\n"
    summary += f"- Total Return: {metrics['total_return']:.2%}\n"
    summary += f"- Annualized Return: {metrics['annualized_return']:.2%}\n"
    summary += f"- Volatility: {metrics['volatility']:.2%}\n"
    
    # Risk-adjusted metrics
    summary += "\nRisk-Adjusted Metrics:\n"
    summary += f"- Sharpe Ratio: {metrics['sharpe_ratio']:.2f}\n"
    summary += f"- Sortino Ratio: {metrics['sortino_ratio']:.2f}\n"
    summary += f"- Calmar Ratio: {metrics['calmar_ratio']:.2f}\n"
    summary += f"- Omega Ratio: {metrics['omega_ratio']:.2f}\n"
    
    # Drawdown metrics
    summary += "\nDrawdown Metrics:\n"
    summary += f"- Maximum Drawdown: {metrics['max_drawdown']:.2%}\n"
    
    # Win/loss metrics
    summary += "\nWin/Loss Metrics:\n"
    summary += f"- Win Rate: {metrics['win_rate']:.2%}\n"
    summary += f"- Gain/Loss Ratio: {metrics['gain_loss_ratio']:.2f}\n"
    summary += f"- VaR (95%): {metrics['var_95']:.2%}\n"
    summary += f"- CVaR (95%): {metrics['cvar_95']:.2%}\n"
    
    # Distribution metrics
    summary += "\nDistribution Metrics:\n"
    summary += f"- Skewness: {metrics['skewness']:.2f}\n"
    summary += f"- Kurtosis: {metrics['kurtosis']:.2f}\n"
    
    # Benchmark metrics
    if include_benchmark and 'information_ratio' in metrics:
        summary += "\nBenchmark Metrics:\n"
        summary += f"- Information Ratio: {metrics['information_ratio']:.2f}\n"
        summary += f"- Beta: {metrics['beta']:.2f}\n"
        summary += f"- Alpha: {metrics['alpha']:.2%}\n"
        summary += f"- R-Squared: {metrics['r_squared']:.2f}\n"
    
    return summary