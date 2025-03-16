"""
Backtesting functionality for evaluating trading strategies.
"""
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import logging
from tqdm import tqdm

# FIXED: Use absolute import instead of relative import
import utils.metrics as metrics 
from utils.metrics import calculate_performance_metrics

# Configure logging
logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Engine for backtesting trading strategies.
    """
    
    def __init__(self, env, model, device=None):
        """
        Initialize backtest engine.
        
        Args:
            env: Trading environment
            model: Trained model
            device: Torch device (cpu or cuda)
        """
        self.env = env
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        # Set model to evaluation mode
        self.model.eval()
        
        # Initialize history storage
        self.reset_history()
        
    def reset_history(self):
        """Reset history storage."""
        self.portfolio_values = []
        self.returns = []
        self.positions = []
        self.transactions = []
        
    def run_backtest(self, deterministic=True):
        """
        Run backtest.
        
        Args:
            deterministic (bool): Whether to use deterministic actions
            
        Returns:
            dict: Performance metrics
        """
        logger.info("Starting backtest")
        
        # Reset environment and history
        obs = self.env.reset()
        self.reset_history()
        self.portfolio_values.append(1.0)  # Start with initial value
        
        # Reset tracking variables
        done = False
        
        # Run through entire environment
        while not done:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action from model
            with torch.no_grad():
                action, _, _ = self.model.get_action(obs_tensor, deterministic=deterministic)
                
            # If action is a dict (hierarchical), convert it to numpy
            if isinstance(action, dict):
                action_np = {
                    'strategic': action['strategic'].squeeze().cpu().numpy(),
                    'tactical': {
                        class_name: action['tactical'][class_name].squeeze().cpu().numpy()
                        for class_name in action['tactical']
                    }
                }
            else:
                action_np = action.squeeze().cpu().numpy()
            
            # Take action in environment
            next_obs, reward, done, info = self.env.step(action_np)
            
            # Store history
            self.portfolio_values.append(info['portfolio_value'])
            self.positions.append(self.env.current_weights.copy())
            
            # Calculate transactions (change in positions)
            if len(self.positions) > 1:
                transaction = np.abs(self.positions[-1] - self.positions[-2])
                self.transactions.append(transaction)
            
            # Store returns
            if len(self.portfolio_values) > 1:
                ret = self.portfolio_values[-1] / self.portfolio_values[-2] - 1
                self.returns.append(ret)
            
            # Update observation
            obs = next_obs
        
        # Calculate performance metrics
        metrics = calculate_performance_metrics(
            returns=self.returns,
            portfolio_values=self.portfolio_values
        )
        
        logger.info(f"Backtest completed with final portfolio value: {self.portfolio_values[-1]:.4f}")
        logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}, Max Drawdown: {metrics['max_drawdown']:.4f}")
        
        return metrics
    
    def plot_results(self, benchmark_returns=None, figsize=(12, 8)):
        """
        Plot backtest results.
        
        Args:
            benchmark_returns (list): List of benchmark returns for comparison
            figsize (tuple): Figure size
            
        Returns:
            tuple: Matplotlib figures
        """
        if not self.portfolio_values:
            logger.error("No backtest results to plot. Run backtest first.")
            return None
        
        # Convert portfolio values to cumulative returns
        cumulative_returns = np.array(self.portfolio_values) / self.portfolio_values[0] - 1
        
        # Create figure for portfolio performance
        fig1, axes1 = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
        
        # Plot cumulative returns
        axes1[0].plot(cumulative_returns, label='Strategy')
        
        # Plot benchmark if provided
        if benchmark_returns is not None:
            benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns)) - 1
            axes1[0].plot(benchmark_cumulative, label='Benchmark')
        
        axes1[0].set_title('Cumulative Returns')
        axes1[0].set_ylabel('Return (%)')
        axes1[0].legend()
        axes1[0].grid(True)
        
        # Plot drawdowns
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / (1 + running_max)
        axes1[1].fill_between(range(len(drawdowns)), 0, drawdowns, color='r', alpha=0.3)
        axes1[1].set_title('Drawdowns')
        axes1[1].set_ylabel('Drawdown (%)')
        axes1[1].set_xlabel('Trading Days')
        axes1[1].grid(True)
        
        fig1.tight_layout()
        
        # Create figure for position allocation
        fig2, ax2 = plt.subplots(figsize=figsize)
        
        # Convert positions to array for plotting
        positions_array = np.array(self.positions)
        
        # Get asset names from environment
        if hasattr(self.env, 'price_data') and hasattr(self.env.price_data, 'columns'):
            asset_names = self.env.price_data.columns
        else:
            asset_names = [f'Asset {i+1}' for i in range(positions_array.shape[1])]
        
        # Plot positions over time
        for i, asset in enumerate(asset_names):
            ax2.plot(positions_array[:, i], label=asset)
        
        ax2.set_title('Portfolio Allocation Over Time')
        ax2.set_ylabel('Weight')
        ax2.set_xlabel('Trading Days')
        ax2.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax2.grid(True)
        
        fig2.tight_layout()
        
        return fig1, fig2
    
    def get_results_dataframe(self):
        """
        Get backtest results as a DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with backtest results
        """
        if not self.portfolio_values:
            logger.error("No backtest results to convert. Run backtest first.")
            return None
        
        # Get asset names from environment
        if hasattr(self.env, 'price_data') and hasattr(self.env.price_data, 'columns'):
            asset_names = self.env.price_data.columns
        else:
            asset_names = [f'Asset {i+1}' for i in range(len(self.positions[0]))]
        
        # Create DataFrame with dates from environment
        if hasattr(self.env, 'price_data') and hasattr(self.env.price_data, 'index'):
            dates = self.env.price_data.index[self.env.window_size:self.env.window_size + len(self.portfolio_values)]
        else:
            dates = pd.date_range(start='2000-01-01', periods=len(self.portfolio_values))
        
        # Create results DataFrame
        results = pd.DataFrame({
            'portfolio_value': self.portfolio_values,
        }, index=dates)
        
        # Add returns
        results['return'] = results['portfolio_value'].pct_change().fillna(0)
        
        # Fix for length mismatch - ensure positions array has the right dimensions
        positions_array = np.array(self.positions)
        
        # Check for length mismatch and fix if necessary
        if len(positions_array) != len(results):
            logger.warning(f"Length mismatch: positions ({len(positions_array)}) vs results ({len(results)})")
            
            # If positions array is shorter, pad with zeros or the last position
            if len(positions_array) < len(results):
                # Create a padding array with the same number of assets
                padding_shape = (len(results) - len(positions_array), positions_array.shape[1])
                # Use the last position as padding
                padding = np.tile(positions_array[-1], (len(results) - len(positions_array), 1))
                positions_array = np.vstack([positions_array, padding])
            # If positions array is longer, truncate it
            else:
                positions_array = positions_array[:len(results)]
        
        # Add positions for each asset
        for i, asset in enumerate(asset_names):
            if i < positions_array.shape[1]:
                results[f'weight_{asset}'] = positions_array[:, i]
        
        # Add cumulative return and drawdown
        results['cumulative_return'] = results['portfolio_value'] / results['portfolio_value'].iloc[0] - 1
        running_max = results['cumulative_return'].cummax()
        results['drawdown'] = (results['cumulative_return'] - running_max) / (1 + running_max)
        
        return results
    
def compare_strategies(strategies, env_factory, names=None, figsize=(12, 8)):
    """
    Compare multiple trading strategies.
    
    Args:
        strategies (list): List of trained models to compare
        env_factory (callable): Function that creates a new environment instance
        names (list): List of strategy names
        figsize (tuple): Figure size
        
    Returns:
        tuple: (DataFrame with comparison, matplotlib figure)
    """
    if names is None:
        names = [f'Strategy {i+1}' for i in range(len(strategies))]
    
    results = []
    
    for i, strategy in enumerate(strategies):
        # Create fresh environment for this strategy
        env = env_factory()
        
        # Create backtest engine
        engine = BacktestEngine(env, strategy)
        
        # Run backtest
        metrics = engine.run_backtest(deterministic=True)
        
        # Get results DataFrame
        df = engine.get_results_dataframe()
        
        # Add strategy name
        df['strategy'] = names[i]
        
        # Add to results list
        results.append(df)
        
    # Concatenate results
    all_results = pd.concat(results, axis=0)
    
    # Create comparison figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot cumulative returns for each strategy
    for name in names:
        strategy_data = all_results[all_results['strategy'] == name]
        ax.plot(strategy_data.index, strategy_data['cumulative_return'], label=name)
    
    ax.set_title('Strategy Comparison - Cumulative Returns')
    ax.set_ylabel('Return (%)')
    ax.legend()
    ax.grid(True)
    
    return all_results, fig