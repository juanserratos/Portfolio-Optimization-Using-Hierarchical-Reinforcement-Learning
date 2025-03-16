#!/usr/bin/env python
"""
Script for backtesting a trained deep RL trading model.
"""
import os
import sys
import argparse
import logging
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Add parent directory to path to make deep_rl_trading importable
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Use absolute imports for all deep_rl_trading modules
from data.data_processor import DataProcessor
from env.trading_env import TradingEnvironment, HierarchicalTradingEnvironment
from models.transformer import MarketTransformer, AssetAttentionTransformer
from models.actor_critic import ActorCritic, HierarchicalActorCritic
from evaluation.backtest import BacktestEngine
from utils.metrics import generate_performance_summary
from config.default_config import (
    DATA_CONFIG, ENV_CONFIG, MODEL_CONFIG, 
    TRAINING_CONFIG, EVALUATION_CONFIG, LOGGING_CONFIG, PATHS
)


def setup_logging(args):
    """
    Set up logging configuration.
    
    Args:
        args: Command-line arguments
    """
    # Create log directory if it doesn't exist
    os.makedirs(PATHS['log_dir'], exist_ok=True)
    
    # Set up logging
    log_level = getattr(logging, LOGGING_CONFIG['log_level'])
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    if args.verbose:
        # Log to console and file
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.FileHandler(os.path.join(PATHS['log_dir'], f"backtest_{args.run_id}.log")),
                logging.StreamHandler()
            ]
        )
    else:
        # Log to file only
        logging.basicConfig(
            level=log_level,
            format=log_format,
            filename=os.path.join(PATHS['log_dir'], f"backtest_{args.run_id}.log")
        )
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info(f"Starting backtest run {args.run_id}")
    
    return logger

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Backtest a trained deep RL trading model')
    
    # Required arguments
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help='Unique identifier for this run')
    
    # Optional arguments
    parser.add_argument('--hierarchical', action='store_true',
                        help='Use hierarchical model and environment')
    parser.add_argument('--asset_attention', action='store_true',
                        help='Use cross-asset attention transformer')
    parser.add_argument('--data_start', type=str,
                        help='Start date for backtest data (YYYY-MM-DD)')
    parser.add_argument('--data_end', type=str,
                        help='End date for backtest data (YYYY-MM-DD)')
    parser.add_argument('--benchmark', type=str, default='SPY',
                        help='Benchmark ticker for comparison')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose output')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Set data dates
    if args.data_start is None:
        args.data_start = DATA_CONFIG['start_date']
    if args.data_end is None:
        args.data_end = DATA_CONFIG['end_date']
    
    return args

def load_and_process_data(args, logger):
    """
    Load and process data for backtesting.
    
    Args:
        args: Command-line arguments
        logger: Logger
        
    Returns:
        tuple: (price_data, features, benchmark_returns)
    """
    logger.info("Loading and processing data...")
    
    # Create data processor
    data_processor = DataProcessor(
        tickers=DATA_CONFIG['tickers'] + [args.benchmark] if args.benchmark not in DATA_CONFIG['tickers'] else DATA_CONFIG['tickers'],
        start_date=args.data_start,
        end_date=args.data_end
    )
    
    # Download data
    price_data = data_processor.download_data()
    
    # Calculate technical indicators
    features = data_processor.calculate_technical_indicators()
    
    # Extract benchmark returns if available
    benchmark_returns = None
    if args.benchmark in price_data.columns:
        benchmark_returns = price_data[args.benchmark].pct_change().dropna()
    
    logger.info(f"Data processed. Backtest period: {args.data_start} to {args.data_end} ({len(price_data)} days)")
    
    return price_data, features, benchmark_returns

def load_model(model_path, env, hierarchical=False, asset_attention=False, device='cpu', logger=None):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint
        env: Trading environment
        hierarchical: Whether to use hierarchical model
        asset_attention: Whether to use cross-asset attention transformer
        device: Device to use
        logger: Logger
        
    Returns:
        model: Loaded model
    """
    if logger:
        logger.info(f"Loading model from {model_path}...")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    
    # Create combined model class
    class CombinedModel(torch.nn.Module):
        def __init__(self, transformer, actor_critic):
            super(CombinedModel, self).__init__()
            self.transformer = transformer
            self.actor_critic = actor_critic
        
        def forward(self, x):
            transformed_x, _ = self.transformer(x)
            return self.actor_critic(transformed_x)
        
        def get_action(self, x, deterministic=False):
            transformed_x, _ = self.transformer(x)
            return self.actor_critic.get_action(transformed_x, deterministic)
        
        def evaluate_action(self, x, action):
            transformed_x, _ = self.transformer(x)
            return self.actor_critic.evaluate_action(transformed_x, action)
    
    # Create model components
    if hierarchical:
        # Create transformer model
        if asset_attention:
            transformer = AssetAttentionTransformer(
                feature_dim=MODEL_CONFIG['feature_dim'],
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                num_heads=MODEL_CONFIG['num_heads'],
                num_layers=MODEL_CONFIG['num_layers'],
                num_assets=len(DATA_CONFIG['tickers']),
                dropout=MODEL_CONFIG['dropout']
            )
        else:
            transformer = MarketTransformer(
                feature_dim=MODEL_CONFIG['feature_dim'],
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                num_heads=MODEL_CONFIG['num_heads'],
                num_layers=MODEL_CONFIG['num_layers'],
                num_assets=len(DATA_CONFIG['tickers']),
                dropout=MODEL_CONFIG['dropout']
            )
        
        # Create hierarchical actor-critic model
        actor_critic = HierarchicalActorCritic(
            state_dim=state_dim,
            asset_classes=ENV_CONFIG['asset_classes'],
            hidden_dim=MODEL_CONFIG['actor_hidden_dim'],
            std_init=MODEL_CONFIG['std_init']
        )
    else:
        # Create transformer model
        if asset_attention:
            transformer = AssetAttentionTransformer(
                feature_dim=MODEL_CONFIG['feature_dim'],
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                num_heads=MODEL_CONFIG['num_heads'],
                num_layers=MODEL_CONFIG['num_layers'],
                num_assets=len(DATA_CONFIG['tickers']),
                dropout=MODEL_CONFIG['dropout']
            )
        else:
            transformer = MarketTransformer(
                feature_dim=MODEL_CONFIG['feature_dim'],
                hidden_dim=MODEL_CONFIG['hidden_dim'],
                num_heads=MODEL_CONFIG['num_heads'],
                num_layers=MODEL_CONFIG['num_layers'],
                num_assets=len(DATA_CONFIG['tickers']),
                dropout=MODEL_CONFIG['dropout']
            )
        
        # Create actor-critic model
        actor_critic = ActorCritic(
            state_dim=state_dim,
            action_dim=env.action_space.shape[0],
            hidden_dim=MODEL_CONFIG['actor_hidden_dim'],
            std_init=MODEL_CONFIG['std_init']
        )
    
    # Create combined model
    model = CombinedModel(transformer, actor_critic)
    
    # Load model weights
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move model to device
    model = model.to(device)
    
    # Set model to evaluation mode
    model.eval()
    
    if logger:
        logger.info("Model loaded successfully")
    
    return model

def main():
    """Main function for backtesting."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up logging
    logger = setup_logging(args)
    
    try:
        # Create results directory if it doesn't exist
        os.makedirs(PATHS['results_dir'], exist_ok=True)
        
        # Load and process data
        price_data, features, benchmark_returns = load_and_process_data(args, logger)
        
        # Create environment
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=features,
            window_size=DATA_CONFIG['window_size'],
            transaction_cost=ENV_CONFIG['transaction_cost'],
            max_position=ENV_CONFIG['max_position'],
            reward_scaling=ENV_CONFIG['reward_scaling']
        ) if not args.hierarchical else HierarchicalTradingEnvironment(
            price_data=price_data,
            feature_data=features,
            asset_classes=ENV_CONFIG['asset_classes'],
            window_size=DATA_CONFIG['window_size'],
            transaction_cost=ENV_CONFIG['transaction_cost'],
            max_position=ENV_CONFIG['max_position'],
            reward_scaling=ENV_CONFIG['reward_scaling']
        )
        
        # Load model
        model = load_model(
            model_path=args.model_path,
            env=env,
            hierarchical=args.hierarchical,
            asset_attention=args.asset_attention,
            device=args.device,
            logger=logger
        )
        
        # Create backtest engine
        backtest_engine = BacktestEngine(env, model, device=args.device)
        
        # Run backtest
        logger.info("Running backtest...")
        metrics = backtest_engine.run_backtest(deterministic=True)
        
        # Generate and log performance summary
        summary = generate_performance_summary(metrics, include_benchmark=benchmark_returns is not None)
        logger.info(f"\n{summary}")
        
        # Save performance summary to file
        with open(os.path.join(PATHS['results_dir'], f"summary_{args.run_id}.txt"), 'w') as f:
            f.write(summary)
        
        # Plot results
        logger.info("Generating plots...")
        fig1, fig2 = backtest_engine.plot_results(benchmark_returns=benchmark_returns, figsize=(12, 8))
        fig1.savefig(os.path.join(PATHS['results_dir'], f"returns_{args.run_id}.png"))
        fig2.savefig(os.path.join(PATHS['results_dir'], f"weights_{args.run_id}.png"))
        
        # Save results dataframe
        results_df = backtest_engine.get_results_dataframe()
        results_df.to_csv(os.path.join(PATHS['results_dir'], f"results_{args.run_id}.csv"))
        
        logger.info(f"Backtest completed. Results saved to {PATHS['results_dir']}")
        
    except Exception as e:
        logger.exception(f"Error during backtesting: {str(e)}")
        raise

if __name__ == '__main__':
    main()