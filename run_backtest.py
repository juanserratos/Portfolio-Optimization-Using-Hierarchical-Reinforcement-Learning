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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use absolute imports for all deep_rl_trading modules
from data.data_processor import DataProcessor
from env.trading_env import TradingEnvironment, HierarchicalTradingEnvironment
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
    
    handlers = [
        logging.FileHandler(os.path.join(PATHS['log_dir'], f"backtest_{args.run_id}.log")),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
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
    parser.add_argument('--data_start', type=str, default=DATA_CONFIG['start_date'],
                        help=f'Start date for backtest data (YYYY-MM-DD), default: {DATA_CONFIG["start_date"]}')
    parser.add_argument('--data_end', type=str, default=DATA_CONFIG['end_date'],
                        help=f'End date for backtest data (YYYY-MM-DD), default: {DATA_CONFIG["end_date"]}')
    parser.add_argument('--benchmark', type=str, default='SPY',
                        help='Benchmark ticker for comparison')
    parser.add_argument('--test_only', action='store_true',
                        help='Only backtest on the test portion of the data')
    parser.add_argument('--transaction_cost', type=float, default=ENV_CONFIG['transaction_cost'],
                        help=f'Transaction cost as fraction, default: {ENV_CONFIG["transaction_cost"]}')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
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
    
    # Create data directory if it doesn't exist
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    
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
    
    # If test_only is specified, use only the test portion of data
    if args.test_only:
        split_idx = int(len(price_data) * DATA_CONFIG['train_test_split'])
        price_data = price_data.iloc[split_idx:]
        features = features.iloc[split_idx:]
        if benchmark_returns is not None:
            benchmark_returns = benchmark_returns.iloc[split_idx:]
        logger.info(f"Using test-only data: {len(price_data)} days")
    else:
        logger.info(f"Using full data: {len(price_data)} days")
    
    logger.info(f"Data processed. Backtest period: {args.data_start} to {args.data_end}")
    
    return price_data, features, benchmark_returns

def create_environment(price_data, feature_data, args, logger):
    """
    Create trading environment for backtesting.
    
    Args:
        price_data: Price data
        feature_data: Feature data
        args: Command-line arguments
        logger: Logger
        
    Returns:
        TradingEnvironment: Trading environment
    """
    logger.info(f"Creating trading environment...")
    
    if args.hierarchical:
        env = HierarchicalTradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            asset_classes=ENV_CONFIG['asset_classes'],
            window_size=DATA_CONFIG['window_size'],
            transaction_cost=args.transaction_cost,
            max_position=ENV_CONFIG['max_position'],
            reward_scaling=ENV_CONFIG['reward_scaling']
        )
        logger.info(f"Created hierarchical trading environment with {len(ENV_CONFIG['asset_classes'])} asset classes")
    else:
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            window_size=DATA_CONFIG['window_size'],
            transaction_cost=args.transaction_cost,
            max_position=ENV_CONFIG['max_position'],
            reward_scaling=ENV_CONFIG['reward_scaling']
        )
        logger.info(f"Created standard trading environment with {len(price_data.columns)} assets")
    
    return env

def load_model(args, env, logger):
    """
    Load trained model from checkpoint.
    
    Args:
        args: Command-line arguments
        env: Trading environment
        logger: Logger
        
    Returns:
        model: Loaded model
    """
    logger.info(f"Loading model from {args.model_path}...")
    
    # Get the device
    device = args.device
    
    try:
        # Load the model state dict
        checkpoint = torch.load(args.model_path, map_location=device)
        
        # Create a new model instance with the same architecture
        if args.hierarchical:
            # Create simplified hierarchical model
            from train_model import create_hierarchical_model
            model = create_hierarchical_model(env, device, logger)
        else:
            # Create simplified model
            from train_model import create_model
            model = create_model(env, False, device, logger)
        
        # Load state dict into the model
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Set model to evaluation mode
        model.eval()
        
        logger.info("Model loaded successfully")
        return model
        
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        raise

def run_backtest(args, env, model, benchmark_returns, logger):
    """
    Run backtest on model.
    
    Args:
        args: Command-line arguments
        env: Trading environment
        model: Trained model
        benchmark_returns: Benchmark returns for comparison
        logger: Logger
        
    Returns:
        BacktestEngine: Backtest engine with results
    """
    logger.info("Running backtest...")
    
    # Create backtest engine
    backtest_engine = BacktestEngine(env, model, device=args.device)
    
    # Run backtest
    metrics = backtest_engine.run_backtest(deterministic=True)
    
    # Generate and log performance summary
    summary = generate_performance_summary(metrics, include_benchmark=benchmark_returns is not None)
    logger.info(f"\n{summary}")
    
    # Save performance summary to file
    os.makedirs(PATHS['results_dir'], exist_ok=True)
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
    
    return backtest_engine

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set up logging
    logger = setup_logging(args)
    
    try:
        # Create all necessary directories
        for directory in PATHS.values():
            os.makedirs(directory, exist_ok=True)
        
        # Load and process data
        price_data, features, benchmark_returns = load_and_process_data(args, logger)
        
        # Create environment
        env = create_environment(price_data, features, args, logger)
        
        # Load model
        model = load_model(args, env, logger)
        
        # Run backtest
        backtest_engine = run_backtest(args, env, model, benchmark_returns, logger)
        
        logger.info("Backtest completed successfully")
        
    except Exception as e:
        logger.exception(f"Error during backtesting: {str(e)}")
        raise
    
if __name__ == '__main__':
    main()