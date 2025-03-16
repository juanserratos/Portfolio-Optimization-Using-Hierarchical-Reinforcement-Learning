"""
Default configuration for the deep RL trading framework.
"""

# Data configuration
DATA_CONFIG = {
    # S&P 500 components and ETFs for major asset classes
    'tickers': [
        # Large Cap US Equities
        'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META',
        # Sector ETFs
        'XLF',  # Financials
        'XLK',  # Technology
        'XLE',  # Energy
        'XLV',  # Healthcare
        'XLP',  # Consumer Staples
        # Bond ETFs
        'AGG',  # Total Bond Market
        'TLT',  # Long-Term Treasury
        # Commodities
        'GLD',  # Gold
        # Market Index
        'SPY',  # S&P 500
    ],
    'start_date': '2013-01-01',
    'end_date': '2023-03-10',
    'train_test_split': 0.8,  # 80% train, 20% test
    'window_size': 60,  # 60 days of history
}

# Environment configuration
ENV_CONFIG = {
    'transaction_cost': 0.003,  # 10 bps per trade
    'max_position': 0.4,  # Maximum position size as fraction of capital
    'reward_scaling': 0.5,  # Scaling factor for rewards
    'asset_classes': {
        'us_equities': [0, 1, 2, 3, 4],
        'sectors': [5, 6, 7, 8, 9],
        'bonds': [10, 11],
        'commodities': [12],
        'market': [13]
    }
}

# Model configuration
MODEL_CONFIG = {
    'feature_dim': 20,  # Number of features per asset
    'hidden_dim': 128,  # Hidden dimension
    'num_heads': 8,  # Number of attention heads
    'num_layers': 4,  # Number of transformer layers
    'dropout': 0.1,  # Dropout rate
    'actor_hidden_dim': 128,  # Actor hidden dimension
    'critic_hidden_dim': 128,  # Critic hidden dimension
    'std_init': 0.5,  # Initial standard deviation for exploration
    'weight_decay': 0.01,  # L2 regularization
    'dropout': 0.3,  # Increased dropout
}

# Training configuration
TRAINING_CONFIG = {
    'num_episodes': 500,  # Number of episodes to train for
    'max_steps': 252,  # Maximum number of steps per episode (1 year of trading days)
    'update_interval': 2048,  # Number of steps between updates
    'learning_rate': 3e-4,  # Learning rate
    'gamma': 0.99,  # Discount factor
    'lambda_gae': 0.95,  # GAE lambda parameter
    'clip_epsilon': 0.2,  # PPO clip parameter
    'entropy_coef': 0.01,  # Entropy coefficient
    'value_coef': 0.5,  # Value loss coefficient
    'max_grad_norm': 0.5,  # Maximum gradient norm
    'batch_size': 64,  # Batch size
    'update_epochs': 10,  # Number of update epochs
    'deterministic_eval': True,  # Use deterministic actions for evaluation
}

# Evaluation configuration
EVALUATION_CONFIG = {
    'risk_free_rate': 0.0,  # Daily risk-free rate
    'periods_per_year': 252,  # Number of trading days per year
    'benchmark_ticker': 'SPY',  # Benchmark ticker
}

# Logging configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_interval': 10,  # Log interval in episodes
    'save_interval': 50,  # Save interval in episodes
    'eval_interval': 20,  # Evaluation interval in episodes
}

# Paths
PATHS = {
    'data_dir': 'data',
    'model_dir': 'models',
    'results_dir': 'results',
    'log_dir': 'logs',
}