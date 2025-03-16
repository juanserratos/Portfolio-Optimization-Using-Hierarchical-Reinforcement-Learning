#!/usr/bin/env python
"""
Simplified RL trading model that uses a projection-based approach.
"""
import os
import sys
import argparse
import logging
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Set up paths and directories
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.insert(0, PROJECT_ROOT)

# Create directory paths
DATA_DIR = os.path.join(PROJECT_ROOT, 'data')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Import other modules directly
import deep_rl_trading.data.data_processor as data_processor
import deep_rl_trading.env.trading_env as trading_env
import deep_rl_trading.models.transformer as transformer_models
import deep_rl_trading.models.actor_critic as actor_critic_models
import deep_rl_trading.training.ppo_trainer as ppo_trainer
import deep_rl_trading.utils.metrics as metrics
import deep_rl_trading.config.default_config as config

# Configuration
from deep_rl_trading.config.default_config import (
    DATA_CONFIG, ENV_CONFIG, MODEL_CONFIG, 
    TRAINING_CONFIG, EVALUATION_CONFIG, LOGGING_CONFIG, PATHS
)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description='Train a deep RL trading model')
    
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help='Unique identifier for this run')
    parser.add_argument('--hierarchical', action='store_true',
                        help='Use hierarchical model and environment')
    parser.add_argument('--asset_attention', action='store_true',
                        help='Use cross-asset attention transformer')
    parser.add_argument('--num_episodes', type=int, default=10,
                        help='Number of episodes to train for')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
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
    
    return args

def main():
    """Main training function."""
    # Parse arguments
    args = parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    logger.info(f"Starting training run {args.run_id} with device {args.device}")
    logger.info(f"Using {'hierarchical' if args.hierarchical else 'standard'} model")
    logger.info(f"Using {'asset attention' if args.asset_attention else 'standard'} transformer")
    
    try:
        # Create a data processor instance
        logger.info("Loading and processing data...")
        processor = data_processor.DataProcessor(
            tickers=DATA_CONFIG['tickers'],
            start_date=DATA_CONFIG['start_date'],
            end_date=DATA_CONFIG['end_date']
        )
        
        # Download data
        price_data = processor.download_data()
        
        # Calculate features
        features = processor.calculate_technical_indicators()
        
        # Split into train and test
        split_idx = int(len(price_data) * DATA_CONFIG['train_test_split'])
        
        train_price_data = price_data.iloc[:split_idx]
        test_price_data = price_data.iloc[split_idx:]
        
        train_features = features.iloc[:split_idx]
        test_features = features.iloc[split_idx:]
        
        logger.info(f"Data processed. Train set: {len(train_price_data)} days, Test set: {len(test_price_data)} days")
        
        # Create environments
        logger.info("Creating environments...")
        if args.hierarchical:
            train_env = trading_env.HierarchicalTradingEnvironment(
                price_data=train_price_data,
                feature_data=train_features,
                asset_classes=ENV_CONFIG['asset_classes'],
                window_size=DATA_CONFIG['window_size'],
                transaction_cost=ENV_CONFIG['transaction_cost'],
                max_position=ENV_CONFIG['max_position'],
                reward_scaling=ENV_CONFIG['reward_scaling']
            )
            
            test_env = trading_env.HierarchicalTradingEnvironment(
                price_data=test_price_data,
                feature_data=test_features,
                asset_classes=ENV_CONFIG['asset_classes'],
                window_size=DATA_CONFIG['window_size'],
                transaction_cost=ENV_CONFIG['transaction_cost'],
                max_position=ENV_CONFIG['max_position'],
                reward_scaling=ENV_CONFIG['reward_scaling']
            )
        else:
            train_env = trading_env.TradingEnvironment(
                price_data=train_price_data,
                feature_data=train_features,
                window_size=DATA_CONFIG['window_size'],
                transaction_cost=ENV_CONFIG['transaction_cost'],
                max_position=ENV_CONFIG['max_position'],
                reward_scaling=ENV_CONFIG['reward_scaling']
            )
            
            test_env = trading_env.TradingEnvironment(
                price_data=test_price_data,
                feature_data=test_features,
                window_size=DATA_CONFIG['window_size'],
                transaction_cost=ENV_CONFIG['transaction_cost'],
                max_position=ENV_CONFIG['max_position'],
                reward_scaling=ENV_CONFIG['reward_scaling']
            )
        
        # Create a simplified model architecture
        logger.info("Creating a simplified model architecture...")
        
        # Calculate dimensions
        window_size = DATA_CONFIG['window_size']
        num_assets = len(DATA_CONFIG['tickers'])
        portfolio_size = num_assets
        
        # Get total feature size from environment
        state_dim = train_env.observation_space.shape[0]
        feature_size = (state_dim - portfolio_size) // window_size
        
        logger.info(f"Environment observation shape: {state_dim}")
        logger.info(f"Features per step: {feature_size}")
        
        # Get action dimension
        if args.hierarchical:
            action_dim = None
        else:
            action_dim = train_env.action_space.shape[0]
            
        # Define the model size parameters
        hidden_dim = MODEL_CONFIG['hidden_dim']
            
        # Create a simplified model
        class SimplifiedModel(nn.Module):
            def __init__(self, input_dim, hidden_dim, output_dim, window_size, num_assets):
                super(SimplifiedModel, self).__init__()
                self.window_size = window_size
                self.num_assets = num_assets
                
                # Calculate the actual feature size (excluding portfolio weights)
                feature_size = input_dim - num_assets
                
                # Network layers - FIXED: Use feature_size instead of input_dim
                self.feature_encoder = nn.Sequential(
                    nn.Linear(feature_size, hidden_dim * 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim * 2, hidden_dim)
                )
                
                # Actor (policy) network
                self.actor = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, output_dim),
                    nn.Tanh()  # Output in [-1, 1] range for position sizing
                )
                
                # Critic (value) network
                self.critic = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
                
                # For hierarchical models
                if hasattr(train_env, 'asset_classes'):
                    self.asset_classes = train_env.asset_classes
                    
            def forward(self, x):
                # Extract features (exclude portfolio weights)
                features = x[:, :-self.num_assets]
                
                # Process features
                encoded = self.feature_encoder(features)
                
                # Get action and value
                action = self.actor(encoded)
                value = self.critic(encoded)
                
                # Return a 3-element tuple to match the expected interface from ActorCritic
                # (action_mean, action_log_std, state_value)
                return action, None, value
            
            def get_action(self, x, deterministic=False):
                action, _, value = self.forward(x)  # Now unpacking 3 values
                
                if deterministic:
                    return action, None, None
                
                # Add noise for exploration during training
                if not deterministic:
                    noise = torch.randn_like(action) * 0.1
                    action = torch.clamp(action + noise, -1, 1)
                
                # Simple implementation - in practice, would calculate log_prob properly
                log_prob = torch.zeros(1, 1)
                entropy = torch.zeros(1, 1)
                
                return action, log_prob, entropy
            
            def evaluate_action(self, x, action):
                _, _, value = self.forward(x)  # Now unpacking 3 values
                
                # Simple implementation - in practice, would calculate these properly
                log_prob = torch.zeros(action.size(0), 1)
                entropy = torch.zeros(action.size(0), 1)
                
                return log_prob, entropy, value
        
        # Create model instance
        model = SimplifiedModel(
            input_dim=state_dim, 
            hidden_dim=hidden_dim, 
            output_dim=num_assets,
            window_size=window_size,
            num_assets=num_assets
        ).to(args.device)
        
        logger.info(f"Created simplified model with input_dim={state_dim}, hidden_dim={hidden_dim}, output_dim={num_assets}")
        
        # Create trainer
        logger.info("Creating trainer...")
        if args.hierarchical:
            trainer = ppo_trainer.HierarchicalPPOTrainer(
                env=train_env,
                model=model,
                device=args.device,
                lr=args.learning_rate,
                gamma=TRAINING_CONFIG['gamma'],
                lambda_gae=TRAINING_CONFIG['lambda_gae'],
                clip_epsilon=TRAINING_CONFIG['clip_epsilon'],
                entropy_coef=TRAINING_CONFIG['entropy_coef'],
                value_coef=TRAINING_CONFIG['value_coef'],
                max_grad_norm=TRAINING_CONFIG['max_grad_norm']
            )
        else:
            trainer = ppo_trainer.PPOTrainer(
                env=train_env,
                model=model,
                device=args.device,
                lr=args.learning_rate,
                gamma=TRAINING_CONFIG['gamma'],
                lambda_gae=TRAINING_CONFIG['lambda_gae'],
                clip_epsilon=TRAINING_CONFIG['clip_epsilon'],
                entropy_coef=TRAINING_CONFIG['entropy_coef'],
                value_coef=TRAINING_CONFIG['value_coef'],
                max_grad_norm=TRAINING_CONFIG['max_grad_norm']
            )
        
        # Train model - using a smaller number of episodes for quick testing
        logger.info(f"Training for {args.num_episodes} episodes...")
        
        try:
            episode_rewards, episode_lengths, losses = trainer.train(
                num_episodes=args.num_episodes,
                max_steps=TRAINING_CONFIG['max_steps'],
                update_interval=TRAINING_CONFIG['update_interval'],
                verbose=args.verbose
            )
            
            # Save trained model
            model_path = os.path.join(MODELS_DIR, f"model_{args.run_id}.pt")
            trainer.save_model(model_path)
            logger.info(f"Model saved to {model_path}")
            
            # Plot training performance
            plt.figure(figsize=(10, 6))
            plt.plot(episode_rewards)
            plt.title('Episode Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.grid(True)
            plt.savefig(os.path.join(RESULTS_DIR, f"rewards_{args.run_id}.png"))
            
            logger.info("Training completed successfully")
            
        except Exception as e:
            logger.exception(f"Error during training: {str(e)}")
            print(f"Error during training: {str(e)}")
            
    except Exception as e:
        logger.exception(f"Error in setup: {str(e)}")
        print(f"Error in setup: {str(e)}")

if __name__ == "__main__":
    main()