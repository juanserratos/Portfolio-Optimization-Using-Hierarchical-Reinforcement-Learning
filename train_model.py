#!/usr/bin/env python
"""
Main script for training a deep RL trading model with progress bar and adaptive training.
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
from tqdm import tqdm
import time

# Add parent directory to path to make deep_rl_trading importable
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Use absolute imports for all deep_rl_trading modules
from data.data_processor import DataProcessor
from env.trading_env import TradingEnvironment, HierarchicalTradingEnvironment
from models.transformer import MarketTransformer, AssetAttentionTransformer
from models.actor_critic import ActorCritic, HierarchicalActorCritic
from training.ppo_trainer import PPOTrainer, HierarchicalPPOTrainer
from evaluation.backtest import BacktestEngine
from utils.metrics import calculate_performance_metrics
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
    
    # We always want console output for better visibility
    handlers = [
        logging.FileHandler(os.path.join(PATHS['log_dir'], f"training_{args.run_id}.log")),
        logging.StreamHandler()
    ]
    
    logging.basicConfig(
        level=log_level,
        format=log_format,
        handlers=handlers
    )
    
    # Get logger
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training run {args.run_id}")
    
    return logger

def parse_args():
    """
    Parse command-line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Train a deep RL trading model')
    
    # Required arguments
    parser.add_argument('--run_id', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help='Unique identifier for this run')
    
    # Optional arguments
    parser.add_argument('--hierarchical', action='store_true',
                        help='Use hierarchical model and environment')
    parser.add_argument('--asset_attention', action='store_true',
                        help='Use cross-asset attention transformer')
    parser.add_argument('--num_episodes', type=int, default=1000,  # Increased for adaptive training
                        help='Maximum number of episodes to train for')
    parser.add_argument('--min_episodes', type=int, default=50,  
                        help='Minimum number of episodes to train for')
    parser.add_argument('--learning_rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--verbose', action='store_true', default=True,  # Always verbose by default
                        help='Enable verbose output')
    parser.add_argument('--log_interval', type=int, default=5,  # Log every 5 episodes for less spam
                        help='Log interval in episodes')
    parser.add_argument('--update_interval', type=int, default=1024,  # Reduced for more frequent updates
                        help='Number of steps between policy updates')
    parser.add_argument('--eval_interval', type=int, default=20,  # Evaluate every 20 episodes
                        help='Evaluation interval in episodes')
    parser.add_argument('--target_sharpe', type=float, default=1.0,  # Target Sharpe ratio
                        help='Target Sharpe ratio to achieve before stopping training')
    parser.add_argument('--early_stopping', action='store_true',  # Enable early stopping
                        help='Enable early stopping based on performance metrics')
    parser.add_argument('--patience', type=int, default=10,  # Patience for early stopping
                        help='Number of evaluations without improvement before stopping')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    return args

def load_and_process_data(logger):
    """
    Load and process data.
    
    Args:
        logger: Logger
        
    Returns:
        tuple: (train_price_data, train_features, test_price_data, test_features)
    """
    logger.info("Loading and processing data...")
    
    # Create data directory if it doesn't exist
    os.makedirs(PATHS['data_dir'], exist_ok=True)
    
    # Create data processor
    data_processor = DataProcessor(
        tickers=DATA_CONFIG['tickers'],
        start_date=DATA_CONFIG['start_date'],
        end_date=DATA_CONFIG['end_date']
    )
    
    # Download data
    price_data = data_processor.download_data()
    
    # Calculate technical indicators
    features = data_processor.calculate_technical_indicators()
    
    # Split into train and test sets
    split_idx = int(len(price_data) * DATA_CONFIG['train_test_split'])
    
    train_price_data = price_data.iloc[:split_idx]
    test_price_data = price_data.iloc[split_idx:]
    
    train_features = features.iloc[:split_idx]
    test_features = features.iloc[split_idx:]
    
    logger.info(f"Data processed. Train set: {len(train_price_data)} days, Test set: {len(test_price_data)} days")
    
    return train_price_data, train_features, test_price_data, test_features

def create_environment(price_data, feature_data, hierarchical=False, logger=None):
    """
    Create trading environment.
    
    Args:
        price_data: Price data
        feature_data: Feature data
        hierarchical: Whether to use hierarchical environment
        logger: Logger
        
    Returns:
        TradingEnvironment: Trading environment
    """
    if logger:
        logger.info(f"Creating trading environment...")
    
    if hierarchical:
        env = HierarchicalTradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            asset_classes=ENV_CONFIG['asset_classes'],
            window_size=DATA_CONFIG['window_size'],
            transaction_cost=ENV_CONFIG['transaction_cost'],
            max_position=ENV_CONFIG['max_position'],
            reward_scaling=ENV_CONFIG['reward_scaling']
        )
    else:
        env = TradingEnvironment(
            price_data=price_data,
            feature_data=feature_data,
            window_size=DATA_CONFIG['window_size'],
            transaction_cost=ENV_CONFIG['transaction_cost'],
            max_position=ENV_CONFIG['max_position'],
            reward_scaling=ENV_CONFIG['reward_scaling']
        )
    
    return env

def create_model(env, hierarchical=False, device='cpu', logger=None):
    """
    Create simplified model.
    
    Args:
        env: Trading environment
        hierarchical: Whether to use hierarchical model
        device: Device to use
        logger: Logger
        
    Returns:
        nn.Module: Simplified model
    """
    if logger:
        logger.info(f"Creating simplified model...")
    
    # Get state and action dimensions
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Define hidden dimension
    hidden_dim = MODEL_CONFIG['hidden_dim']
    num_assets = len(DATA_CONFIG['tickers'])
    
    # Create simplified model
    class SimplifiedModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super(SimplifiedModel, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.output_dim = output_dim
            
            # Network layers - use smaller hidden layers due to large input size
            self.feature_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
            
            # Actor (policy) network
            self.actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
                nn.Tanh()  # Output in [-1, 1] range for position sizing
            )
            
            # Initialize action standard deviation (for exploration)
            self.actor_log_std = nn.Parameter(torch.ones(output_dim) * np.log(0.5))
            
            # Critic (value) network
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
            
            # Asset classes for hierarchical models
            if hierarchical:
                self.asset_classes = env.asset_classes
        
        def init_weights(self):
            """Initialize weights with Xavier initialization"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
        def forward(self, x):
            # Process features
            encoded = self.feature_encoder(x)
            
            # Get action mean and log standard deviation
            action_mean = self.actor_mean(encoded)
            action_log_std = self.actor_log_std.expand_as(action_mean)
            
            # Get value
            value = self.critic(encoded)
            
            return action_mean, action_log_std, value
            
        def get_action(self, x, deterministic=False):
            action_mean, action_log_std, _ = self.forward(x)
            
            if deterministic:
                return action_mean, None, None
            
            # Sample from Gaussian distribution
            action_std = torch.exp(action_log_std)
            normal = torch.distributions.Normal(action_mean, action_std)
            
            # Sample action
            action = normal.sample()
            
            # Calculate log probability and entropy
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = normal.entropy().sum(dim=-1, keepdim=True)
            
            # Clip action to valid range
            action = torch.clamp(action, -1, 1)
            
            return action, log_prob, entropy
            
        def evaluate_action(self, x, action):
            action_mean, action_log_std, value = self.forward(x)
            
            # Calculate log probability and entropy using Gaussian distribution
            action_std = torch.exp(action_log_std)
            normal = torch.distributions.Normal(action_mean, action_std)
            
            log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
            entropy = normal.entropy().sum(dim=-1, keepdim=True)
            
            return log_prob, entropy, value
    
    # Create model instance
    model = SimplifiedModel(
        input_dim=state_dim, 
        hidden_dim=hidden_dim, 
        output_dim=action_dim
    )
    
    # Initialize weights
    model.init_weights()
    
    # Move model to device
    model = model.to(device)
    
    logger.info(f"Created simplified model with input_dim={state_dim}, hidden_dim={hidden_dim}, output_dim={action_dim}")
    
    return model

def create_hierarchical_model(env, device='cpu', logger=None):
    """
    Create simplified hierarchical model.
    
    Args:
        env: Hierarchical trading environment
        device: Device to use
        logger: Logger
        
    Returns:
        nn.Module: Simplified hierarchical model
    """
    if logger:
        logger.info(f"Creating simplified hierarchical model...")
    
    # Get state dimension
    state_dim = env.observation_space.shape[0]
    
    # Get asset classes
    asset_classes = env.asset_classes
    num_classes = len(asset_classes)
    
    # Define hidden dimension
    hidden_dim = MODEL_CONFIG['hidden_dim']
    
    # Create simplified hierarchical model
    class SimplifiedHierarchicalModel(nn.Module):
        def __init__(self, input_dim, hidden_dim, asset_classes):
            super(SimplifiedHierarchicalModel, self).__init__()
            self.input_dim = input_dim
            self.hidden_dim = hidden_dim
            self.asset_classes = asset_classes
            self.num_classes = len(asset_classes)
            
            # Network layers - use smaller hidden layers due to large input size
            self.feature_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU()
            )
            
            # Strategic actor (asset class allocation)
            self.strategic_actor_mean = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_classes),
                nn.Tanh()  # Outputs in [-1, 1] range
            )
            
            self.strategic_actor_log_std = nn.Parameter(torch.ones(self.num_classes) * np.log(0.5))
            
            # Tactical actors (asset allocation within each class)
            self.tactical_actors_mean = nn.ModuleDict({
                class_name: nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, len(asset_indices)),
                    nn.Tanh()  # Outputs in [-1, 1] range
                ) for class_name, asset_indices in asset_classes.items()
            })
            
            self.tactical_actors_log_std = nn.ParameterDict({
                class_name: nn.Parameter(torch.ones(len(asset_indices)) * np.log(0.5))
                for class_name, asset_indices in asset_classes.items()
            })
            
            # Critic (value function)
            self.critic = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            )
        
        def init_weights(self):
            """Initialize weights with Xavier initialization"""
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
            
        def forward(self, x):
            # Process features
            encoded = self.feature_encoder(x)
            
            # Strategic actor (policy for asset classes)
            strategic_mean = self.strategic_actor_mean(encoded)
            strategic_log_std = self.strategic_actor_log_std.expand_as(strategic_mean)
            
            # Tactical actors (policies for assets within each class)
            tactical_means = {}
            tactical_log_stds = {}
            
            for class_name, asset_indices in self.asset_classes.items():
                tactical_means[class_name] = self.tactical_actors_mean[class_name](encoded)
                tactical_log_stds[class_name] = self.tactical_actors_log_std[class_name].expand_as(tactical_means[class_name])
            
            # Critic (value function)
            value = self.critic(encoded)
            
            return strategic_mean, strategic_log_std, tactical_means, tactical_log_stds, value
            
        def get_action(self, x, deterministic=False):
            strategic_mean, strategic_log_std, tactical_means, tactical_log_stds, _ = self.forward(x)
            
            if deterministic:
                action = {
                    'strategic': strategic_mean,
                    'tactical': {class_name: tactical_means[class_name] for class_name in self.asset_classes}
                }
                return action, None, None
            
            # Sample strategic action
            strategic_std = torch.exp(strategic_log_std)
            strategic_normal = torch.distributions.Normal(strategic_mean, strategic_std)
            strategic_action = strategic_normal.sample()
            strategic_log_prob = strategic_normal.log_prob(strategic_action).sum(dim=-1, keepdim=True)
            strategic_entropy = strategic_normal.entropy().sum(dim=-1, keepdim=True)
            
            # Clip strategic action to valid range
            strategic_action = torch.clamp(strategic_action, -1, 1)
            
            # Sample tactical actions
            tactical_actions = {}
            tactical_log_probs = {}
            tactical_entropies = {}
            
            for class_name in self.asset_classes:
                tactical_std = torch.exp(tactical_log_stds[class_name])
                tactical_normal = torch.distributions.Normal(tactical_means[class_name], tactical_std)
                
                tactical_action = tactical_normal.sample()
                tactical_log_prob = tactical_normal.log_prob(tactical_action).sum(dim=-1, keepdim=True)
                tactical_entropy = tactical_normal.entropy().sum(dim=-1, keepdim=True)
                
                # Clip tactical action to valid range
                tactical_action = torch.clamp(tactical_action, -1, 1)
                
                tactical_actions[class_name] = tactical_action
                tactical_log_probs[class_name] = tactical_log_prob
                tactical_entropies[class_name] = tactical_entropy
            
            # Combine actions, log probs, and entropies
            action = {
                'strategic': strategic_action,
                'tactical': tactical_actions
            }
            
            log_prob = {
                'strategic': strategic_log_prob,
                'tactical': tactical_log_probs
            }
            
            entropy = {
                'strategic': strategic_entropy,
                'tactical': tactical_entropies
            }
            
            return action, log_prob, entropy
            
        def evaluate_action(self, x, action):
            strategic_mean, strategic_log_std, tactical_means, tactical_log_stds, value = self.forward(x)
            
            # Evaluate strategic action
            strategic_std = torch.exp(strategic_log_std)
            strategic_normal = torch.distributions.Normal(strategic_mean, strategic_std)
            
            strategic_log_prob = strategic_normal.log_prob(action['strategic']).sum(dim=-1, keepdim=True)
            strategic_entropy = strategic_normal.entropy().sum(dim=-1, keepdim=True)
            
            # Evaluate tactical actions
            tactical_log_probs = {}
            tactical_entropies = {}
            
            for class_name in self.asset_classes:
                tactical_std = torch.exp(tactical_log_stds[class_name])
                tactical_normal = torch.distributions.Normal(tactical_means[class_name], tactical_std)
                
                tactical_log_prob = tactical_normal.log_prob(action['tactical'][class_name]).sum(dim=-1, keepdim=True)
                tactical_entropy = tactical_normal.entropy().sum(dim=-1, keepdim=True)
                
                tactical_log_probs[class_name] = tactical_log_prob
                tactical_entropies[class_name] = tactical_entropy
            
            # Combine log probs and entropies
            log_prob = {
                'strategic': strategic_log_prob,
                'tactical': tactical_log_probs
            }
            
            entropy = {
                'strategic': strategic_entropy,
                'tactical': tactical_entropies
            }
            
            return log_prob, entropy, value
    
    # Create model instance
    model = SimplifiedHierarchicalModel(
        input_dim=state_dim, 
        hidden_dim=hidden_dim, 
        asset_classes=asset_classes
    )
    
    # Initialize weights
    model.init_weights()
    
    # Move model to device
    model = model.to(device)
    
    logger.info(f"Created simplified hierarchical model with input_dim={state_dim}, hidden_dim={hidden_dim}")
    
    return model

def modified_ppo_trainer_train(self, num_episodes, max_steps, update_interval=2048, verbose=True, log_interval=1, 
                              pbar=None, early_stopping=False, eval_interval=20, eval_env=None, eval_model=None, 
                              target_sharpe=1.0, patience=10, device='cpu', min_episodes=50):
    """
    Enhanced train method with progress bar and early stopping.
    
    Args:
        num_episodes (int): Number of episodes to train for
        max_steps (int): Maximum number of steps per episode
        update_interval (int): Number of steps between updates
        verbose (bool): Whether to print progress
        log_interval (int): How often to log progress (in episodes)
        pbar (tqdm): Progress bar object
        early_stopping (bool): Whether to use early stopping
        eval_interval (int): How often to evaluate the model (in episodes)
        eval_env: Evaluation environment
        eval_model: Model to evaluate
        target_sharpe (float): Target Sharpe ratio to achieve
        patience (int): Number of evaluations without improvement before stopping
        device (str): Device to use for evaluation
        min_episodes (int): Minimum number of episodes to train before early stopping
        
    Returns:
        tuple: (episode_rewards, episode_lengths, losses)
    """
    # Track progress
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    # Early stopping variables
    best_sharpe = -float('inf')
    no_improvement_count = 0
    best_model_state = None
    
    # Training loop
    total_steps = 0
    episode = 0
    
    logger = logging.getLogger(__name__)
    logger.info(f"Starting training with {num_episodes} episodes, {max_steps} max steps per episode")
    logger.info(f"Update interval: {update_interval}, Log interval: {log_interval}, Eval interval: {eval_interval}")
    
    # Create a progress bar if not provided
    if pbar is None and verbose:
        pbar = tqdm(total=num_episodes, desc="Training Progress")
    
    start_time = time.time()
    
    while episode < num_episodes:
        # Storage buffers
        observations_buffer = []
        actions_buffer = []
        rewards_buffer = []
        values_buffer = []
        log_probs_buffer = []
        dones_buffer = []
        
        # Steps collected in current interval
        steps_in_interval = 0
        
        # Collect experience for the current update interval
        while steps_in_interval < update_interval and episode < num_episodes:
            # Collect a trajectory
            observations, actions, rewards, values, log_probs, dones, final_value = self.collect_trajectory(max_steps)
            
            # Add trajectory to buffers
            observations_buffer.extend(observations)
            actions_buffer.extend(actions)
            rewards_buffer.extend(rewards)
            values_buffer.extend(values)
            log_probs_buffer.extend(log_probs)
            dones_buffer.extend(dones)
            
            # Calculate episode statistics
            episode_reward = sum(rewards)
            episode_length = len(rewards)
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Update counters
            steps_in_interval += episode_length
            total_steps += episode_length
            episode += 1
            
            # Update progress bar
            if pbar is not None and verbose:
                elapsed = time.time() - start_time
                if len(episode_rewards) > 0:
                    recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                    pbar.set_postfix({
                        'episode': f'{episode}/{num_episodes}',
                        'steps': total_steps,
                        'reward': f'{episode_reward:.2f}',
                        'avg_reward': f'{avg_reward:.2f}',
                        'elapsed': f'{elapsed:.1f}s'
                    })
                pbar.update(1)
            
            # Log progress
            if verbose and (episode % log_interval == 0 or episode == num_episodes):
                recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                avg_reward = sum(recent_rewards) / len(recent_rewards)
                if len(rewards) > 0:
                    min_reward = min(rewards)
                    max_reward = max(rewards)
                    logger.info(f"Episode {episode}/{num_episodes}, Total Steps: {total_steps}, "
                               f"Episode Reward: {episode_reward:.4f}, Avg Reward: {avg_reward:.4f}, "
                               f"Episode Length: {episode_length}, "
                               f"Min Step Reward: {min_reward:.4f}, Max Step Reward: {max_reward:.4f}")
                else:
                    logger.warning(f"Episode {episode} had 0 steps!")
            
            # Evaluate model
            if early_stopping and eval_env is not None and eval_model is not None and episode % eval_interval == 0 and episode >= min_episodes:
                # Create backtest engine
                backtest_engine = BacktestEngine(eval_env, eval_model, device=device)
                
                # Run backtest
                metrics = backtest_engine.run_backtest(deterministic=True)
                
                # Get Sharpe ratio
                sharpe_ratio = metrics.get('sharpe_ratio', -float('inf'))
                
                logger.info(f"Eval at episode {episode} - Sharpe Ratio: {sharpe_ratio:.4f} (Best: {best_sharpe:.4f})")
                
                # Check for improvement
                if sharpe_ratio > best_sharpe:
                    best_sharpe = sharpe_ratio
                    no_improvement_count = 0
                    # Save the best model state
                    best_model_state = {k: v.cpu().clone() for k, v in eval_model.state_dict().items()}
                    
                    logger.info(f"New best Sharpe ratio: {best_sharpe:.4f}")
                    
                    # Check if we've met the target
                    if best_sharpe >= target_sharpe:
                        logger.info(f"Reached target Sharpe ratio of {target_sharpe:.2f}. Stopping training.")
                        
                        # Restore the best model state
                        if best_model_state is not None:
                            eval_model.load_state_dict(best_model_state)
                            self.model.load_state_dict(best_model_state)
                            
                        break
                else:
                    no_improvement_count += 1
                    logger.info(f"No improvement for {no_improvement_count} evaluations")
                    
                    # Check if we should stop
                    if no_improvement_count >= patience:
                        logger.info(f"No improvement for {patience} evaluations. Stopping training.")
                        
                        # Restore the best model state
                        if best_model_state is not None:
                            eval_model.load_state_dict(best_model_state)
                            self.model.load_state_dict(best_model_state)
                            
                        break
        
        # Skip update if we have no data
        if len(observations_buffer) == 0:
            logger.warning("No data collected for update, skipping.")
            continue
        
        # Log buffer sizes for debugging
        logger.info(f"Collected {len(observations_buffer)} observations for update")
        
        # Convert NumPy arrays to contiguous arrays before creating tensors
        # This addresses the warning about slow tensor creation
        observations_buffer = np.array(observations_buffer, dtype=np.float32)
        
        if isinstance(actions_buffer[0], dict):  # Hierarchical actions
            strategic_actions = np.array([a['strategic'] for a in actions_buffer], dtype=np.float32)
            tactical_actions = {
                class_name: np.array([a['tactical'][class_name] for a in actions_buffer], dtype=np.float32)
                for class_name in actions_buffer[0]['tactical']
            }
            
            strategic_log_probs = np.array([lp['strategic'] for lp in log_probs_buffer], dtype=np.float32).reshape(-1, 1)
            tactical_log_probs = {
                class_name: np.array([lp['tactical'][class_name] for lp in log_probs_buffer], dtype=np.float32).reshape(-1, 1)
                for class_name in log_probs_buffer[0]['tactical']
            }
            
            actions_dict = {
                'strategic': torch.FloatTensor(strategic_actions),
                'tactical': {
                    class_name: torch.FloatTensor(tactical_actions[class_name]) 
                    for class_name in tactical_actions
                }
            }
            
            log_probs_dict = {
                'strategic': torch.FloatTensor(strategic_log_probs),
                'tactical': {
                    class_name: torch.FloatTensor(tactical_log_probs[class_name])
                    for class_name in tactical_log_probs
                }
            }
            
            actions_tensor = actions_dict
            log_probs_tensor = log_probs_dict
        else:  # Regular actions
            actions_tensor = torch.FloatTensor(np.array(actions_buffer, dtype=np.float32))
            log_probs_tensor = torch.FloatTensor(np.array(log_probs_buffer, dtype=np.float32)).unsqueeze(1)
        
        # Compute returns and advantages
        returns, advantages = self.compute_gae(
            rewards_buffer, values_buffer, final_value, dones_buffer
        )
        
        # Convert to tensors
        observations_tensor = torch.FloatTensor(observations_buffer)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1)
        advantages_tensor = torch.FloatTensor(advantages).unsqueeze(1)
        
        # Update policy and value function
        loss_dict = self.update(
            observations_tensor, 
            actions_tensor, 
            log_probs_tensor, 
            returns_tensor, 
            advantages_tensor
        )
        
        losses.append(loss_dict)
        
        # Log update
        if verbose:
            logger.info(f"Policy update at episode {episode}, Total Steps: {total_steps}, "
                       f"Policy Loss: {loss_dict['policy_loss']:.4f}, "
                       f"Value Loss: {loss_dict['value_loss']:.4f}, "
                       f"Entropy Loss: {loss_dict['entropy_loss']:.4f}")
    
    # Close progress bar
    if pbar is not None and verbose:
        pbar.close()
    
    return episode_rewards, episode_lengths, losses

def evaluate_model(model, env, device='cpu', logger=None):
    """
    Evaluate model on environment.
    
    Args:
        model: Model to evaluate
        env: Environment to evaluate on
        device: Device to use
        logger: Logger
        
    Returns:
        dict: Performance metrics
    """
    if logger:
        logger.info("Evaluating model...")
    
    # Create backtest engine
    backtest_engine = BacktestEngine(env, model, device=device)
    
    # Run backtest
    metrics = backtest_engine.run_backtest(deterministic=True)
    
    if logger:
        logger.info(f"Evaluation results - Sharpe Ratio: {metrics.get('sharpe_ratio', -float('inf')):.4f}, "
                   f"Total Return: {metrics.get('total_return', 0.0):.4f}, "
                   f"Max Drawdown: {metrics.get('max_drawdown', 1.0):.4f}")
    
    return metrics

def train_model(train_env, test_env, model, args, logger):
    """
    Train the model.
    
    Args:
        train_env: Training environment
        test_env: Testing environment
        model: Model to train
        args: Command-line arguments
        logger: Logger
        
    Returns:
        tuple: (Trainer, training results)
    """
    logger.info("Starting model training...")
    
    # Create model directory if it doesn't exist
    os.makedirs(PATHS['model_dir'], exist_ok=True)
    
    # Create results directory if it doesn't exist
    os.makedirs(PATHS['results_dir'], exist_ok=True)
    
    # Create trainer
    if args.hierarchical:
        trainer = HierarchicalPPOTrainer(
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
        trainer = PPOTrainer(
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
    
    # Replace the train method with our enhanced version
    import types
    trainer.train = types.MethodType(modified_ppo_trainer_train, trainer)
    
    # Train model
    num_episodes = args.num_episodes
    max_steps = TRAINING_CONFIG['max_steps']
    
    logger.info(f"Training for maximum {num_episodes} episodes, max {max_steps} steps per episode...")
    logger.info(f"Early stopping: {args.early_stopping}, Target Sharpe: {args.target_sharpe}, Patience: {args.patience}")
    
    # Create progress bar
    pbar = tqdm(total=num_episodes, desc="Training Progress")
    
    # Start training
    training_results = {
        'episode_rewards': [],
        'episode_lengths': [],
        'losses': [],
        'test_metrics': [],
    }
    
    # Train for specified number of episodes
    episode_rewards, episode_lengths, losses = trainer.train(
        num_episodes=num_episodes,
        max_steps=max_steps,
        update_interval=args.update_interval,
        verbose=args.verbose,
        log_interval=args.log_interval,
        pbar=pbar,
        early_stopping=args.early_stopping,
        eval_interval=args.eval_interval,
        eval_env=test_env,
        eval_model=model,
        target_sharpe=args.target_sharpe,
        patience=args.patience,
        device=args.device,
        min_episodes=args.min_episodes
    )
    
    # Store training results
    training_results['episode_rewards'].extend(episode_rewards)
    training_results['episode_lengths'].extend(episode_lengths)
    training_results['losses'].extend(losses)
    
    # Calculate average reward
    avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
    logger.info(f"Training completed with average reward: {avg_reward:.4f}")
    
    # Evaluate the model
    test_metrics = evaluate_model(model, test_env, args.device, logger)
    training_results['test_metrics'] = test_metrics
    
    # Save model
    model_path = os.path.join(PATHS['model_dir'], f"model_{args.run_id}.pt")
    trainer.save_model(model_path)
    logger.info(f"Model saved to {model_path}")

    # Plot training performance
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(episode_rewards)
    plt.title('Episode Rewards')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    window_size = min(10, len(episode_rewards))
    if window_size > 0:
        rolling_mean = pd.Series(episode_rewards).rolling(window=window_size).mean()
        plt.plot(rolling_mean)
        plt.title(f'Rolling Average Reward (window={window_size})')
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(PATHS['results_dir'], f"rewards_{args.run_id}.png"))
    
    # Plot additional training metrics
    if len(losses) > 0:
        plt.figure(figsize=(12, 8))
        
        # Extract loss values
        policy_losses = [loss.get('policy_loss', 0) for loss in losses]
        value_losses = [loss.get('value_loss', 0) for loss in losses]
        entropy_losses = [loss.get('entropy_loss', 0) for loss in losses]
        
        plt.subplot(3, 1, 1)
        plt.plot(policy_losses)
        plt.title('Policy Loss')
        plt.grid(True)
        
        plt.subplot(3, 1, 2)
        plt.plot(value_losses)
        plt.title('Value Loss')
        plt.grid(True)
        
        plt.subplot(3, 1, 3)
        plt.plot(entropy_losses)
        plt.title('Entropy Loss')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(PATHS['results_dir'], f"losses_{args.run_id}.png"))
    
    logger.info("Training completed successfully")
    
    return trainer, training_results

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
            
        # Log key parameters
        logger.info(f"Training parameters: max_episodes={args.num_episodes}, min_episodes={args.min_episodes}, learning_rate={args.learning_rate}")
        logger.info(f"Using device: {args.device}")
            
        # Load and process data
        train_price_data, train_features, test_price_data, test_features = load_and_process_data(logger)
        
        # Create environments
        train_env = create_environment(train_price_data, train_features, args.hierarchical, logger)
        test_env = create_environment(test_price_data, test_features, args.hierarchical, logger)
        
        # Create model - choose between regular and hierarchical
        if args.hierarchical:
            model = create_hierarchical_model(train_env, args.device, logger)
        else:
            model = create_model(train_env, args.hierarchical, args.device, logger)
        
        # Train model
        trainer, training_results = train_model(
            train_env, test_env, model, args, logger
        )
        
        logger.info("Training and evaluation completed successfully")
        
        # Display final evaluation results
        logger.info("\nFinal Evaluation Results:")
        test_metrics = training_results['test_metrics']
        logger.info(f"Sharpe Ratio: {test_metrics.get('sharpe_ratio', -float('inf')):.4f}")
        logger.info(f"Total Return: {test_metrics.get('total_return', 0.0):.4f}")
        logger.info(f"Annualized Return: {test_metrics.get('annualized_return', 0.0):.4f}")
        logger.info(f"Max Drawdown: {test_metrics.get('max_drawdown', 1.0):.4f}")
        logger.info(f"Calmar Ratio: {test_metrics.get('calmar_ratio', -float('inf')):.4f}")
        logger.info(f"Win Rate: {test_metrics.get('win_rate', 0.0):.4f}")
        
        # Print instructions for running the backtest
        logger.info("\nTo run a backtest on this model, use the following command:")
        logger.info(f"python run_backtest.py --model_path models/model_{args.run_id}.pt {'--hierarchical' if args.hierarchical else ''}")
        
    except Exception as e:
        logger.exception(f"Error during training: {str(e)}")
        raise
    
if __name__ == '__main__':
    main()