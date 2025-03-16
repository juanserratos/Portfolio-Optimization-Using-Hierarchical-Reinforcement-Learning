"""
Trading environment that implements the OpenAI Gym interface.
"""
import numpy as np
import gym
from gym import spaces
import pandas as pd
import logging

# Configure logging
logger = logging.getLogger(__name__)

class TradingEnvironment(gym.Env):
    """Custom trading environment that implements gym interface"""
    
    def __init__(self, price_data, feature_data, window_size=60, transaction_cost=0.001, 
                 max_position=1.0, reward_scaling=1.0):
        """
        Initialize the trading environment.
        
        Args:
            price_data (pd.DataFrame): DataFrame with asset prices
            feature_data (pd.DataFrame): DataFrame with features
            window_size (int): Look-back window size for features
            transaction_cost (float): Cost per trade as a fraction
            max_position (float): Maximum position size as a fraction of capital
            reward_scaling (float): Scaling factor for rewards
        """
        super(TradingEnvironment, self).__init__()
        
        self.price_data = price_data
        self.feature_data = feature_data
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.max_position = max_position
        self.reward_scaling = reward_scaling
        
        self.num_assets = len(price_data.columns)
        
        # Action space: portfolio weights for each asset (continuous between -1 and 1)
        # -1 means maximum short position, 1 means maximum long position
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.num_assets,), dtype=np.float32
        )
        
        # Observation space: historical features and current portfolio weights
        # Calculate observation dimension based on features and window size
        feature_dim = self.feature_data.shape[1]
        obs_dim = window_size * feature_dim + self.num_assets  # Features + current weights
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Initialize state
        self.reset()
        
        logger.info(f"Initialized TradingEnvironment with {self.num_assets} assets and " 
                    f"observation space of dimension {obs_dim}")
        
    def reset(self):
        """
        Reset the environment to the initial state.
        
        Returns:
            np.array: Initial observation
        """
        self.current_step = self.window_size
        self.portfolio_value = 1.0
        self.current_weights = np.zeros(self.num_assets)
        self.previous_weights = np.zeros(self.num_assets)
        self.returns_history = []
        
        logger.debug("Environment reset")
        return self._get_observation()
    
    def _get_observation(self):
        """
        Construct the observation vector.
        
        Returns:
            np.array: Observation vector
        """
        # Get historical window of features
        features = self.feature_data.iloc[self.current_step - self.window_size:self.current_step].values
        
        # Flatten and combine with current portfolio weights
        obs = np.concatenate([features.flatten(), self.current_weights])
        return obs
    
    def _calculate_returns(self, weights):
        """
        Calculate portfolio returns based on weights.
        
        Args:
            weights (np.array): Portfolio weights
            
        Returns:
            tuple: (portfolio_return, asset_returns, transaction_costs)
        """
        # Ensure we're not past the end of the data
        if self.current_step >= len(self.price_data) - 1:
            return 0, np.zeros(self.num_assets), 0
        
        # Get current and next day prices
        current_prices = self.price_data.iloc[self.current_step].values
        next_prices = self.price_data.iloc[self.current_step + 1].values
        
        # Calculate asset returns
        asset_returns = next_prices / current_prices - 1
        
        # Calculate transaction costs
        weight_changes = np.abs(weights - self.previous_weights)
        transaction_costs = np.sum(weight_changes) * self.transaction_cost
        
        # Calculate portfolio return (weighted sum of asset returns minus transaction costs)
        portfolio_return = np.sum(weights * asset_returns) - transaction_costs
        
        return portfolio_return, asset_returns, transaction_costs
    
    def step(self, action):
        """
        Take an action in the environment.
        
        Args:
            action (np.array): Action vector representing portfolio weights
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Check if episode is done
        if self.current_step >= len(self.price_data) - 1:
            return self._get_observation(), 0, True, {'portfolio_value': self.portfolio_value}
        
        # Scale the action to the allowed position range
        scaled_action = np.clip(action, -1, 1) * self.max_position
        
        # Calculate returns based on weights
        portfolio_return, asset_returns, transaction_costs = self._calculate_returns(scaled_action)
        
        # Update portfolio value
        self.portfolio_value *= (1 + portfolio_return)
        
        # Store weights for next step
        self.previous_weights = self.current_weights
        self.current_weights = scaled_action
        
        # Store return for computing metrics
        self.returns_history.append(portfolio_return)
        
        # Move to next step
        self.current_step += 1
        
        # Calculate reward
        # Use Sharpe ratio component as reward to balance return and risk
        if len(self.returns_history) > 1:
            returns_array = np.array(self.returns_history[-20:])  # Use last 20 returns for stability
            reward = self.reward_scaling * (
                portfolio_return / (np.std(returns_array) + 1e-8)  # Sharpe-like ratio
            )
        else:
            reward = self.reward_scaling * portfolio_return
        
        # Check if episode is done
        done = self.current_step >= len(self.price_data) - 1
        
        # Additional info
        info = {
            'portfolio_value': self.portfolio_value,
            'portfolio_return': portfolio_return,
            'transaction_costs': transaction_costs,
            'asset_returns': asset_returns
        }
        
        return self._get_observation(), reward, done, info
    
    def render(self, mode='human'):
        """
        Render the environment.
        
        Args:
            mode (str): Rendering mode
        """
        if mode == 'human':
            print(f"Step: {self.current_step}, Portfolio Value: {self.portfolio_value:.4f}")
            if len(self.returns_history) > 0:
                print(f"Last Return: {self.returns_history[-1]:.4f}")
            print("Current Weights:", self.current_weights)
            print()

    def seed(self, seed=None):
        """
        Set the random seed.
        
        Args:
            seed (int): Random seed
            
        Returns:
            list: List containing the seed
        """
        if seed is not None:
            np.random.seed(seed)
        return [seed]

    def get_portfolio_history(self):
        """
        Get the history of portfolio values.
        
        Returns:
            list: History of portfolio values
        """
        portfolio_values = [1.0]  # Start with initial value
        for ret in self.returns_history:
            portfolio_values.append(portfolio_values[-1] * (1 + ret))
        return portfolio_values

class HierarchicalTradingEnvironment(TradingEnvironment):
    """
    Trading environment with hierarchical action space.
    
    This environment separates strategic allocation (asset classes) from
    tactical execution (individual assets within each class).
    """
    
    def __init__(self, price_data, feature_data, asset_classes, window_size=60, 
                 transaction_cost=0.001, max_position=1.0, reward_scaling=1.0):
        """
        Initialize the hierarchical trading environment.
        
        Args:
            price_data (pd.DataFrame): DataFrame with asset prices
            feature_data (pd.DataFrame): DataFrame with features
            asset_classes (dict): Dictionary mapping asset class names to lists of asset indices
            window_size (int): Look-back window size for features
            transaction_cost (float): Cost per trade as a fraction
            max_position (float): Maximum position size as a fraction of capital
            reward_scaling (float): Scaling factor for rewards
        """
        super(HierarchicalTradingEnvironment, self).__init__(
            price_data, feature_data, window_size, transaction_cost, max_position, reward_scaling
        )
        
        self.asset_classes = asset_classes
        self.num_classes = len(asset_classes)
        
        # Define action space for both levels
        # Level 1: Asset class weights (-1 to 1 for each class)
        # Level 2: Asset weights within each class (-1 to 1 for each asset)
        self.action_space = spaces.Dict({
            'strategic': spaces.Box(low=-1, high=1, shape=(self.num_classes,), dtype=np.float32),
            'tactical': spaces.Dict({
                class_name: spaces.Box(
                    low=-1, high=1, shape=(len(asset_indices),), dtype=np.float32
                ) for class_name, asset_indices in asset_classes.items()
            })
        })
        
        logger.info(f"Initialized HierarchicalTradingEnvironment with {self.num_classes} "
                    f"asset classes and {self.num_assets} total assets")
    
    def step(self, action):
        """
        Take a hierarchical action in the environment.
        
        Args:
            action (dict): Dictionary with 'strategic' and 'tactical' actions
            
        Returns:
            tuple: (observation, reward, done, info)
        """
        # Extract strategic and tactical actions
        strategic_action = action['strategic']
        tactical_action = action['tactical']
        
        # Normalize strategic weights to sum to 1
        strategic_weights = np.clip(strategic_action, -1, 1) * self.max_position
        
        # Compute final asset weights based on hierarchical allocation
        final_weights = np.zeros(self.num_assets)
        
        for i, (class_name, asset_indices) in enumerate(self.asset_classes.items()):
            # Get strategic weight for this asset class
            class_weight = strategic_weights[i]
            
            # Get tactical weights for assets in this class
            assets_tactical_weights = np.clip(tactical_action[class_name], -1, 1)
            
            # Normalize tactical weights within the class
            if np.sum(np.abs(assets_tactical_weights)) > 0:
                assets_tactical_weights = assets_tactical_weights / np.sum(np.abs(assets_tactical_weights))
            
            # Apply strategic weight to tactical weights
            for j, asset_idx in enumerate(asset_indices):
                final_weights[asset_idx] = class_weight * assets_tactical_weights[j]
        
        # Call the parent class step method with the computed weights
        return super().step(final_weights)