"""
Actor-critic model for reinforcement learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

# Configure logging
logger = logging.getLogger(__name__)

class ActorCritic(nn.Module):
    """
    Actor-critic network for reinforcement learning.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=128, std_init=0.5, action_range=1.0):
        """
        Initialize actor-critic network.
        
        Args:
            state_dim (int): Dimension of state space
            action_dim (int): Dimension of action space
            hidden_dim (int): Hidden dimension
            std_init (float): Initial standard deviation for exploration
            action_range (float): Range of actions (-action_range to action_range)
        """
        super(ActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_range = action_range
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor head (policy)
        self.actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Outputs in [-1, 1] range for position sizing
        )
        
        # Initialize action standard deviation (for exploration)
        self.actor_log_std = nn.Parameter(torch.ones(action_dim) * np.log(std_init))
        
        # Critic head (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized ActorCritic with state_dim={state_dim}, action_dim={action_dim}, "
                   f"hidden_dim={hidden_dim}")
        
    def _init_weights(self, module):
        """
        Initialize weights of the model.
        
        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (Action mean, action log std, state value)
        """
        # Extract features
        shared_features = self.shared(state)
        
        # Actor (policy)
        action_mean = self.actor_mean(shared_features) * self.action_range
        action_log_std = self.actor_log_std.expand_as(action_mean)
        
        # Critic (value function)
        state_value = self.critic(shared_features)
        
        return action_mean, action_log_std, state_value
    
    def get_action(self, state, deterministic=False):
        """
        Sample action from policy.
        
        Args:
            state (torch.Tensor): State tensor
            deterministic (bool): If True, return mean action
            
        Returns:
            tuple: (Action, log probability, entropy)
        """
        action_mean, action_log_std, _ = self.forward(state)
        
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
        action = torch.clamp(action, -self.action_range, self.action_range)
        
        return action, log_prob, entropy
    
    def evaluate_action(self, state, action):
        """
        Evaluate action (used during training).
        
        Args:
            state (torch.Tensor): State tensor
            action (torch.Tensor): Action tensor
            
        Returns:
            tuple: (Log probability, entropy, state value)
        """
        action_mean, action_log_std, state_value = self.forward(state)
        
        # Calculate log probability and entropy using Gaussian distribution
        action_std = torch.exp(action_log_std)
        normal = torch.distributions.Normal(action_mean, action_std)
        
        log_prob = normal.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = normal.entropy().sum(dim=-1, keepdim=True)
        
        return log_prob, entropy, state_value

class HierarchicalActorCritic(nn.Module):
    """
    Hierarchical actor-critic network for reinforcement learning with hierarchical action space.
    """
    def __init__(self, state_dim, asset_classes, hidden_dim=128, std_init=0.5, action_range=1.0):
        """
        Initialize hierarchical actor-critic network.
        
        Args:
            state_dim (int): Dimension of state space
            asset_classes (dict): Dictionary mapping asset class names to lists of asset indices
            hidden_dim (int): Hidden dimension
            std_init (float): Initial standard deviation for exploration
            action_range (float): Range of actions (-action_range to action_range)
        """
        super(HierarchicalActorCritic, self).__init__()
        
        self.state_dim = state_dim
        self.asset_classes = asset_classes
        self.num_classes = len(asset_classes)
        self.hidden_dim = hidden_dim
        self.action_range = action_range
        
        # Get total number of assets
        self.total_assets = sum(len(indices) for indices in asset_classes.values())
        
        # Shared feature extractor
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Strategic actor (asset class allocation)
        self.strategic_actor_mean = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.num_classes),
            nn.Tanh()  # Outputs in [-1, 1] range
        )
        
        self.strategic_actor_log_std = nn.Parameter(torch.ones(self.num_classes) * np.log(std_init))
        
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
            class_name: nn.Parameter(torch.ones(len(asset_indices)) * np.log(std_init))
            for class_name, asset_indices in asset_classes.items()
        })
        
        # Critic (value function)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
        logger.info(f"Initialized HierarchicalActorCritic with {self.num_classes} asset classes and "
                   f"{self.total_assets} total assets")
        
    def _init_weights(self, module):
        """
        Initialize weights of the model.
        
        Args:
            module (nn.Module): Module to initialize
        """
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state):
        """
        Forward pass.
        
        Args:
            state (torch.Tensor): State tensor
            
        Returns:
            tuple: (Strategic action mean, strategic action log std, 
                   tactical action means dict, tactical action log stds dict,
                   state value)
        """
        # Extract features
        shared_features = self.shared(state)
        
        # Strategic actor (policy for asset classes)
        strategic_mean = self.strategic_actor_mean(shared_features) * self.action_range
        strategic_log_std = self.strategic_actor_log_std.expand_as(strategic_mean)
        
        # Tactical actors (policies for assets within each class)
        tactical_means = {}
        tactical_log_stds = {}
        
        for class_name, asset_indices in self.asset_classes.items():
            tactical_means[class_name] = self.tactical_actors_mean[class_name](shared_features) * self.action_range
            tactical_log_stds[class_name] = self.tactical_actors_log_std[class_name].expand_as(tactical_means[class_name])
        
        # Critic (value function)
        state_value = self.critic(shared_features)
        
        return strategic_mean, strategic_log_std, tactical_means, tactical_log_stds, state_value
    
    def get_action(self, state, deterministic=False):
        """
        Sample hierarchical action from policy.
        
        Args:
            state (torch.Tensor): State tensor
            deterministic (bool): If True, return mean action
            
        Returns:
            tuple: (Action dict, log probability dict, entropy dict)
        """
        strategic_mean, strategic_log_std, tactical_means, tactical_log_stds, _ = self.forward(state)
        
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
        strategic_action = torch.clamp(strategic_action, -self.action_range, self.action_range)
        
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
            tactical_action = torch.clamp(tactical_action, -self.action_range, self.action_range)
            
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
    
    def evaluate_action(self, state, action):
        """
        Evaluate hierarchical action (used during training).
        
        Args:
            state (torch.Tensor): State tensor
            action (dict): Hierarchical action dictionary
            
        Returns:
            tuple: (Log probability dict, entropy dict, state value)
        """
        strategic_mean, strategic_log_std, tactical_means, tactical_log_stds, state_value = self.forward(state)
        
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
        
        return log_prob, entropy, state_value