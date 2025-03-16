"""
PPO (Proximal Policy Optimization) trainer for reinforcement learning.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added missing import
import torch.optim as optim
import numpy as np
import logging
from collections import deque

# Configure logging
logger = logging.getLogger(__name__)

class PPOTrainer:
    """
    Proximal Policy Optimization trainer for the trading agent.
    """
    
    def __init__(self, env, model, device=None, lr=3e-4, gamma=0.99, lambda_gae=0.95, 
                 clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        """
        Initialize PPO trainer.
        
        Args:
            env: Trading environment
            model: Actor-critic model
            device: Torch device (cpu or cuda)
            lr (float): Learning rate
            gamma (float): Discount factor
            lambda_gae (float): GAE lambda parameter
            clip_epsilon (float): PPO clip parameter
            entropy_coef (float): Entropy coefficient
            value_coef (float): Value loss coefficient
            max_grad_norm (float): Maximum gradient norm
        """
        self.env = env
        self.model = model
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Move model to device
        self.model.to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        
        self.gamma = gamma
        self.lambda_gae = lambda_gae
        self.clip_epsilon = clip_epsilon
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        
        logger.info(f"Initialized PPOTrainer with lr={lr}, gamma={gamma}, "
                   f"lambda_gae={lambda_gae}, clip_epsilon={clip_epsilon}")
        logger.info(f"Using device: {self.device}")
        
    def compute_gae(self, rewards, values, next_value, dones):
        """
        Compute Generalized Advantage Estimation.
        
        Args:
            rewards (list): List of rewards
            values (list): List of state values
            next_value (float): Next state value
            dones (list): List of done flags
            
        Returns:
            tuple: (returns, advantages)
        """
        gae = 0
        returns = []
        advantages = []
        
        # Append next_value to values for convenience
        values = values + [next_value]
        
        # Compute returns and advantages in reverse order
        for i in reversed(range(len(rewards))):
            # Calculate TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = rewards[i] + self.gamma * values[i+1] * (1 - dones[i]) - values[i]
            
            # Calculate GAE
            gae = delta + self.gamma * self.lambda_gae * (1 - dones[i]) * gae
            
            # Prepend to maintain correct order
            returns.insert(0, gae + values[i])
            advantages.insert(0, gae)
            
        return returns, advantages
    
    def update(self, observations, actions, old_log_probs, returns, advantages, update_epochs=4, mini_batch_size=64):
        """
        Update policy and value function.
        
        Args:
            observations (torch.Tensor): Observations
            actions (torch.Tensor): Actions
            old_log_probs (torch.Tensor): Log probabilities of actions
            returns (torch.Tensor): Discounted returns
            advantages (torch.Tensor): Advantages
            update_epochs (int): Number of update epochs
            mini_batch_size (int): Mini-batch size
            
        Returns:
            dict: Dictionary of losses
        """
        # Move data to device
        observations = observations.to(self.device)
        actions = actions.to(self.device)
        old_log_probs = old_log_probs.to(self.device)
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get dataset size
        dataset_size = observations.size(0)
        
        # Track losses
        policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        
        for _ in range(update_epochs):
            # Generate random permutation of indices
            indices = torch.randperm(dataset_size).to(self.device)
            
            # Update in mini-batches
            for start_idx in range(0, dataset_size, mini_batch_size):
                # Get mini-batch indices
                end_idx = min(start_idx + mini_batch_size, dataset_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_observations = observations[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Evaluate actions
                new_log_probs, entropy, values = self.model.evaluate_action(mb_observations, mb_actions)
                
                # Calculate policy loss (PPO clip objective)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Calculate value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Calculate entropy loss
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Perform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Store losses
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
        
        # Return average losses
        return {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses)
        }
    
    def collect_trajectory(self, max_steps):
        """
        Collect a trajectory of experience.
        
        Args:
            max_steps (int): Maximum number of steps to collect
            
        Returns:
            tuple: (observations, actions, rewards, values, log_probs, dones, final_value)
        """
        observations = []
        actions = []
        rewards = []
        values = []
        log_probs = []
        dones = []
        
        # Reset environment
        obs = self.env.reset()
        
        for _ in range(max_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get action, log probability, and value
            with torch.no_grad():
                action, log_prob, _ = self.model.get_action(obs_tensor)
                _, _, value = self.model(obs_tensor)
            
            # Take action in environment
            next_obs, reward, done, _ = self.env.step(action.squeeze().cpu().numpy())
            
            # Store data
            observations.append(obs)
            actions.append(action.squeeze().cpu().numpy())
            rewards.append(reward)
            values.append(value.item())
            log_probs.append(log_prob.item())
            dones.append(done)
            
            obs = next_obs
            
            if done:
                break
        
        # Get value of final state
        if done:
            final_value = 0.0
        else:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get value of final state
            with torch.no_grad():
                _, _, final_value = self.model(obs_tensor)
                final_value = final_value.item()
        
        return observations, actions, rewards, values, log_probs, dones, final_value
    
    def train(self, num_episodes, max_steps, update_interval=2048, verbose=True):
        """
        Train the model.
        
        Args:
            num_episodes (int): Number of episodes to train for
            max_steps (int): Maximum number of steps per episode
            update_interval (int): Number of steps between updates
            verbose (bool): Whether to print progress
            
        Returns:
            tuple: (episode_rewards, episode_lengths, losses)
        """
        # Track progress
        episode_rewards = []
        episode_lengths = []
        losses = []
        
        # Training loop
        total_steps = 0
        episode = 0
        
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
                
                # Log progress
                if verbose and episode % 10 == 0:
                    recent_rewards = episode_rewards[-10:] if len(episode_rewards) >= 10 else episode_rewards
                    avg_reward = sum(recent_rewards) / len(recent_rewards)
                    logger.info(f"Episode {episode}/{num_episodes}, Total Steps: {total_steps}, "
                               f"Episode Reward: {episode_reward:.4f}, Avg Reward: {avg_reward:.4f}, "
                               f"Episode Length: {episode_length}")
            
            # Skip update if we have no data
            if not observations_buffer:
                continue
            
            # Compute returns and advantages
            returns, advantages = self.compute_gae(
                rewards_buffer, values_buffer, final_value, dones_buffer
            )
            
            # Convert to tensors
            observations_tensor = torch.FloatTensor(observations_buffer)
            actions_tensor = torch.FloatTensor(actions_buffer)
            log_probs_tensor = torch.FloatTensor(log_probs_buffer).unsqueeze(1)
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
                logger.info(f"Update at episode {episode}, Total Steps: {total_steps}, "
                           f"Policy Loss: {loss_dict['policy_loss']:.4f}, "
                           f"Value Loss: {loss_dict['value_loss']:.4f}, "
                           f"Entropy Loss: {loss_dict['entropy_loss']:.4f}")
        
        return episode_rewards, episode_lengths, losses

    def save_model(self, path):
        """
        Save model to disk.
        
        Args:
            path (str): Path to save model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        """
        Load model from disk.
        
        Args:
            path (str): Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        logger.info(f"Model loaded from {path}")

class HierarchicalPPOTrainer(PPOTrainer):
    """
    PPO trainer for hierarchical actor-critic models.
    """
    
    def __init__(self, env, model, device=None, lr=3e-4, gamma=0.99, lambda_gae=0.95, 
                 clip_epsilon=0.2, entropy_coef=0.01, value_coef=0.5, max_grad_norm=0.5):
        """
        Initialize hierarchical PPO trainer.
        
        Args:
            env: Trading environment with hierarchical action space
            model: Hierarchical actor-critic model
            device: Torch device (cpu or cuda)
            lr (float): Learning rate
            gamma (float): Discount factor
            lambda_gae (float): GAE lambda parameter
            clip_epsilon (float): PPO clip parameter
            entropy_coef (float): Entropy coefficient
            value_coef (float): Value loss coefficient
            max_grad_norm (float): Maximum gradient norm
        """
        super(HierarchicalPPOTrainer, self).__init__(
            env, model, device, lr, gamma, lambda_gae, 
            clip_epsilon, entropy_coef, value_coef, max_grad_norm
        )
        
        # Ensure model and environment have compatible hierarchical structure
        if not hasattr(self.env, 'asset_classes') or not hasattr(self.model, 'asset_classes'):
            raise ValueError("Environment and model must have hierarchical structure")
        
        logger.info("Initialized HierarchicalPPOTrainer")
    
    def collect_trajectory(self, max_steps):
        """
        Collect a trajectory of experience with hierarchical actions.
        
        Args:
            max_steps (int): Maximum number of steps to collect
            
        Returns:
            tuple: (observations, actions, rewards, values, log_probs, dones, final_value)
        """
        observations = []
        actions = {
            'strategic': [],
            'tactical': {class_name: [] for class_name in self.model.asset_classes}
        }
        log_probs = {
            'strategic': [],
            'tactical': {class_name: [] for class_name in self.model.asset_classes}
        }
        rewards = []
        values = []
        dones = []
        
        # Reset environment
        obs = self.env.reset()
        
        for _ in range(max_steps):
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get hierarchical action, log probability, and value
            with torch.no_grad():
                action_dict, log_prob_dict, _ = self.model.get_action(obs_tensor)
                _, _, _, _, value = self.model(obs_tensor)
            
            # Convert actions to numpy
            action_np = {
                'strategic': action_dict['strategic'].squeeze().cpu().numpy(),
                'tactical': {
                    class_name: action_dict['tactical'][class_name].squeeze().cpu().numpy()
                    for class_name in self.model.asset_classes
                }
            }
            
            # Take action in environment
            next_obs, reward, done, _ = self.env.step(action_np)
            
            # Store data
            observations.append(obs)
            actions['strategic'].append(action_np['strategic'])
            for class_name in self.model.asset_classes:
                actions['tactical'][class_name].append(action_np['tactical'][class_name])
            
            log_probs['strategic'].append(log_prob_dict['strategic'].item())
            for class_name in self.model.asset_classes:
                log_probs['tactical'][class_name].append(log_prob_dict['tactical'][class_name].item())
            
            rewards.append(reward)
            values.append(value.item())
            dones.append(done)
            
            obs = next_obs
            
            if done:
                break
        
        # Get value of final state
        if done:
            final_value = 0.0
        else:
            # Convert observation to tensor
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            # Get value of final state
            with torch.no_grad():
                _, _, _, _, final_value = self.model(obs_tensor)
                final_value = final_value.item()
        
        return observations, actions, rewards, values, log_probs, dones, final_value
    
    def update(self, observations, actions, old_log_probs, returns, advantages, update_epochs=4, mini_batch_size=64):
        """
        Update policy and value function for hierarchical model.
        
        Args:
            observations (torch.Tensor): Observations
            actions (dict): Hierarchical actions
            old_log_probs (dict): Hierarchical log probabilities
            returns (torch.Tensor): Discounted returns
            advantages (torch.Tensor): Advantages
            update_epochs (int): Number of update epochs
            mini_batch_size (int): Mini-batch size
            
        Returns:
            dict: Dictionary of losses
        """
        # Move data to device
        observations = observations.to(self.device)
        actions_strategic = actions['strategic'].to(self.device)
        old_log_probs_strategic = old_log_probs['strategic'].to(self.device)
        
        actions_tactical = {
            class_name: actions['tactical'][class_name].to(self.device)
            for class_name in self.model.asset_classes
        }
        
        old_log_probs_tactical = {
            class_name: old_log_probs['tactical'][class_name].to(self.device)
            for class_name in self.model.asset_classes
        }
        
        returns = returns.to(self.device)
        advantages = advantages.to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Get dataset size
        dataset_size = observations.size(0)
        
        # Track losses
        strategic_policy_losses = []
        tactical_policy_losses = []
        value_losses = []
        entropy_losses = []
        total_losses = []
        
        for _ in range(update_epochs):
            # Generate random permutation of indices
            indices = torch.randperm(dataset_size).to(self.device)
            
            # Update in mini-batches
            for start_idx in range(0, dataset_size, mini_batch_size):
                # Get mini-batch indices
                end_idx = min(start_idx + mini_batch_size, dataset_size)
                mb_indices = indices[start_idx:end_idx]
                
                # Get mini-batch data
                mb_observations = observations[mb_indices]
                mb_actions_strategic = actions_strategic[mb_indices]
                mb_old_log_probs_strategic = old_log_probs_strategic[mb_indices]
                
                mb_actions_tactical = {
                    class_name: actions_tactical[class_name][mb_indices]
                    for class_name in self.model.asset_classes
                }
                
                mb_old_log_probs_tactical = {
                    class_name: old_log_probs_tactical[class_name][mb_indices]
                    for class_name in self.model.asset_classes
                }
                
                mb_returns = returns[mb_indices]
                mb_advantages = advantages[mb_indices]
                
                # Evaluate hierarchical actions
                mb_actions = {
                    'strategic': mb_actions_strategic,
                    'tactical': mb_actions_tactical
                }
                
                new_log_probs, entropy, values = self.model.evaluate_action(mb_observations, mb_actions)
                
                # Calculate strategic policy loss
                ratio_strategic = torch.exp(new_log_probs['strategic'] - mb_old_log_probs_strategic)
                surr1_strategic = ratio_strategic * mb_advantages
                surr2_strategic = torch.clamp(ratio_strategic, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                strategic_policy_loss = -torch.min(surr1_strategic, surr2_strategic).mean()
                
                # Calculate tactical policy losses
                tactical_policy_loss = 0
                for class_name in self.model.asset_classes:
                    ratio_tactical = torch.exp(new_log_probs['tactical'][class_name] - mb_old_log_probs_tactical[class_name])
                    surr1_tactical = ratio_tactical * mb_advantages
                    surr2_tactical = torch.clamp(ratio_tactical, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * mb_advantages
                    tactical_policy_loss += -torch.min(surr1_tactical, surr2_tactical).mean()
                
                # Average tactical policy loss
                tactical_policy_loss /= len(self.model.asset_classes)
                
                # Calculate value loss
                value_loss = F.mse_loss(values, mb_returns)
                
                # Calculate entropy loss (average of strategic and tactical entropies)
                entropy_loss = -entropy['strategic'].mean()
                for class_name in self.model.asset_classes:
                    entropy_loss -= entropy['tactical'][class_name].mean()
                entropy_loss /= (1 + len(self.model.asset_classes))  # Average
                
                # Total loss
                policy_loss = strategic_policy_loss + tactical_policy_loss
                loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Perform optimization step
                self.optimizer.zero_grad()
                loss.backward()
                
                # Clip gradients
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                
                self.optimizer.step()
                
                # Store losses
                strategic_policy_losses.append(strategic_policy_loss.item())
                tactical_policy_losses.append(tactical_policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())
                total_losses.append(loss.item())
        
        # Return average losses
        return {
            'strategic_policy_loss': np.mean(strategic_policy_losses),
            'tactical_policy_loss': np.mean(tactical_policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'total_loss': np.mean(total_losses)
        }