#!/usr/bin/env python3
"""
Example DQN Model for NeatRL Arena
This model can be uploaded to the RL Hub and used in battles
"""

import sys
import json
import numpy as np
import gymnasium as gym
from collections import deque
import random

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Simple neural network for DQN"""
        # For simplicity, we'll use a basic Q-table approach
        # In a real implementation, you'd use a neural network
        return {}

    def update_target_model(self):
        """Update target model"""
        pass

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory"""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        # Simple Q-learning approach
        if isinstance(state, np.ndarray):
            state = state.tolist()
        
        # For simplicity, use a basic heuristic
        if len(state) >= 4:  # CartPole-like environment
            cart_pos, cart_vel, pole_ang, pole_vel = state[:4]
            
            # Simple heuristic: move cart in direction of pole
            if pole_ang > 0:
                return 1  # Move right
            else:
                return 0  # Move left
        
        return random.randrange(self.action_size)

    def replay(self, batch_size):
        """Train the model"""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.act(next_state))
            
            # Update Q-value (simplified)
            pass
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """Load model weights"""
        pass

    def save(self, name):
        """Save model weights"""
        pass

def train_dqn(env_id, episodes=100):
    """Train a DQN agent"""
    env = gym.make(env_id)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done, truncated, _ = env.step(action)
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if done or truncated:
                break
        
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
        
        episode_rewards.append(total_reward)
        
        if episode % 10 == 0:
            avg_reward = np.mean(episode_rewards[-10:])
            print(f"Episode: {episode}/{episodes}, Average Reward: {avg_reward:.2f}")
    
    env.close()
    return agent, episode_rewards

def evaluate_model(env_id, agent, episodes=10):
    """Evaluate the trained model"""
    env = gym.make(env_id)
    episode_rewards = []
    
    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        
        for time in range(500):
            action = agent.act(state)
            state, reward, done, truncated, _ = env.step(action)
            total_reward += reward
            
            if done or truncated:
                break
        
        episode_rewards.append(total_reward)
    
    env.close()
    return episode_rewards

def main():
    """Main function for training and evaluation"""
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing environment ID"}))
        sys.exit(1)
    
    env_id = sys.argv[1]
    
    try:
        # Train the model
        print(f"Training DQN agent on {env_id}...")
        agent, training_rewards = train_dqn(env_id, episodes=50)
        
        # Evaluate the model
        print(f"Evaluating DQN agent on {env_id}...")
        eval_rewards = evaluate_model(env_id, agent, episodes=10)
        
        # Calculate final score
        final_score = np.mean(eval_rewards)
        
        # Print results in JSON format
        result = {
            "score": float(final_score),
            "metrics": [float(r) for r in eval_rewards],
            "episodes": len(eval_rewards),
            "model_type": "DQN",
            "environment": env_id
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    main()
