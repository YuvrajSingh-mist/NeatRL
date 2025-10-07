---
title: "RLHub - Agent Performance Gallery"
layout: single
permalink: /rlhub/
author_profile: false
classes: wide
---

<div class="rlhub-container">
  <div class="rlhub-intro">
    <h1>🤖 RLHub - Agent Performance Gallery</h1>
    <p class="rlhub-description">
      Explore how different reinforcement learning algorithms perform across various environments. 
      Watch trained agents in action and see the diverse behaviors that emerge from different learning approaches.
    </p>
  </div>

  <div class="algorithm-grid">
    
    <!-- DQN Section -->
    <div class="algorithm-section">
      <h2 class="algorithm-title">🎯 Deep Q-Network (DQN)</h2>
      <div class="agent-gallery">
        <div class="agent-card">
          <div class="agent-gif">
            <img src="/images/agents/dqn_cartpole.gif" alt="DQN CartPole Agent" loading="lazy">
          </div>
          <div class="agent-info">
            <h3>CartPole-v1</h3>
            <div class="performance-stats">
              <span class="stat">📊 Avg Score: 498.2</span>
              <span class="stat">🏆 Max Score: 500</span>
              <span class="stat">⏱️ Episodes: 1,200</span>
            </div>
            <p class="agent-description">
              DQN agent mastering the classic CartPole environment with stable performance after training convergence.
            </p>
          </div>
        </div>
        
        <div class="agent-card">
          <div class="agent-gif">
            <img src="/images/agents/dqn_lunar_lander.gif" alt="DQN LunarLander Agent" loading="lazy">
          </div>
          <div class="agent-info">
            <h3>LunarLander-v2</h3>
            <div class="performance-stats">
              <span class="stat">📊 Avg Score: 245.3</span>
              <span class="stat">🏆 Best Landing: 289.4</span>
              <span class="stat">⏱️ Episodes: 2,500</span>
            </div>
            <p class="agent-description">
              Smooth lunar landings achieved through careful Q-value optimization and epsilon-greedy exploration.
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- PPO Section -->
    <div class="algorithm-section">
      <h2 class="algorithm-title">🚀 Proximal Policy Optimization (PPO)</h2>
      <div class="agent-gallery">
        <div class="agent-card">
          <div class="agent-gif">
            <img src="/images/agents/ppo_cartpole.gif" alt="PPO CartPole Agent" loading="lazy">
          </div>
          <div class="agent-info">
            <h3>CartPole-v1</h3>
            <div class="performance-stats">
              <span class="stat">📊 Avg Score: 495.8</span>
              <span class="stat">🏆 Max Score: 500</span>
              <span class="stat">⏱️ Episodes: 800</span>
            </div>
            <p class="agent-description">
              PPO's policy gradient approach showing robust and consistent balancing behavior.
            </p>
          </div>
        </div>
        
        <div class="agent-card">
          <div class="agent-gif">
            <img src="/images/agents/ppo_bipedal_walker.gif" alt="PPO BipedalWalker Agent" loading="lazy">
          </div>
          <div class="agent-info">
            <h3>BipedalWalker-v3</h3>
            <div class="performance-stats">
              <span class="stat">📊 Avg Score: 312.7</span>
              <span class="stat">🏆 Best Run: 340.2</span>
              <span class="stat">⏱️ Episodes: 3,000</span>
            </div>
            <p class="agent-description">
              Complex locomotion patterns learned through continuous action space optimization.
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- A2C Section -->
    <div class="algorithm-section">
      <h2 class="algorithm-title">🎭 Advantage Actor-Critic (A2C)</h2>
      <div class="agent-gallery">
        <div class="agent-card">
          <div class="agent-gif">
            <img src="/images/agents/a2c_mountain_car.gif" alt="A2C MountainCar Agent" loading="lazy">
          </div>
          <div class="agent-info">
            <h3>MountainCar-v0</h3>
            <div class="performance-stats">
              <span class="stat">📊 Avg Score: -95.2</span>
              <span class="stat">🏆 Best: -89</span>
              <span class="stat">⏱️ Episodes: 1,500</span>
            </div>
            <p class="agent-description">
              Strategic momentum building showcasing the actor-critic learning paradigm.
            </p>
          </div>
        </div>
        
        <div class="agent-card">
          <div class="agent-gif">
            <img src="/images/agents/a2c_acrobot.gif" alt="A2C Acrobot Agent" loading="lazy">
          </div>
          <div class="agent-info">
            <h3>Acrobot-v1</h3>
            <div class="performance-stats">
              <span class="stat">📊 Avg Score: -85.4</span>
              <span class="stat">🏆 Best: -73</span>
              <span class="stat">⏱️ Episodes: 2,000</span>
            </div>
            <p class="agent-description">
              Impressive swing-up behavior demonstrating policy and value function coordination.
            </p>
          </div>
        </div>
      </div>
    </div>

    <!-- SAC Section -->
    <div class="algorithm-section">
      <h2 class="algorithm-title">🌊 Soft Actor-Critic (SAC)</h2>
      <div class="agent-gallery">
        <div class="agent-card">
          <div class="agent-gif">
            <img src="/images/agents/sac_pendulum.gif" alt="SAC Pendulum Agent" loading="lazy">
          </div>
          <div class="agent-info">
            <h3>Pendulum-v1</h3>
            <div class="performance-stats">
              <span class="stat">📊 Avg Score: -145.2</span>
              <span class="stat">🏆 Best: -89.3</span>
              <span class="stat">⏱️ Episodes: 1,000</span>
            </div>
            <p class="agent-description">
              Smooth continuous control with entropy regularization for robust exploration.
            </p>
          </div>
        </div>
        
        <div class="agent-card">
          <div class="agent-gif">
            <img src="/images/agents/sac_half_cheetah.gif" alt="SAC HalfCheetah Agent" loading="lazy">
          </div>
          <div class="agent-info">
            <h3>HalfCheetah-v4</h3>
            <div class="performance-stats">
              <span class="stat">📊 Avg Score: 4,250.7</span>
              <span class="stat">🏆 Top Speed: 4,890.2</span>
              <span class="stat">⏱️ Episodes: 2,500</span>
            </div>
            <p class="agent-description">
              High-speed locomotion showcasing SAC's effectiveness in complex continuous control tasks.
            </p>
          </div>
        </div>
      </div>
    </div>

  </div>

  <div class="rlhub-footer">
    <div class="implementation-note">
      <h3>🔬 Implementation Details</h3>
      <p>
        All agents are trained using custom PyTorch implementations with hyperparameter tuning. 
        Each algorithm includes detailed logging, visualization tools, and reproducible training scripts.
      </p>
      <div class="cta-buttons">
        <a href="/rl/" class="btn btn--primary">Explore Implementations</a>
        <a href="/leaderboard/" class="btn btn--inverse">Test Your Agent</a>
      </div>
    </div>
  </div>
</div>