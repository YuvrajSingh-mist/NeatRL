# Agent Performance GIFs

This directory contains GIF recordings of trained RL agents performing in various environments.

## How to Add Agent GIFs

1. **Record your agent**: Use tools like `gymnasium` with rendering enabled to record agent performance
2. **Convert to GIF**: Use tools like `imageio` or `ffmpeg` to convert recordings to GIF format
3. **Naming convention**: Use format `{algorithm}_{environment}.gif`
   - Example: `dqn_cartpole.gif`, `ppo_lunar_lander.gif`
4. **Optimal size**: Keep GIFs under 5MB for fast loading
5. **Dimensions**: Recommended 400x400 or 600x400 pixels

## Current Agent GIFs Needed:

### DQN Algorithm:
- `dqn_cartpole.gif` - DQN agent playing CartPole-v1
- `dqn_lunar_lander.gif` - DQN agent landing on moon

### PPO Algorithm:
- `ppo_cartpole.gif` - PPO agent balancing pole
- `ppo_bipedal_walker.gif` - PPO agent walking

### A2C Algorithm:
- `a2c_mountain_car.gif` - A2C agent climbing mountain
- `a2c_acrobot.gif` - A2C agent swinging up

### SAC Algorithm:
- `sac_pendulum.gif` - SAC agent controlling pendulum
- `sac_half_cheetah.gif` - SAC agent running

## Example Recording Code:

```python
import gymnasium as gym
import imageio
import numpy as np

# Record agent performance
env = gym.make('CartPole-v1', render_mode='rgb_array')
frames = []

for episode in range(5):  # Record 5 episodes
    obs, _ = env.reset()
    done = False
    
    while not done:
        action = agent.predict(obs)  # Your trained agent
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        
        # Capture frame
        frame = env.render()
        frames.append(frame)

# Save as GIF
imageio.mimsave('dqn_cartpole.gif', frames, fps=30)
```

## Placeholder Status:
Currently using placeholder images. Replace with actual agent recordings when available.