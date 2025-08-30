# NeatRL Quick Start Guide

## 🚀 Getting Started with NeatRL

### Prerequisites
- Docker and Docker Compose v2
- Git

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd RL\ Leaderboard
```

### 2. Environment Configuration
Create a `.env` file:
```env
# FastAPI app security
SECRET_KEY=your-secret-key

# Supabase configuration
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_ANON_KEY=your-public-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_BUCKET=submissions

# Database connection
DATABASE_URL=postgresql://postgres:<encoded_password>@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require&options=project%3D<project-ref>

# Redis configuration
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/1
```

### 3. Build and Start
```bash
# Build the evaluator image
docker build -f docker/Dockerfile.evaluator -t rl-evaluator:latest .

# Start the entire stack
docker compose up -d --build
```

### 4. Access the Platform
- **Gradio Frontend**: http://localhost:7860
- **API Documentation**: http://localhost:8000/docs
- **Arena System**: http://localhost:8765 (WebSocket)

---

## 🏠 RL Hub - Upload Your Models

### Step 1: Prepare Your Model
Your model should be a Python file that:
- Takes environment ID as command line argument
- Outputs JSON with `score` field
- Uses only the libraries provided in the evaluator

Example model structure:
```python
import sys
import json
import gymnasium as gym

def main():
    env_id = sys.argv[1]
    env = gym.make(env_id)
    
    # Your RL algorithm here
    total_reward = 0
    for episode in range(10):
        state, _ = env.reset()
        episode_reward = 0
        
        for step in range(500):
            action = env.action_space.sample()  # Your policy here
            state, reward, done, truncated, _ = env.step(action)
            episode_reward += reward
            
            if done or truncated:
                break
        
        total_reward += episode_reward
    
    env.close()
    
    # Output final result
    print(json.dumps({
        "score": total_reward / 10,
        "metrics": [total_reward / 10]
    }))

if __name__ == "__main__":
    main()
```

### Step 2: Upload to RL Hub
1. Go to http://localhost:7860
2. Click on the "🏠 RL Hub" tab
3. Fill in the model details:
   - **Model Name**: Give your model a descriptive name
   - **Description**: Explain what your model does
   - **Algorithm**: Specify the RL algorithm used (e.g., DQN, PPO, A2C)
   - **Environment**: Select the environment your model was trained on
   - **User ID**: Your username or identifier
4. Upload your Python file
5. Click "🚀 Upload to RL Hub"

### Step 3: Browse Models
- Use the "Browse Models" section to discover other models
- Filter by environment, algorithm, or creator
- View model statistics and performance

---

## ⚔️ RL Arena - Battle Your Models

### Step 1: Create a Battle
1. Go to the "⚔️ RL Arena" tab
2. Enter your model ID (from RL Hub)
3. Optionally specify an opponent model ID (or let the system choose randomly)
4. Select the battle environment
5. Click "⚔️ Start Battle"

### Step 2: Monitor Battle Progress
- Watch real-time battle updates
- Track scores and progress
- View battle statistics

### Step 3: View Results
- See detailed battle results
- Check winner and final scores
- View battle duration and statistics

### Step 4: Check Rankings
- View arena rankings
- See win/loss statistics
- Track model performance over time

---

## 🎮 Interactive Features

### Keyboard Control
For human vs AI battles:
- **CartPole-v1**: Left/Right arrows
- **LunarLander-v2**: Arrow keys for engine control
- **Acrobot-v1**: Left/Right arrows for torque
- **MountainCar-v0**: Arrow keys for movement

### Real-time Visualization
- Watch battles as they happen
- See agent actions and environment state
- Monitor scores in real-time

### Informative End Screens
- Detailed battle results
- Opponent information
- Model performance statistics
- Battle duration and metrics

---

## 📊 Traditional Leaderboard

The original leaderboard functionality is still available:
1. Go to the "📊 Leaderboard" tab
2. Submit your agent for evaluation
3. Check status and view results
4. Compare with other submissions

---

## 🔧 Development

### Testing the System
```bash
# Run the test script
python test_neatrl.py
```

### API Endpoints
- **RL Hub**: `/api/rlhub/*`
- **RL Arena**: `/api/rlarena/*`
- **Traditional**: `/api/submit/*`, `/api/leaderboard/*`

### WebSocket Communication
The arena system uses WebSocket connections for real-time updates:
- Connect to: `ws://localhost:8765/{battle_id}`
- Message format: JSON

---

## 🐛 Troubleshooting

### Common Issues
1. **Model upload fails**: Check file format and size
2. **Battle doesn't start**: Verify model IDs and environment compatibility
3. **WebSocket connection fails**: Check if arena service is running
4. **Docker issues**: Ensure Docker is running and ports are available

### Logs
```bash
# View all logs
docker compose logs

# View specific service logs
docker compose logs arena
docker compose logs api
docker compose logs frontend
```

---

## 🚀 Next Steps

1. **Upload your first model** to the RL Hub
2. **Create a battle** in the RL Arena
3. **Explore other models** and challenge them
4. **Improve your models** based on battle results
5. **Share your models** with the community

Happy battling! ⚔️


