# NeatRL

A comprehensive Reinforcement Learning platform featuring a model hub for sharing trained agents. Built with FastAPI, Celery, Redis, and PostgreSQL, providing a production-ready environment for RL enthusiasts to upload, evaluate, and compare their trained models.

## 🚀 New Features

### 🏠 Leaderboard
- **Model Upload**: Upload your trained RL models using standardized libraries
- **Model Sharing**: Share models publicly or keep them private
- **Model Discovery**: Browse and discover models from other users
- **Model Management**: Delete and manage your uploaded models

### 🎮 Interactive Features
- **Model Evaluation**: Comprehensive testing of model performance
- **Real-time Updates**: WebSocket-based real-time evaluation updates
- **Ranking System**: Track model performance with detailed statistics
- **Low Latency**: Optimized for minimal communication delays
- **Docker Environment**: Evaluations run in isolated Docker containers

---

## Table of Contents
- [Features](#features)
- [Architecture](#architecture)
- [Quickstart](#quickstart)
- [Environment Variables](#environment-variables)
- [Usage](#usage)
  - [Leaderboard - Model Upload](#rl-hub---model-upload)
  - [Submitting via API](#submitting-via-api)
  - [Checking Results](#checking-results)
  - [Querying the Leaderboard](#querying-the-leaderboard)
- [Submission Contract](#submission-contract)
- [Project Structure](#project-structure)
- [Local Development (without Docker)](#local-development-without-docker)
- [Troubleshooting](#troubleshooting)
- [Contributing](#contributing)
- [License](#license)

---

## Features
- **Leaderboard**: Upload and share your trained RL models with standardized library support
- **RL Evaluation**: Competitive multiplayer evaluations between user-uploaded models
- **Real-time leaderboard**: Redis-sorted sets with DB durability fallback
- **Asynchronous evaluation**: Celery worker executes user submissions in an isolated Docker container
- **Safe execution**: Containers run with no network, memory/CPU/pids limits, and capability drops
- **Persistent storage**: PostgreSQL for submissions and durable leaderboard entries
- **Object storage**: Supabase Storage for user-submitted scripts and models
- **Web Interface**: Simple web app to submit agents, check status, view leaderboards, and evaluation in the evaluation
- **Dockerized**: One command to bring up the full stack

---


## Project Structure

```
app/
  api/                 # FastAPI routers (submissions, leaderboard)
  core/                # Config, Celery, Docker client, Supabase client
  db/                  # SQLAlchemy engine/session and Base
  models/              # SQLAlchemy models (Submission, EvaluationMetric, LeaderboardEntry)
  services/            # Leaderboard (Redis) and evaluation orchestration
  main.py              # FastAPI app factory and startup hooks
frontend/              # Gradio web app
docker/                # Evaluator Dockerfile
scripts/entrypoint.sh  # Evaluator container entrypoint
example_agents/        # Sample agents (e.g., q_learning.py)
docker-compose.yml     # Orchestrates API, Worker, DB, Redis, Frontend
```

---

## Architecture
```
┌─────────────────┐      ┌─────────────────┐      ┌────────────────────┐
│  Gradio Frontend│ <--> │   API (FastAPI) │ <--> │  Celery Worker      │
└─────────────────┘      └─────────────────┘      └────────────────────┘
          │                       │                          │
          v                       v                          v
┌─────────────────┐      ┌─────────────────┐      ┌────────────────────┐
│ PostgreSQL (DB) │      │  Redis (cache)  │      │ Docker Engine (host│
└─────────────────┘      └─────────────────┘      └────────────────────┘
                                                (runs evaluator containers)
```

- The API exposes submission, results, and leaderboard endpoints.
- Submissions are uploaded to Supabase Storage, recorded in PostgreSQL, and queued via Celery.
- The worker pulls the script, runs it inside the `rl-evaluator:latest` image with strict limits, parses the JSON result, updates DB and Redis.
- Leaderboards are served from Redis for speed with an automatic fallback to DB for durability.

---

## Quickstart

### Prerequisites
- Docker and Docker Compose v2
- Git

### Clone
```bash
git clone <your-repo-url>
cd RL\ Leaderboard
```

### Environment
Create a `.env` file at the repo root (values are examples; use your own secrets):
```env
# SEO Configuration
PUBLIC_BASE_URL=https://rl-eval-leaderboard.onrender.com

# FastAPI app security
SECRET_KEY=please-change-this

# Supabase (required for uploads/downloads and DB)
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_ANON_KEY=your-public-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_BUCKET=submissions

# Supabase Postgres (use Connection Pooling host and encoded password)
# Example with pooling (replace region and project-ref):
# DATABASE_URL=postgresql://postgres:<encoded_password>@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require&options=project%3D<project-ref>

# Redis
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/1
```

In Supabase, create a Storage bucket named `submissions`. The backend uses the service role key to upload and download submission files.

### Build the evaluator image
The worker launches evaluation jobs using the `rl-evaluator:latest` image. Build it once:
```bash
docker build -f docker/Dockerfile.evaluator -t rl-evaluator:latest .
```

Alternatively (Compose profile):
```bash
docker compose build evaluator
```

### Start the stack
```bash
docker compose up -d --build
```

### Open the apps
- Gradio Frontend: `http://localhost:7860`
- API (OpenAPI docs): `http://localhost:8000/docs`
- Redis Commander (optional UI): `http://localhost:8081`
- Prometheus: `http://localhost:9090`
- Grafana: `http://localhost:3000` (admin/admin)

To stop everything: `docker compose down`

---

## Environment Variables

These are consumed by the services (see `docker-compose.yml` and `app/core/config.py`).

| Variable                 | Description                               | Default (compose/app)                     |
|--------------------------|-------------------------------------------|-------------------------------------------|
| DATABASE_URL             | SQLAlchemy URL (Supabase pooling)         | required                                  |
| REDIS_URL                | Redis URL (leaderboard cache)             | `redis://redis:6379/0`                    |
| CELERY_BROKER_URL        | Celery broker                             | `redis://redis:6379/1`                    |
| CELERY_RESULT_BACKEND    | Celery result backend                     | `redis://redis:6379/1`                    |
| SUPABASE_URL             | Supabase project URL                      | required                                  |
| SUPABASE_ANON_KEY        | Supabase anon key                         | optional (frontend or clients)            |
| SUPABASE_SERVICE_KEY     | Supabase service role key                 | required (server-side Storage access)     |
| SUPABASE_BUCKET          | Supabase Storage bucket name              | `submissions`                             |
| SECRET_KEY               | FastAPI app secret                        | `supersecret` (override in prod)          |
| DOCKER_HOST              | Docker socket for worker                  | `unix:///var/run/docker.sock`             |
| SENTRY_DSN               | Sentry DSN (optional)                     | -                                         |
| SENTRY_ENVIRONMENT       | Sentry environment name                   | `development`                             |
| SENTRY_TRACES_SAMPLE_RATE| Sentry APM sampling rate (0..1)           | `0.1`                                     |

---

## Usage

### Leaderboard - Model Upload
1. Go to `http://localhost:7860`.
2. In the Submit tab, choose an environment (e.g., `CartPole-v1`), provide optional user/algorithm labels, and upload your `.py` file.
3. Copy the shown Submission ID and check its status in the Check Status tab.
4. View the Leaderboard tab for real-time rankings.

### RL Evaluation - Competitive Battles
1. Go to `http://localhost:7860`.
2. In the Arena tab, you can select two models to evaluation.
3. The evaluation will run asynchronously, and you can check the results in the Check Status tab.
4. View the Leaderboard tab for real-time rankings.

### Using the Gradio Frontend
1. Go to `http://localhost:7860`.
2. In the Submit tab, choose an environment (e.g., `CartPole-v1`), provide optional user/algorithm labels, and upload your `.py` file.
3. Copy the shown Submission ID and check its status in the Check Status tab.
4. View the Leaderboard tab for real-time rankings.

### Submitting via API

Endpoint: `POST /api/submit/`

- Single-file mode (backward compatible):
  - `file`: Python file to evaluate (`.py`)
- Common fields:
  - `env_id`: Gym environment ID (default `CartPole-v1`)
  - `algorithm`: Label for your method (default `Custom`)
  - `user_id`: Your identifier (default `anonymous`)
  - `client_id` (optional): Provide your own UUID to track the submission immediately.

Examples:

Single file:
```bash
curl -X POST \
  -F "file=@example_agents/q_learning.py" \
  -F "env_id=CartPole-v1" \
  -F "algorithm=Q-Learning" \
  -F "user_id=team-rocket" \
  http://localhost:8000/api/submit/
```

Response:
```json
{
  "id": "<submission_uuid>",
  "status": "queued",
  "env_id": "CartPole-v1",
  "algorithm": "DQN"
}
```

### Checking Results

Endpoint: `GET /api/results/{submission_id}`

Returns status (`pending` | `processing` | `completed` | `failed`), final score if completed, and any error.

```bash
curl http://localhost:8000/api/results/<submission_uuid>
```

### Querying the Leaderboard

Endpoint: `GET /api/leaderboard/`

Query params:
- `env_id` (string, default `CartPole-v1`)
- `limit` (int, 1..100, default 50)

```bash
curl "http://localhost:8000/api/leaderboard/?env_id=CartPole-v1&limit=50"
```

### Health
`GET /health` → `{ "status": "healthy", ... }`

---

## Submission Contract

Your submission must:
1. Consist of one Python file (`.py`).
2. Accept the environment ID as its first CLI argument: your script will be invoked as:
   ```bash
   python -u submission.py <ENV_ID>
   ```
4. Print exactly one final JSON line to stdout that includes a numeric `score`. Optionally include `metrics` for per-episode rewards.

Example final output (printed as a single line):
```json
{"score": 123.45, "metrics": [9.0, 10.0, 11.0]}
```

Notes on the evaluator runtime (see `scripts/entrypoint.sh` and `app/core/docker.py`):
- Network disabled (`network_mode="none"`).
- Memory limit `512MiB`, CPU quota ~50% of one core, PIDs limit 50.
- Process is wrapped with `timeout 300s`, `nice`, `ionice`, and `ulimit`.
- The worker parses container logs and extracts the last valid JSON line. If no `score` is found or the process exits non-zero, the submission is marked failed with a helpful log tail.

See `example_agents/q_learning.py` for a simple reference implementation.

### Exact JSON Output Requirements

- Required: `score` (number)
- Optional: `metrics` (array of numbers, e.g., per-episode rewards)
- Optional: `episodes` (integer)
- Single final line: The evaluator extracts the last valid JSON object from your combined stdout/stderr. Ensure your final print is the JSON line and do not print anything after it.
- Be strict: Use `json.dumps(...)` for the final print. Avoid printing Python dicts directly.

Minimal schema (informal):
```json
{
  "type": "object",
  "required": ["score"],
  "properties": {
    "score": { "type": "number" },
    "metrics": { "type": "array", "items": { "type": "number" } },
    "episodes": { "type": "integer" }
  }
}
```

### Minimal `submission.py` template (with main function)

```python
import sys
import json
import logging

logger = logging.getLogger(__name__)

def train(env_id: str) -> dict:
    """Run your algorithm and return a dict with at least 'score'."""
    # TODO: Implement your algorithm here
    metrics = []
    score = 0.0
    return {"score": float(score), "metrics": metrics}

def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing environment ID"}))
        sys.exit(1)

    env_id = sys.argv[1]
    logger.info(f"Starting evaluation for env_id={env_id}")

    result = train(env_id)
    if not isinstance(result, dict) or "score" not in result:
        print(json.dumps({"error": "Result must be a dict containing 'score'"}))
        sys.exit(1)

    # Print exactly one final JSON line. Do not print anything after this.
    print(json.dumps({
        "score": float(result["score"]),
        "metrics": result.get("metrics", [])
    }))

if __name__ == "__main__":
    main()
```

### Real-time Progress Monitoring with Weights & Biases

For real-time progress monitoring during evaluation, you can include Weights & Biases (wandb) in your submission. This allows you to track training progress, metrics, and charts in real-time.

**Example with wandb integration:**

```python
import sys
import json
import logging
import wandb
import numpy as np

logger = logging.getLogger(__name__)

def train(env_id: str) -> dict:
    """Run your algorithm with wandb logging for real-time monitoring."""
    
    # Initialize wandb for this evaluation run
    wandb.init(
        project="rl-leaderboard-evaluation",
        name=f"submission-{env_id}",
        config={
            "env_id": env_id,
            "algorithm": "your-algorithm-name",
            "submission_id": "your-submission-id"
        }
    )
    
    # Your training loop
    episode_rewards = []
    for episode in range(100):
        # Your training logic here
        episode_reward = np.random.normal(100, 20)  # Example
        episode_rewards.append(episode_reward)
        
        # Log to wandb for real-time monitoring
        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward,
            "avg_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards)
        })
    
    # Close wandb run
    wandb.finish()
    
    return {
        "score": float(np.mean(episode_rewards)),
        "metrics": episode_rewards
    }

def main() -> None:
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing environment ID"}))
        sys.exit(1)

    env_id = sys.argv[1]
    logger.info(f"Starting evaluation for env_id={env_id}")

    result = train(env_id)
    if not isinstance(result, dict) or "score" not in result:
        print(json.dumps({"error": "Result must be a dict containing 'score'"}))
        sys.exit(1)

    # Print exactly one final JSON line. Do not print anything after this.
    print(json.dumps({
        "score": float(result["score"]),
        "metrics": result.get("metrics", [])
    }))

if __name__ == "__main__":
    main()
```

**Benefits of wandb integration:**
- 📊 **Real-time charts**: See training progress as it happens
- 📈 **Live metrics**: Monitor rewards, losses, and other metrics
- 🔍 **Debugging**: Identify issues during training
- 📱 **Mobile access**: Check progress from anywhere
- 🎯 **Performance tracking**: Compare different runs and algorithms

**Note**: The evaluator container includes wandb support, so you can use `import wandb` directly in your submissions.

### Example: Simple Q-learning agent (discrete envs)

This example mirrors `example_agents/q_learning.py` and satisfies the evaluator contract. It expects a Gymnasium environment ID and prints a single JSON line with a numeric `score` and optional `metrics`.

```python
import sys, json
import numpy as np
import gymnasium as gym


def train_q_learning(env_id: str, episodes: int = 200, max_steps: int = 100) -> dict:
    # For FrozenLake, use deterministic dynamics for faster convergence
    env_kwargs = {"is_slippery": False} if str(env_id).startswith("FrozenLake") else {}
    env = gym.make(env_id, **env_kwargs)

    # Discrete state/action spaces only
    if not hasattr(env.action_space, "n") or not hasattr(env.observation_space, "n"):
        return {"error": "Requires discrete state and action spaces"}

    num_states = int(env.observation_space.n)
    num_actions = int(env.action_space.n)
    q_table = np.zeros((num_states, num_actions), dtype=np.float32)

    alpha, gamma = 0.1, 0.95
    epsilon, min_epsilon, decay = 1.0, 0.05, 0.995
    episode_rewards = []

    for _ in range(episodes):
        reset_out = env.reset()
        state = reset_out[0] if isinstance(reset_out, tuple) else reset_out
        state = int(state)
        total_reward = 0.0

        for _ in range(max_steps):
            action = env.action_space.sample() if np.random.rand() < epsilon else int(np.argmax(q_table[state]))
            step_out = env.step(action)
            next_state, reward = step_out[0], step_out[1]
            terminated, truncated = step_out[2], step_out[3]
            done = bool(terminated or truncated)

            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_state = int(next_state)

            best_next = float(np.max(q_table[next_state]))
            q_table[state, action] = (1 - alpha) * q_table[state, action] + alpha * (reward + gamma * best_next)

            state = next_state
            total_reward += float(reward)
            if done:
                break

        epsilon = max(min_epsilon, epsilon * decay)
        episode_rewards.append(total_reward)

    env.close()
    return {"score": float(np.mean(episode_rewards)), "metrics": episode_rewards, "episodes": episodes}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(json.dumps({"error": "Missing environment ID"}))
        raise SystemExit(1)
    result = train_q_learning(sys.argv[1])
    if not isinstance(result, dict) or "score" not in result:
        print(json.dumps({"error": "No score produced"}))
        raise SystemExit(1)
    # Print exactly one final JSON line
    print(json.dumps(result))
```

### Local smoke test

- Run your script locally to verify it prints one final JSON line:
```bash
python submission.py CartPole-v1
```
- You should see a single-line JSON with a numeric `score` as the last output.

### Checklist before submitting

- `python -u submission.py <ENV_ID>` works locally and prints a final JSON line
- The last printed line contains a numeric `score`
- No extra prints after the final JSON line
- Optional `metrics` is an array of numbers (if included)
- If multi-file, you uploaded all required modules and set `main_file` correctly

---



## Docker Deployment

### Production Deployment with Docker

This section provides complete instructions for deploying the OpenRL Leaderboard using Docker in production environments.

#### Prerequisites
- Docker and Docker Compose v2
- Git
- Supabase account and project setup

#### Step-by-Step Deployment

1. **Clone the repository:**
```bash
git clone <your-repo-url>
cd RL\ Leaderboard
```

2. **Create environment configuration:**
Create a `.env` file at the repo root with your production configuration:
```env
# FastAPI app security (CHANGE THIS IN PRODUCTION)
SECRET_KEY=your-super-secret-production-key

# Supabase configuration (required)
SUPABASE_URL=https://<your-project-ref>.supabase.co
SUPABASE_ANON_KEY=your-public-anon-key
SUPABASE_SERVICE_KEY=your-service-role-key
SUPABASE_BUCKET=submissions

# Database connection (use Supabase connection pooling)
DATABASE_URL=postgresql://postgres:<encoded_password>@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require&options=project%3D<project-ref>

# Redis configuration
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/1
CELERY_RESULT_BACKEND=redis://redis:6379/1

# Optional: Sentry for error tracking
SENTRY_DSN=your-sentry-dsn
SENTRY_ENVIRONMENT=production
```

3. **Set up Supabase:**
   - Create a Storage bucket named `submissions` in your Supabase project
   - Ensure your service role key has proper permissions for file uploads/downloads
   - Configure connection pooling for better database performance

4. **Build the evaluator image:**
```bash
docker build -f docker/Dockerfile.evaluator -t rl-evaluator:latest .
```

5. **Deploy the entire stack:**
```bash
docker compose up -d --build
```

6. **Verify deployment:**
```bash
docker compose ps
```

#### Production Access Points

Once deployed, your services will be available at:
- **Gradio Frontend**: `http://your-domain:7860`
- **API Documentation**: `http://your-domain:8000/docs`
- **Health Check**: `http://your-domain:8000/health`
- **Metrics**: `http://your-domain:8000/metrics`
- **Prometheus**: `http://your-domain:9090`
- **Grafana**: `http://your-domain:3000` (admin/admin)


#### Management Commands

```bash
# Stop all services
docker compose down

# View logs
docker compose logs -f

# Restart specific service
docker compose restart api

# Update and redeploy
git pull
docker compose up -d --build

# Scale workers (if needed)
docker compose up -d --scale worker=3
```

---

## Local Development (with Docker)

This is useful for iterating on API/worker code. You still need Docker Engine installed to run evaluator containers.

### 1) Python deps
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Services
Run Redis (e.g., via Docker):
```bash
docker run -d --name rl-redis -p 6379:6379 redis:7
```

Export environment (adjust as needed):
```bash
export DATABASE_URL=postgresql://postgres:<encoded_password>@aws-0-<region>.pooler.supabase.com:6543/postgres?sslmode=require\&options=project%3D<project-ref>
export REDIS_URL=redis://localhost:6379/0
export CELERY_BROKER_URL=redis://localhost:6379/1
export CELERY_RESULT_BACKEND=redis://localhost:6379/1
export SUPABASE_URL=...; export SUPABASE_SERVICE_KEY=...; export SUPABASE_BUCKET=submissions
```

Build the evaluator image once:
```bash
docker build -f docker/Dockerfile.evaluator -t rl-evaluator:latest .
```

Ensure the worker can reach Docker (often default works):
```bash
export DOCKER_HOST=unix:///var/run/docker.sock
```

### 3) Run API and Worker
```bash
uvicorn app.main:app --reload --port 8000
celery -A app.core.celery.celery_app worker --loglevel=info
```

Open `http://localhost:8000/docs` for API docs. Optionally run the frontend via `python web interface`.

---

## Troubleshooting

- **Evaluator image not found**: Build it with `docker build -f docker/Dockerfile.evaluator -t rl-evaluator:latest .`.
- **Docker socket permission denied**: On Linux/macOS, ensure your user can access `/var/run/docker.sock`. In Compose, the worker runs as `root` and mounts the socket.
- **Redis/DB connection errors**: Verify services are healthy (`docker compose ps`) and env vars match.
- **Supabase upload/download errors**: Check keys and that the `submissions` bucket exists.
- **Submission fails with "No 'score' found"**: Ensure your script prints one final JSON line with a `score` field.
- **Frontend cannot reach API**: The frontend container uses `API_URL=http://api:8000`. When running locally without Compose, set `API_URL=http://localhost:8000`.

---

## Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes with clear commit messages
4. Open a Pull Request

---

## License
MIT

---

## Observability Stack (Prometheus, Grafana, Loki)

Production-grade observability is included:

- Prometheus metrics from API and Celery worker
- Grafana dashboards (pre-provisioned)
- Loki for logs

### New endpoints/ports
- API `/metrics` on port 8000
- Celery worker metrics server on port 9100
- Prometheus on port 9090
- Grafana on port 3000



### Metrics exposed
- `submissions_received_total{mode}`
- `submissions_validation_failures_total{reason}`
- `submissions_upload_bytes_total`
- `evaluation_started_total`
- `evaluation_completed_total{env_id}`
- `evaluation_failed_total{reason}`
- `evaluation_duration_seconds_bucket/sum/count{env_id}`
- `leaderboard_queries_total{env_id,sort}`
- `leaderboard_query_duration_seconds_bucket/sum/count`

Plus default FastAPI metrics (requests, latencies, status codes, exceptions).

