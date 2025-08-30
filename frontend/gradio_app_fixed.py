# from app.core.docker import logger
import gradio as gr
import requests
import time
import json
from datetime import datetime, timezone
import os
import logging
import asyncio
import websockets
from typing import Dict, List, Optional

API_URL = os.getenv("API_URL", "http://localhost:8000")
PORT = int(os.getenv("PORT", "7860"))
# Public site URL for SEO; uses environment variable or defaults to Render URL
SITE_URL = os.getenv("PUBLIC_BASE_URL", "https://rl-eval-leaderboard.onrender.com").rstrip("/")
GITHUB_URL = "https://github.com/YuvrajSingh-mist/RL-Eval-Leaderboard"
_last_submission_id = None
logger = logging.getLogger(__name__)

# Global variables for WebSocket connections
websocket_connections = {}

# Comprehensive environment list
ALL_ENVIRONMENTS = [
    # Classic Gym environments
    "CartPole-v1", "LunarLander-v2", "Acrobot-v1", "MountainCar-v0",
    # ALE environments
    "basketball_pong", "boxing", "combat_plane", "combat_tank", "double_dunk",
    "entombed_competitive", "entombed_cooperative", "flag_capture", "foozpong",
    "ice_hockey", "joust", "mario_bros", "maze_craze", "othello", "pong",
    "quadrapong", "space_invaders", "space_war", "surround", "tennis",
    "video_checkers", "volleyball_pong", "warlords", "wizard_of_wor"
]

# PettingZoo Atari environments for human vs AI battles
ALE_ENVIRONMENTS = [
    "pong_v3", "boxing_v2", "tennis_v3", "ice_hockey_v2", "double_dunk_v3", 
    "wizard_of_wor_v3", "space_invaders_v2", "basketball_pong_v3", "volleyball_pong_v3",
    "foozpong_v3", "quadrapong_v4", "joust_v3", "mario_bros_v3", "maze_craze_v3",
    "othello_v3", "video_checkers_v4", "warlords_v3", "combat_plane_v2", "combat_tank_v2",
    "flag_capture_v2", "surround_v2", "space_war_v2", "entombed_competitive_v3", "entombed_cooperative_v3"
]

def refresh_environments():
    """Refresh environment list from API"""
    try:
        response = requests.get(f"{API_URL}/api/environments")
        if response.status_code == 200:
            data = response.json()
            return data.get("environments", ALL_ENVIRONMENTS)
        else:
            return ALL_ENVIRONMENTS
    except Exception as e:
        logger.error(f"Error refreshing environments: {e}")
        return ALL_ENVIRONMENTS
    
def submit_script(file, env_id, algorithm, user_id):
    # Single-file only
    if not file:
        return "⚠️ Please upload a Python script (.py)"

    try:
        import uuid as _uuid
        client_id = str(_uuid.uuid4())

        # Single-file path only
        if not str(file.name).lower().endswith('.py'):
            return "⚠️ Only .py files are accepted"
        submit_files = {'file': (file.name, open(file.name, 'rb'), 'text/plain')}
        data = {
            'env_id': env_id,
            'algorithm': algorithm,
            'user_id': user_id or "anonymous",
            'client_id': client_id
        }

        response = requests.post(f"{API_URL}/api/submit/", files=submit_files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            # Persist locally for Check Status tab (until refresh)
            global _last_submission_id
            _last_submission_id = result.get('id', client_id)
            sid = _last_submission_id
            html = f"""
            <div class="status-box success">
              <div class="status-header">
                <span>✅</span>
                <span>Submission queued</span>
                <span class="status-pill">Waiting to evaluate</span>
              </div>
              <div class="status-id">
                <b>ID:</b> <code>{sid}</code>
                <button class="copy-btn" onclick="navigator.clipboard.writeText('{sid}'); this.innerText='Copied'; setTimeout(()=>{{ this.innerText='Copy ID'; }}, 1600);">Copy ID</button>
              </div>
              <div class="status-kv">
                <div class="label">Environment</div><div class="value">{result['env_id']}</div>
                <div class="label">Algorithm</div><div class="value">{result['algorithm']}</div>
              </div>
              <div class="status-foot"><b>Your model is now available in the RL Arena!</b> Other users can battle against it.</div>
            </div>
            """
            return html
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ Error {response.status_code}: {error_detail}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def create_battle(player_model_id, opponent_model_id, env_id):
    """Create a battle in the RL Arena"""
    try:
        data = {
            'player_model_id': player_model_id,
            'env_id': env_id
        }
        
        if opponent_model_id:
            data['opponent_model_id'] = opponent_model_id

        response = requests.post(f"{API_URL}/api/rlarena/battles", json=data)
        
        if response.status_code == 200:
            result = response.json()
            battle_id = result['battle_id']
            
            html = f"""
            <div class="status-box success">
              <div class="status-header">
                <span>⚔️</span>
                <span>Battle Created</span>
                <span class="status-pill">Ready</span>
              </div>
              <div class="status-id">
                <b>Battle ID:</b> <code>{battle_id}</code>
                <button class="copy-btn" onclick="navigator.clipboard.writeText('{battle_id}'); this.innerText='Copied'; setTimeout(()=>{{ this.innerText='Copy ID'; }}, 1600);">Copy ID</button>
              </div>
              <div class="battle-details">
                <div class="battle-player">
                  <h4>Player</h4>
                  <p><strong>Model:</strong> {player_model_id}</p>
                  <p><strong>Environment:</strong> {env_id}</p>
                </div>
                <div class="battle-vs">VS</div>
                <div class="battle-opponent">
                  <h4>Opponent</h4>
                  <p><strong>Model:</strong> {opponent_model_id or 'Random AI'}</p>
                </div>
              </div>
              <div class="status-foot">
                <b>Battle is ready!</b> Use the Battle ID to check status and view results.
              </div>
            </div>
            """
            return html, battle_id
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ Error {response.status_code}: {error_detail}", None
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def create_human_vs_ai_battle(opponent_model_id, env_id):
    """Create a human vs AI battle in ALE environment"""
    try:
        data = {
            'env_id': env_id
        }
        
        if opponent_model_id:
            data['opponent_model_id'] = opponent_model_id

        response = requests.post(f"{API_URL}/api/rlarena/human-vs-ai", json=data)
        
        if response.status_code == 200:
            result = response.json()
            battle_id = result['battle_id']
            
            # Get keyboard controls
            controls = result.get('keyboard_controls', {})
            controls_text = ""
            if 'controls' in controls:
                for key, action in controls['controls'].items():
                    controls_text += f"{key}: {action}<br>"
            
            html = f"""
            <div class="status-box success">
              <div class="status-header">
                <span>🎮</span>
                <span>Human vs AI Battle Ready</span>
                <span class="status-pill">Ready to Play</span>
              </div>
              <div class="status-id">
                <b>Battle ID:</b> <code>{battle_id}</code>
                <button class="copy-btn" onclick="navigator.clipboard.writeText('{battle_id}'); this.innerText='Copied'; setTimeout(()=>{{ this.innerText='Copy ID'; }}, 1600);">Copy ID</button>
              </div>
              <div class="battle-details">
                <div class="battle-player">
                  <h4>You (Human)</h4>
                  <p><strong>Environment:</strong> {result['environment'].title()}</p>
                  <p><strong>Description:</strong> {controls.get('description', 'ALE Environment')}</p>
                </div>
                <div class="battle-vs">VS</div>
                <div class="battle-opponent">
                  <h4>AI Opponent</h4>
                  <p><strong>Model:</strong> {result['opponent_model']['name']}</p>
                  <p><strong>Algorithm:</strong> {result['opponent_model']['algorithm']}</p>
                  <p><strong>Creator:</strong> {result['opponent_model']['user_id']}</p>
                </div>
              </div>
              <div class="keyboard-controls">
                <h4>🎯 Keyboard Controls:</h4>
                <div class="controls-grid">
                  {controls_text}
                </div>
              </div>
              <div class="status-foot">
                <b>Ready to play!</b> The game will start in a new window. Use the keyboard controls to play against the AI.
              </div>
            </div>
            """
            return html, battle_id
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ Error {response.status_code}: {error_detail}", None
            
    except Exception as e:
        return f"❌ Error: {str(e)}", None

def launch_web_game(opponent_model_id, env_id):
    """Launch the web game interface"""
    try:
        # Create a unique battle ID
        import uuid
        battle_id = str(uuid.uuid4())
        
        # Build the web game URL
        base_url = SITE_URL if SITE_URL != "https://rl-eval-leaderboard.onrender.com" else "http://localhost:8080"
        game_url = f"{base_url}/game_interface.html?battle_id={battle_id}&environment={env_id}"
        
        if opponent_model_id:
            game_url += f"&opponent_id={opponent_model_id}"
        
        # Get environment info
        env_names = {
            'pong_v3': 'Pong',
            'boxing_v2': 'Boxing',
            'tennis_v3': 'Tennis',
            'ice_hockey_v2': 'Ice Hockey',
            'double_dunk_v3': 'Double Dunk',
            'wizard_of_wor_v3': 'Wizard of Wor',
            'space_invaders_v2': 'Space Invaders',
            'basketball_pong_v3': 'Basketball Pong',
            'volleyball_pong_v3': 'Volleyball Pong',
            'foozpong_v3': 'Foozpong',
            'quadrapong_v4': 'Quadrapong',
            'joust_v3': 'Joust',
            'mario_bros_v3': 'Mario Bros',
            'maze_craze_v3': 'Maze Craze',
            'othello_v3': 'Othello',
            'video_checkers_v4': 'Video Checkers',
            'warlords_v3': 'Warlords',
            'combat_plane_v2': 'Combat: Plane',
            'combat_tank_v2': 'Combat: Tank',
            'flag_capture_v2': 'Flag Capture',
            'surround_v2': 'Surround',
            'space_war_v2': 'Space War',
            'entombed_competitive_v3': 'Entombed: Competitive',
            'entombed_cooperative_v3': 'Entombed: Cooperative'
        }
        
        env_name = env_names.get(env_id, env_id)
        
        html = f"""
        <div class="status-box success">
          <div class="status-header">
            <span>🌐</span>
            <span>Web Game Launched</span>
            <span class="status-pill">Ready to Play</span>
          </div>
          <div class="web-game-info">
            <h4>🎮 {env_name}</h4>
            <p><strong>Battle ID:</strong> <code>{battle_id}</code></p>
            <p><strong>Environment:</strong> {env_name}</p>
            {f'<p><strong>AI Opponent:</strong> {opponent_model_id}</p>' if opponent_model_id else '<p><strong>AI Opponent:</strong> Random AI</p>'}
          </div>
          <div class="web-game-launch">
            <a href="{game_url}" target="_blank" class="launch-button">
              🚀 Launch Game in New Tab
            </a>
            <p style="margin-top: 10px; font-size: 0.9em; color: #666;">
              The game will open in a new browser tab. Make sure to allow popups if prompted.
            </p>
          </div>
          <div class="status-foot">
            <b>Game Features:</b> Beautiful interface, real-time controls, live statistics, and cross-platform support!
          </div>
        </div>
        """
        return html
        
    except Exception as e:
        return f"❌ Error launching web game: {str(e)}"

def get_battle_status(battle_id):
    """Get the status of a battle"""
    if not battle_id:
        return "⚠️ Please enter a Battle ID"
    
    try:
        response = requests.get(f"{API_URL}/api/rlarena/battles/{battle_id}")
        
        if response.status_code == 200:
            result = response.json()
            status = result['status']
            
            if status == "completed" and result.get('result'):
                battle_result = result['result']
                winner = battle_result['winner']
                winner_text = "Player" if winner == "player" else "Opponent" if winner == "opponent" else "Draw"
                
                html = f"""
                <div class="status-box {'success' if winner == 'player' else 'warning' if winner == 'draw' else 'error'}">
                  <div class="status-header">
                    <span>🏆</span>
                    <span>Battle Complete</span>
                    <span class="status-pill">{winner_text}</span>
                  </div>
                  <div class="battle-result">
                    <div class="result-stats">
                      <div class="stat">
                        <span class="label">Player Score</span>
                        <span class="value">{battle_result.get('player_score', 0)}</span>
                      </div>
                      <div class="stat">
                        <span class="label">Opponent Score</span>
                        <span class="value">{battle_result.get('opponent_score', 0)}</span>
                      </div>
                      <div class="stat">
                        <span class="label">Duration</span>
                        <span class="value">{battle_result.get('duration', 0):.2f}s</span>
                      </div>
                    </div>
                  </div>
                  <div class="status-foot">
                    <b>Battle finished!</b> {winner_text} won the match.
                  </div>
                </div>
                """
                return html
            elif status == "processing":
                return """
                <div class="status-box info">
                  <div class="status-header">
                    <span>⚙️</span>
                    <span>Processing</span>
                    <span class="status-pill">Running</span>
                  </div>
                  <div class="status-foot">Battle is in progress...</div>
                </div>
                """
            elif status == "pending":
                return """
                <div class="status-box info">
                  <div class="status-header">
                    <span>⏳</span>
                    <span>Queued</span>
                    <span class="status-pill">Waiting</span>
                  </div>
                  <div class="status-foot">Battle is waiting to start...</div>
                </div>
                """
            else:
                return f"Unknown status: {status}"
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ Error {response.status_code}: {error_detail}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_arena_rankings(environment):
    """Get arena rankings for a specific environment"""
    try:
        params = {}
        if environment and environment != "All":
            params['environment'] = environment
            
        response = requests.get(f"{API_URL}/api/rlarena/rankings", params=params)
        
        if response.status_code == 200:
            data = response.json()
            rankings = data.get('rankings', [])
            
            if not rankings:
                return "No rankings available for this environment."
            
            html = "<div class='rankings-table'>"
            html += """
            <div class="rankings-header">
                <div class="rank-col">Rank</div>
                <div class="model-col">Model</div>
                <div class="wins-col">Wins</div>
                <div class="losses-col">Losses</div>
                <div class="winrate-col">Win Rate</div>
            </div>
            """
            
            for i, ranking in enumerate(rankings, 1):
                win_rate = (ranking.get('wins', 0) / max(ranking.get('total_games', 1), 1)) * 100
                html += f"""
                <div class="rankings-row">
                    <div class="rank-col">#{i}</div>
                    <div class="model-col">{ranking.get('model_name', 'Unknown')}</div>
                    <div class="wins-col">{ranking.get('wins', 0)}</div>
                    <div class="losses-col">{ranking.get('losses', 0)}</div>
                    <div class="winrate-col">{win_rate:.1f}%</div>
                </div>
                """
            html += "</div>"
            return html
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ Error {response.status_code}: {error_detail}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def list_hub_models(env_id, algorithm, user_id):
    """List models from the RL Hub"""
    try:
        params = {}
        if env_id:
            params['env_id'] = env_id
        if algorithm:
            params['algorithm'] = algorithm
        if user_id:
            params['user_id'] = user_id
            
        response = requests.get(f"{API_URL}/api/rlhub/models", params=params)
        
        if response.status_code == 200:
            data = response.json()
            models = data.get('models', [])
            
            if not models:
                return "No models found matching your criteria."
            
            html = "<div class='models-table'>"
            html += """
            <div class="models-header">
                <div class="model-id-col">Model ID</div>
                <div class="name-col">Name</div>
                <div class="user-col">User</div>
                <div class="algorithm-col">Algorithm</div>
                <div class="env-col">Environment</div>
                <div class="date-col">Created</div>
            </div>
            """
            
            for model in models:
                date_str = model.get("created_at", "")[:10] if model.get("created_at") else "N/A"
                html += f"""
                <div class="models-row">
                    <div class="model-id-col"><code>{model.get('id', 'N/A')}</code></div>
                    <div class="name-col">{model.get('name', 'N/A')}</div>
                    <div class="user-col">{model.get('user_id', 'N/A')}</div>
                    <div class="algorithm-col">{model.get('algorithm', 'N/A')}</div>
                    <div class="env-col">{model.get('env_id', 'N/A')}</div>
                    <div class="date-col">{date_str}</div>
                </div>
                """
            html += "</div>"
            return html
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ Error {response.status_code}: {error_detail}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def get_leaderboard(env_id, algorithm, user_id):
    """Get leaderboard entries"""
    try:
        params = {}
        if env_id:
            params['env_id'] = env_id
        if algorithm:
            params['algorithm'] = algorithm
        if user_id:
            params['user_id'] = user_id
            
        response = requests.get(f"{API_URL}/api/leaderboard", params=params)
        
        if response.status_code == 200:
            data = response.json()
            entries = data.get('entries', [])
            
            if not entries:
                return "No entries found matching your criteria."
            
            html = "<div class='leaderboard-table'>"
            html += """
            <div class="leaderboard-header">
                <div class="rank-col">Rank</div>
                <div class="score-col">Score</div>
                <div class="user-col">User</div>
                <div class="algorithm-col">Algorithm</div>
                <div class="date-col">Date</div>
            </div>
            """
            
            for i, entry in enumerate(entries, 1):
                date_str = entry.get("created_at", "")[:10] if entry.get("created_at") else "N/A"
                html += f"""
                <div class="leaderboard-row">
                    <div class="rank-col">#{i}</div>
                    <div class="score-col">{entry.get('score', 0):.2f}</div>
                    <div class="user-col">{entry.get('user_id', 'N/A')}</div>
                    <div class="algorithm-col">{entry.get('algorithm', 'N/A')}</div>
                    <div class="date-col">{date_str}</div>
                </div>
                """
            html += "</div>"
            return html
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ Error {response.status_code}: {error_detail}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

def check_status(submission_id):
    """Check the status of a submission"""
    if not submission_id:
        return "⚠️ Please enter a Submission ID"
    
    try:
        response = requests.get(f"{API_URL}/api/results/{submission_id}")
        
        if response.status_code == 200:
            result = response.json()
            status = result.get('status', 'unknown')
            
            if status == "completed":
                score = result.get('score', 'N/A')
                duration = result.get('duration_seconds', 'N/A')
                html = f"""
                <div class="status-box success">
                  <div class="status-header">
                    <span>✅</span>
                    <span>Evaluation Complete</span>
                    <span class="status-pill">Success</span>
                  </div>
                  <div class="status-kv">
                    <div class="label">Score</div><div class="value">{score}</div>
                    <div class="label">Duration</div><div class="value">{duration}s</div>
                  </div>
                </div>
                """
                return html
            elif status == "processing":
                return """
                <div class="status-box info">
                  <div class="status-header">
                    <span>⚙️</span>
                    <span>Processing</span>
                    <span class="status-pill">Running</span>
                  </div>
                  <div class="status-foot">Your submission is being evaluated...</div>
                </div>
                """
            elif status == "pending":
                return """
                <div class="status-box info">
                  <div class="status-header">
                    <span>⏳</span>
                    <span>Queued</span>
                    <span class="status-pill">Waiting</span>
                  </div>
                  <div class="status-foot">Your submission is waiting to be processed...</div>
                </div>
                """
            elif status == "failed":
                error = result.get('error', 'Unknown error')
                return f"""
                <div class="status-box error">
                  <div class="status-header">
                    <span>❌</span>
                    <span>Failed</span>
                    <span class="status-pill">Error</span>
                  </div>
                  <div class="status-foot">Error: {error}</div>
                </div>
                """
            else:
                return f"Unknown status: {status}"
        else:
            error_detail = response.json().get('detail', 'Unknown error')
            return f"❌ Error {response.status_code}: {error_detail}"
            
    except Exception as e:
        return f"❌ Error: {str(e)}"

# Custom CSS for modern UI
custom_css = """
<style>
/* Modern UI Styles */
.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}

/* Status boxes */
.status-box {
    border-radius: 12px;
    padding: 20px;
    margin: 10px 0;
    border-left: 4px solid;
}

.status-box.success {
    background: linear-gradient(135deg, #d4edda, #c3e6cb);
    border-left-color: #28a745;
}

.status-box.error {
    background: linear-gradient(135deg, #f8d7da, #f5c6cb);
    border-left-color: #dc3545;
}

.status-box.info {
    background: linear-gradient(135deg, #d1ecf1, #bee5eb);
    border-left-color: #17a2b8;
}

.status-box.warning {
    background: linear-gradient(135deg, #fff3cd, #ffeaa7);
    border-left-color: #ffc107;
}

.status-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 15px;
    font-weight: 600;
}

.status-pill {
    background: rgba(255, 255, 255, 0.8);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.8em;
    font-weight: 500;
}

.status-id {
    background: rgba(255, 255, 255, 0.5);
    padding: 10px;
    border-radius: 8px;
    margin: 10px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}

.copy-btn {
    background: #007bff;
    color: white;
    border: none;
    padding: 4px 8px;
    border-radius: 4px;
    cursor: pointer;
    font-size: 0.8em;
}

.status-kv {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 10px;
    margin: 15px 0;
}

.status-kv .label {
    font-weight: 500;
    color: #666;
}

.status-kv .value {
    font-weight: 600;
}

.status-foot {
    margin-top: 15px;
    padding-top: 15px;
    border-top: 1px solid rgba(0, 0, 0, 0.1);
    font-weight: 500;
}

/* Battle details */
.battle-details {
    display: grid;
    grid-template-columns: 1fr auto 1fr;
    gap: 20px;
    align-items: center;
    margin: 15px 0;
    padding: 15px;
    background: rgba(255, 255, 255, 0.5);
    border-radius: 8px;
}

.battle-player, .battle-opponent {
    text-align: center;
}

.battle-vs {
    font-size: 1.5em;
    font-weight: bold;
    color: #666;
}

/* Tables */
.leaderboard-table, .rankings-table, .models-table {
    border-radius: 8px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.leaderboard-header, .rankings-header, .models-header {
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    display: grid;
    grid-template-columns: 0.5fr 1fr 1fr 1fr 1fr;
    padding: 12px;
    font-weight: 600;
}

.rankings-header {
    grid-template-columns: 0.5fr 1fr 1fr 1fr 1fr;
}

.models-header {
    grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr;
}

.leaderboard-row, .rankings-row, .models-row {
    display: grid;
    grid-template-columns: 0.5fr 1fr 1fr 1fr 1fr;
    padding: 10px 12px;
    border-bottom: 1px solid #eee;
    transition: background-color 0.2s;
}

.rankings-row {
    grid-template-columns: 0.5fr 1fr 1fr 1fr 1fr;
}

.models-row {
    grid-template-columns: 1fr 1fr 1fr 1fr 1fr 1fr;
}

.leaderboard-row:hover, .rankings-row:hover, .models-row:hover {
    background-color: #f8f9fa;
}

.leaderboard-row:nth-child(even), .rankings-row:nth-child(even), .models-row:nth-child(even) {
    background-color: #f8f9fa;
}

/* Web game launch button */
.launch-button {
    display: inline-block;
    background: linear-gradient(135deg, #667eea, #764ba2);
    color: white;
    padding: 12px 24px;
    border-radius: 25px;
    text-decoration: none;
    font-weight: 600;
    transition: transform 0.2s;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
}

.launch-button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
}

.web-game-info {
    background: rgba(255, 255, 255, 0.5);
    padding: 15px;
    border-radius: 8px;
    margin: 15px 0;
}

.web-game-launch {
    text-align: center;
    margin: 20px 0;
}

/* Responsive design */
@media (max-width: 768px) {
    .status-kv {
        grid-template-columns: 1fr;
    }
    
    .battle-details {
        grid-template-columns: 1fr;
        gap: 10px;
    }
    
    .leaderboard-header, .leaderboard-row,
    .rankings-header, .rankings-row,
    .models-header, .models-row {
        grid-template-columns: 1fr;
        text-align: center;
    }
}
</style>
"""

# Create the Gradio interface
with gr.Blocks(css=custom_css, title="NeatRL - Reinforcement Learning Platform") as demo:
    gr.Markdown(
        f"""
        # 🧠 NeatRL - Reinforcement Learning Platform
        
        Welcome to **NeatRL**, a comprehensive platform for evaluating and competing with reinforcement learning agents!
        
        [![GitHub](https://img.shields.io/badge/GitHub-Repository-blue?style=flat&logo=github)]({GITHUB_URL})
        [![API](https://img.shields.io/badge/API-Documentation-green?style=flat)]({SITE_URL}/docs)
        """
    )
    
    with gr.Tabs():
        # RL Hub Tab
        with gr.Tab("🏠 RL Hub", id=0):
            gr.Markdown("### Upload and Manage Your Models")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Upload Model")
                    file = gr.File(label="Python Script (.py)", file_types=[".py"])
                    env_id = gr.Dropdown(
                        choices=ALL_ENVIRONMENTS,
                        label="Environment",
                        value="CartPole-v1"
                    )
                    algorithm = gr.Textbox(label="Algorithm", placeholder="e.g., PPO, DQN, A2C")
                    user_id = gr.Textbox(label="User ID", placeholder="Your name or ID")
                    submit_btn = gr.Button("📤 Submit Model", variant="primary")
                    submit_output = gr.HTML(label="Submission Status")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Browse Models")
                    browse_env = gr.Dropdown(
                        choices=["All"] + ALL_ENVIRONMENTS,
                        label="Environment Filter",
                        value="All"
                    )
                    browse_algorithm = gr.Textbox(label="Algorithm Filter", placeholder="Filter by algorithm")
                    browse_user = gr.Textbox(label="User Filter", placeholder="Filter by user")
                    browse_btn = gr.Button("🔍 Browse Models", variant="secondary")
                    models_output = gr.HTML(label="Models")
            
            submit_btn.click(
                submit_script,
                inputs=[file, env_id, algorithm, user_id],
                outputs=[submit_output]
            )
            
            browse_btn.click(
                list_hub_models,
                inputs=[browse_env, browse_algorithm, browse_user],
                outputs=models_output
            )
        
        # RL Arena Tab
        with gr.Tab("⚔️ RL Arena", id=1):
            gr.Markdown("### Battle Your Models Against Others")
            
            with gr.Tabs():
                # AI vs AI Battles
                with gr.Tab("🤖 AI vs AI"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Create AI Battle")
                            player_model = gr.Textbox(label="Your Model ID", placeholder="Enter your model ID")
                            opponent_model = gr.Textbox(label="Opponent Model ID (Optional)", placeholder="Leave empty for random opponent")
                            battle_env = gr.Dropdown(
                                choices=ALL_ENVIRONMENTS,
                                label="Battle Environment",
                                value="CartPole-v1"
                            )
                            battle_btn = gr.Button("⚔️ Start AI Battle", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("#### Battle Status")
                            battle_id_input = gr.Textbox(label="Battle ID", placeholder="Enter battle ID to check status")
                            check_battle_btn = gr.Button("🔍 Check Status", variant="secondary")
                            battle_status_output = gr.HTML(label="Battle Status")
                    
                    battle_btn.click(
                        create_battle,
                        inputs=[player_model, opponent_model, battle_env],
                        outputs=[battle_status_output, battle_id_input]
                    )
                    
                    check_battle_btn.click(
                        get_battle_status,
                        inputs=[battle_id_input],
                        outputs=[battle_status_output]
                    )
                
                # Human vs AI Battles
                with gr.Tab("🎮 Human vs AI"):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Create Human vs AI Battle")
                            gr.Markdown("**Play against AI models in ALE environments!**")
                            ale_opponent_model = gr.Textbox(label="AI Model ID (Optional)", placeholder="Leave empty for random AI opponent")
                            ale_env = gr.Dropdown(
                                choices=ALE_ENVIRONMENTS,
                                label="ALE Environment",
                                value="pong_v3"
                            )
                            ale_battle_btn = gr.Button("🎮 Start Human vs AI Battle", variant="primary")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("#### ALE Battle Status")
                            ale_battle_id_input = gr.Textbox(label="Battle ID", placeholder="Enter battle ID to check status")
                            ale_check_battle_btn = gr.Button("🔍 Check Status", variant="secondary")
                            ale_battle_status_output = gr.HTML(label="ALE Battle Status")
                    
                    ale_battle_btn.click(
                        create_human_vs_ai_battle,
                        inputs=[ale_opponent_model, ale_env],
                        outputs=[ale_battle_status_output, ale_battle_id_input]
                    )
                    
                    ale_check_battle_btn.click(
                        get_battle_status,
                        inputs=[ale_battle_id_input],
                        outputs=[ale_battle_status_output]
                    )
                
                # Web Game Interface
                with gr.Tab("🌐 Web Game Interface"):
                    gr.Markdown("### 🎮 Play in Your Browser")
                    gr.Markdown("**Experience NeatRL games directly in your web browser with beautiful graphics and real-time controls!**")
                    
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("#### Game Setup")
                            web_opponent_model = gr.Textbox(
                                label="AI Model ID (Optional)", 
                                placeholder="Leave empty for random AI opponent",
                                info="Enter a specific model ID or leave empty to battle a random AI"
                            )
                            web_env = gr.Dropdown(
                                choices=ALE_ENVIRONMENTS,
                                label="Game Environment",
                                value="pong_v3",
                                info="Choose from 24 classic Atari games"
                            )
                            web_battle_btn = gr.Button("🚀 Launch Web Game", variant="primary", size="lg")
                        
                        with gr.Column(scale=1):
                            gr.Markdown("#### Game Features")
                            gr.Markdown("""
                            ✨ **Beautiful Interface** - Modern, responsive design
                            🎮 **Real-time Controls** - Keyboard input with visual feedback
                            📊 **Live Statistics** - Real-time score tracking
                            🔗 **Seamless Integration** - Works with existing NeatRL system
                            📱 **Cross-platform** - Works on desktop, tablet, and mobile
                            """)
                    
                    web_game_output = gr.HTML(label="Game Launch Status")
                    
                    web_battle_btn.click(
                        launch_web_game,
                        inputs=[web_opponent_model, web_env],
                        outputs=[web_game_output]
                    )
            
            gr.Markdown("### 🏆 Arena Rankings")
            rankings_env = gr.Dropdown(
                choices=["All"] + ALL_ENVIRONMENTS,
                label="Environment",
                value="All"
            )
            rankings_btn = gr.Button("📊 View Rankings", variant="secondary")
            rankings_output = gr.HTML(label="Arena Rankings")
            
            rankings_btn.click(
                get_arena_rankings,
                inputs=[rankings_env],
                outputs=[rankings_output]
            )
        
        # Original Leaderboard Tab
        with gr.Tab("📊 Leaderboard", id=2):
            gr.Markdown("### Traditional RL Evaluation Leaderboard")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Submit Agent")
                    file = gr.File(label="Python Script (.py)", file_types=[".py"])
                    env_id = gr.Dropdown(
                        choices=ALL_ENVIRONMENTS,
                        label="Environment",
                        value="CartPole-v1"
                    )
                    algorithm = gr.Textbox(label="Algorithm", placeholder="e.g., PPO, DQN, A2C")
                    user_id = gr.Textbox(label="User ID", placeholder="Your name or ID")
                    submit_btn = gr.Button("📤 Submit", variant="primary")
                    submit_output = gr.HTML(label="Submission Status")
                
                with gr.Column(scale=1):
                    gr.Markdown("#### Check Status")
                    submission_id = gr.Textbox(label="Submission ID", placeholder="Enter submission ID")
                    check_btn = gr.Button("🔍 Check Status", variant="secondary")
                    status_output = gr.HTML(label="Status")
            
            submit_btn.click(
                submit_script,
                inputs=[file, env_id, algorithm, user_id],
                outputs=[submit_output]
            )
            
            check_btn.click(
                check_status,
                inputs=[submission_id],
                outputs=[status_output]
            )
            
            gr.Markdown("### 📈 View Leaderboard")
            leaderboard_env = gr.Dropdown(
                choices=["All"] + ALL_ENVIRONMENTS,
                label="Environment",
                value="All"
            )
            leaderboard_algorithm = gr.Textbox(label="Algorithm Filter", placeholder="Filter by algorithm")
            leaderboard_user = gr.Textbox(label="User Filter", placeholder="Filter by user")
            leaderboard_btn = gr.Button("📊 View Leaderboard", variant="secondary")
            leaderboard_output = gr.HTML(label="Leaderboard")
            
            leaderboard_btn.click(
                get_leaderboard,
                inputs=[leaderboard_env, leaderboard_algorithm, leaderboard_user],
                outputs=[leaderboard_output]
            )
        
        # Settings Tab
        with gr.Tab("⚙️ Settings", id=3):
            gr.Markdown("### Platform Configuration")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Environment Management")
                    refresh_env_btn = gr.Button("🔄 Refresh Environments", variant="secondary")
                    env_status = gr.Textbox(label="Environment Status", value="Ready", interactive=False)
                
                with gr.Column(scale=1):
                    gr.Markdown("#### System Information")
                    gr.Markdown(f"""
                    **API URL:** {API_URL}
                    **Site URL:** {SITE_URL}
                    **Port:** {PORT}
                    **GitHub:** {GITHUB_URL}
                    """)
            
            refresh_env_btn.click(
                refresh_environments,
                outputs=[env_status]
            )

# Launch the interface
if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=PORT, share=False)
