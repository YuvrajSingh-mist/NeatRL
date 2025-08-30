#!/usr/bin/env python3
"""
ALE Battle System for NeatRL
Handles human vs AI battles in ALE environments with keyboard controls
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
import subprocess
import tempfile
from typing import Dict, Optional
import websockets
from websockets.server import serve
import cv2
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ALEBattleSystem:
    def __init__(self):
        self.active_battles: Dict[str, 'ALEBattle'] = {}
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.running = True
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8766):
        """Start the ALE WebSocket server"""
        logger.info(f"Starting ALE Battle System on {host}:{port}")
        
        async with serve(self.handle_websocket, host, port):
            await asyncio.Future()
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections for ALE battles"""
        try:
            battle_id = path.strip('/')
            if not battle_id:
                await websocket.close(1008, "Invalid battle ID")
                return
            
            self.websocket_connections[battle_id] = websocket
            logger.info(f"ALE WebSocket connected for battle {battle_id}")
            
            await websocket.send(json.dumps({
                "type": "connection_established",
                "battle_id": battle_id,
                "message": "Connected to ALE Battle System"
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_battle_message(battle_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"ALE WebSocket disconnected for battle {battle_id}")
        finally:
            if battle_id in self.websocket_connections:
                del self.websocket_connections[battle_id]
    
    async def handle_battle_message(self, battle_id: str, data: Dict):
        """Handle ALE battle messages"""
        message_type = data.get("type")
        
        if message_type == "start_ale_battle":
            await self.start_ale_battle(battle_id, data)
        elif message_type == "keyboard_input":
            await self.handle_keyboard_input(battle_id, data)
        elif message_type == "end_battle":
            await self.end_battle(battle_id)
    
    async def start_ale_battle(self, battle_id: str, data: Dict):
        """Start a human vs AI battle in ALE environment"""
        try:
            opponent_model_path = data.get("opponent_model_path")
            env_id = data.get("env_id", "boxing")
            
            battle = ALEBattle(
                battle_id=battle_id,
                opponent_model_path=opponent_model_path,
                env_id=env_id,
                battle_system=self
            )
            
            self.active_battles[battle_id] = battle
            asyncio.create_task(battle.run())
            
            logger.info(f"ALE Battle {battle_id} started in {env_id}")
            
        except Exception as e:
            logger.error(f"Error starting ALE battle {battle_id}: {e}")
    
    async def handle_keyboard_input(self, battle_id: str, data: Dict):
        """Handle keyboard input for human player"""
        if battle_id in self.active_battles:
            battle = self.active_battles[battle_id]
            key_code = data.get("key_code")
            key_pressed = data.get("pressed", True)
            
            await battle.handle_keyboard_input(key_code, key_pressed)
    
    async def end_battle(self, battle_id: str):
        """End a battle"""
        if battle_id in self.active_battles:
            battle = self.active_battles[battle_id]
            await battle.end()
            del self.active_battles[battle_id]
    
    async def send_battle_update(self, battle_id: str, data: Dict):
        """Send battle update to connected clients"""
        if battle_id in self.websocket_connections:
            try:
                websocket = self.websocket_connections[battle_id]
                await websocket.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Error sending ALE battle update: {e}")

class ALEBattle:
    def __init__(self, battle_id: str, opponent_model_path: str, env_id: str, battle_system: ALEBattleSystem):
        self.battle_id = battle_id
        self.opponent_model_path = opponent_model_path
        self.env_id = env_id
        self.battle_system = battle_system
        
        self.running = False
        self.human_score = 0
        self.ai_score = 0
        self.game_duration = 0
        self.start_time = None
        
        # Keyboard input handling
        self.key_pressed = None
        self.key_mappings = self.get_key_mappings(env_id)
    
    def get_key_mappings(self, env_id: str) -> Dict:
        """Get keyboard mappings for different ALE environments"""
        mappings = {
            "boxing": {
                ord('w'): 0,  # Move up
                ord('s'): 1,  # Move down
                ord('a'): 2,  # Move left
                ord('d'): 3,  # Move right
                ord('f'): 4,  # Punch
                ord('g'): 5,  # Block
            },
            "pong": {
                ord('w'): 2,  # Move up
                ord('s'): 3,  # Move down
                ord('f'): 1,  # Fire
                ord('d'): 4,  # Fire right
                ord('a'): 5,  # Fire left
            },
            "tennis": {
                ord('w'): 0,  # Move up
                ord('s'): 1,  # Move down
                ord('a'): 2,  # Move left
                ord('d'): 3,  # Move right
                ord('f'): 4,  # Hit ball
            },
            "volleyball": {
                ord('w'): 0,  # Move up
                ord('s'): 1,  # Move down
                ord('a'): 2,  # Move left
                ord('d'): 3,  # Move right
                ord('f'): 4,  # Jump/spike
                ord('g'): 5,  # Block
            },
            "warlords": {
                ord('w'): 0,  # Move up
                ord('s'): 1,  # Move down
                ord('a'): 2,  # Move left
                ord('d'): 3,  # Move right
                ord('f'): 4,  # Fire
            },
            "joust": {
                ord('w'): 0,  # Fly up
                ord('s'): 1,  # Fly down
                ord('a'): 2,  # Move left
                ord('d'): 3,  # Move right
                ord('f'): 4,  # Attack
            },
            "combat": {
                ord('w'): 0,  # Move forward
                ord('s'): 1,  # Move backward
                ord('a'): 2,  # Turn left
                ord('d'): 3,  # Turn right
                ord('f'): 4,  # Fire
                ord('g'): 5,  # Change weapon
            }
        }
        return mappings.get(env_id, {})
    
    async def run(self):
        """Run the ALE battle"""
        try:
            await self.battle_system.send_battle_update(self.battle_id, {
                "type": "battle_started",
                "battle_id": self.battle_id,
                "env_id": self.env_id,
                "message": f"Starting {self.env_id} battle..."
            })
            
            self.running = True
            self.start_time = time.time()
            
            # Start the ALE game process
            await self.start_ale_game()
            
        except Exception as e:
            logger.error(f"Error in ALE battle {self.battle_id}: {e}")
            await self.battle_system.send_battle_update(self.battle_id, {
                "type": "battle_error",
                "error": str(e)
            })
    
    async def start_ale_game(self):
        """Start the ALE game process"""
        try:
            # Create the play script with the opponent model
            play_script = self.create_play_script()
            
            # Save script to temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(play_script)
                script_path = f.name
            
            # Run the game process
            cmd = ["python", script_path, self.opponent_model_path, self.env_id]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # Monitor the process
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=300)
            
            # Clean up
            os.unlink(script_path)
            
            # Parse results
            if process.returncode == 0:
                try:
                    result = json.loads(stdout.decode().strip())
                    self.human_score = result.get("human_score", 0)
                    self.ai_score = result.get("ai_score", 0)
                    self.game_duration = time.time() - self.start_time
                    
                    await self.finish_battle()
                except json.JSONDecodeError:
                    logger.error("Failed to parse game results")
                    await self.finish_battle()
            else:
                logger.error(f"Game process failed: {stderr.decode()}")
                await self.battle_system.send_battle_update(self.battle_id, {
                    "type": "battle_error",
                    "error": "Game process failed"
                })
                
        except Exception as e:
            logger.error(f"Error starting ALE game: {e}")
            raise
    
    def create_play_script(self) -> str:
        """Create the ALE play script with keyboard controls"""
        return f'''
import torch
import numpy as np
import cv2
import sys
import time
import json
import ale_py
from ale_py import ALEInterface
import supersuit as ss
from pettingzoo.atari import {self.env_id}_v3

# Import your Agent class (assuming it's available)
try:
    from play import Agent
except ImportError:
    # Fallback agent class
    class Agent:
        def __init__(self, action_space):
            self.action_space = action_space
        
        def load_state_dict(self, state_dict):
            pass
        
        def eval(self):
            pass
        
        def get_action(self, obs, deterministic=True):
            return torch.tensor([np.random.randint(0, self.action_space)]), None, None

def load_agent(model_path, action_space):
    """Load the AI agent"""
    try:
        agent = Agent(action_space)
        state_dict = torch.load(model_path, map_location="cpu")
        if 'model_state_dict' in state_dict:
            agent.load_state_dict(state_dict['model_state_dict'])
        else:
            agent.load_state_dict(state_dict)
        agent.eval()
        return agent
    except Exception as e:
        print(f"Error loading agent: {{e}}")
        return Agent(action_space)

def preprocess_obs(obs):
    """Preprocess observation"""
    if isinstance(obs, np.ndarray):
        if obs.shape[-1] != 6:
            obs = np.repeat(obs, 6 // obs.shape[-1], axis=-1)
        obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
    return obs

def get_keyboard_action(key, env_id):
    """Map keyboard input to action"""
    key_mappings = {{
        "boxing": {{
            ord('w'): 0, ord('s'): 1, ord('a'): 2, ord('d'): 3,
            ord('f'): 4, ord('g'): 5
        }},
        "pong": {{
            ord('w'): 2, ord('s'): 3, ord('f'): 1,
            ord('d'): 4, ord('a'): 5
        }},
        "tennis": {{
            ord('w'): 0, ord('s'): 1, ord('a'): 2, ord('d'): 3, ord('f'): 4
        }},
        "volleyball": {{
            ord('w'): 0, ord('s'): 1, ord('a'): 2, ord('d'): 3,
            ord('f'): 4, ord('g'): 5
        }},
        "warlords": {{
            ord('w'): 0, ord('s'): 1, ord('a'): 2, ord('d'): 3, ord('f'): 4
        }},
        "joust": {{
            ord('w'): 0, ord('s'): 1, ord('a'): 2, ord('d'): 3, ord('f'): 4
        }},
        "combat": {{
            ord('w'): 0, ord('s'): 1, ord('a'): 2, ord('d'): 3,
            ord('f'): 4, ord('g'): 5
        }}
    }}
    
    mappings = key_mappings.get(env_id, {{}})
    return mappings.get(key, 0)

def play_ale_game(model_path, env_id):
    """Main game loop for ALE environment"""
    try:
        # Initialize environment
        env = {self.env_id}_v3.env(render_mode="rgb_array")
        env = ss.max_observation_v0(env, 2)
        env = ss.frame_skip_v0(env, 4)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)
        env = ss.agent_indicator_v0(env, type_only=False)
        
        possible_agents = env.possible_agents
        ai_agent_name = possible_agents[0]  # AI plays first
        human_agent_name = possible_agents[1]  # Human plays second
        
        env.reset(seed=42)
        obs, _, _, _, _ = env.last()
        action_space = env.action_space(ai_agent_name).n
        
        # Load AI agent
        agent = load_agent(model_path, action_space)
        
        # Game variables
        human_score = 0
        ai_score = 0
        game_frames = 0
        max_frames = 10000  # Prevent infinite games
        
        # Create window for display
        cv2.namedWindow(f"{{env_id.title()}} - You vs AI", cv2.WINDOW_NORMAL)
        cv2.resizeWindow(f"{{env_id.title()}} - You vs AI", 800, 600)
        
        # Display controls
        controls = f"Controls: W=Up, S=Down, A=Left, D=Right, F=Action, Q=Quit"
        print(controls)
        
        done = False
        while not done and game_frames < max_frames:
            for agent_name in env.agent_iter():
                obs, reward, terminated, truncated, info = env.last()
                done = terminated or truncated
                
                if done:
                    env.step(None)
                    continue
                
                if agent_name == ai_agent_name:
                    # AI agent's turn
                    with torch.no_grad():
                        action, _, _ = agent.get_action(obs, deterministic=True)
                        action = action.cpu().item()
                    ai_score += reward if reward else 0
                else:
                    # Human agent's turn
                    frame = env.render()
                    
                    # Display controls on frame
                    cv2.putText(frame, controls, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    cv2.putText(frame, f"AI Score: {{ai_score}} | Your Score: {{human_score}}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                    
                    cv2.imshow(f"{{env_id.title()}} - You vs AI", frame)
                    key = cv2.waitKey(50) & 0xFF
                    
                    if key == ord('q'):
                        print("Game ended by user")
                        cv2.destroyAllWindows()
                        done = True
                        break
                    
                    action = get_keyboard_action(key, "{self.env_id}")
                    human_score += reward if reward else 0
                
                env.step(action)
                game_frames += 1
                
                if done:
                    break
        
        cv2.destroyAllWindows()
        env.close()
        
        # Return results
        return {{
            "human_score": human_score,
            "ai_score": ai_score,
            "winner": "human" if human_score > ai_score else "ai" if ai_score > human_score else "draw",
            "game_frames": game_frames
        }}
        
    except Exception as e:
        print(f"Error in game: {{e}}")
        return {{
            "human_score": 0,
            "ai_score": 0,
            "winner": "error",
            "error": str(e)
        }}

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python script.py <model_path> <env_id>")
        sys.exit(1)
    
    model_path = sys.argv[1]
    env_id = sys.argv[2]
    
    result = play_ale_game(model_path, env_id)
    print(json.dumps(result))
'''
    
    async def handle_keyboard_input(self, key_code: int, pressed: bool):
        """Handle keyboard input from WebSocket"""
        if pressed:
            self.key_pressed = key_code
        else:
            self.key_pressed = None
    
    async def end(self):
        """End the battle"""
        self.running = False
    
    async def finish_battle(self):
        """Finish the battle and send results"""
        try:
            # Determine winner
            if self.human_score > self.ai_score:
                winner = "human"
            elif self.ai_score > self.human_score:
                winner = "ai"
            else:
                winner = "draw"
            
            result = {{
                "type": "battle_finished",
                "battle_id": self.battle_id,
                "env_id": self.env_id,
                "human_score": self.human_score,
                "ai_score": self.ai_score,
                "winner": winner,
                "game_duration": self.game_duration,
                "message": f"Game Over! {{winner.title()}} wins!"
            }}
            
            await self.battle_system.send_battle_update(self.battle_id, result)
            
            logger.info(f"ALE Battle {{self.battle_id}} finished. Winner: {{winner}}")
            
        except Exception as e:
            logger.error(f"Error finishing ALE battle {{self.battle_id}}: {{e}}")

async def main():
    """Main entry point"""
    ale_system = ALEBattleSystem()
    
    try:
        await ale_system.start_server(port=8766)
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        ale_system.running = False

if __name__ == "__main__":
    asyncio.run(main())
