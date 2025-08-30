#!/usr/bin/env python3
"""
PettingZoo Atari Play Script for Human vs AI Battles
This script handles human vs AI gameplay in PettingZoo Atari environments using AEC
"""

import torch
import numpy as np
import cv2
import sys
import time
import json
import os
from typing import Dict, Optional

# Try to import the Agent class
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

def load_agent(model_path: str, action_space: int) -> Agent:
    """Load the AI agent from model file"""
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
        print(f"Error loading agent: {e}")
        return Agent(action_space)

def get_keyboard_mappings(env_id: str) -> Dict:
    """Get keyboard mappings for PettingZoo Atari environments"""
    
    # PettingZoo Atari Action Space Reference (0-17)
    ATARI_ACTIONS = {
        0: "NOOP",
        1: "FIRE", 
        2: "UP",
        3: "RIGHT",
        4: "LEFT", 
        5: "DOWN",
        6: "UPRIGHT",
        7: "UPLEFT",
        8: "DOWNRIGHT", 
        9: "DOWNLEFT",
        10: "UPFIRE",
        11: "RIGHTFIRE",
        12: "LEFTFIRE",
        13: "DOWNFIRE",
        14: "UPRIGHTFIRE",
        15: "UPLEFTFIRE",
        16: "DOWNRIGHTFIRE",
        17: "DOWNLEFTFIRE"
    }
    
    # Environment-specific keyboard mappings for PettingZoo Atari
    env_mappings = {
        "pong_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('f'): 1,    # FIRE
            ord('d'): 11,   # RIGHTFIRE
            ord('a'): 12,   # LEFTFIRE
            ord('q'): 0,    # NOOP
        },
        "boxing_v2": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE (punch)
            ord('g'): 10,   # UPFIRE (block)
            ord('q'): 0,    # NOOP
        },
        "tennis_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE (hit ball)
            ord('q'): 0,    # NOOP
        },
        "ice_hockey_v2": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE (shoot)
            ord('g'): 10,   # UPFIRE (pass)
            ord('q'): 0,    # NOOP
        },
        "double_dunk_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE (shoot)
            ord('g'): 10,   # UPFIRE (pass)
            ord('q'): 0,    # NOOP
        },
        "wizard_of_wor_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('g'): 10,   # UPFIRE (special)
            ord('q'): 0,    # NOOP
        },
        "space_invaders_v2": {
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "basketball_pong_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('f'): 1,    # FIRE
            ord('d'): 11,   # RIGHTFIRE
            ord('a'): 12,   # LEFTFIRE
            ord('q'): 0,    # NOOP
        },
        "volleyball_pong_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('f'): 1,    # FIRE
            ord('d'): 11,   # RIGHTFIRE
            ord('a'): 12,   # LEFTFIRE
            ord('q'): 0,    # NOOP
        },
        "foozpong_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('f'): 1,    # FIRE
            ord('d'): 11,   # RIGHTFIRE
            ord('a'): 12,   # LEFTFIRE
            ord('q'): 0,    # NOOP
        },
        "quadrapong_v4": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('f'): 1,    # FIRE
            ord('d'): 11,   # RIGHTFIRE
            ord('a'): 12,   # LEFTFIRE
            ord('q'): 0,    # NOOP
        },
        "joust_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "mario_bros_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "maze_craze_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "othello_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "video_checkers_v4": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "warlords_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "combat_plane_v2": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "combat_tank_v2": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "flag_capture_v2": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "surround_v2": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "space_war_v2": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "entombed_competitive_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        },
        "entombed_cooperative_v3": {
            ord('w'): 2,    # UP
            ord('s'): 5,    # DOWN
            ord('a'): 4,    # LEFT
            ord('d'): 3,    # RIGHT
            ord('f'): 1,    # FIRE
            ord('q'): 0,    # NOOP
        }
    }
    
    return env_mappings.get(env_id, {
        ord('w'): 2,    # UP
        ord('s'): 5,    # DOWN
        ord('a'): 4,    # LEFT
        ord('d'): 3,    # RIGHT
        ord('f'): 1,    # FIRE
        ord('q'): 0,    # NOOP
    })

def get_environment_info(env_id: str) -> Dict:
    """Get environment-specific information for PettingZoo Atari"""
    env_info = {
        "pong_v3": {
            "name": "Pong",
            "description": "Classic Pong - Move your paddle to hit the ball",
            "controls": "W=Up, S=Down, F=Fire, D=Fire Right, A=Fire Left"
        },
        "boxing_v2": {
            "name": "Boxing",
            "description": "Punch and move to defeat your opponent",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Punch, G=Block"
        },
        "tennis_v3": {
            "name": "Tennis",
            "description": "Hit the ball over the net",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Hit Ball"
        },
        "ice_hockey_v2": {
            "name": "Ice Hockey",
            "description": "Fast-paced hockey game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Shoot, G=Pass"
        },
        "double_dunk_v3": {
            "name": "Double Dunk",
            "description": "Basketball game with two hoops",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Shoot, G=Pass"
        },
        "wizard_of_wor_v3": {
            "name": "Wizard of Wor",
            "description": "Maze-based action game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire, G=Special"
        },
        "space_invaders_v2": {
            "name": "Space Invaders",
            "description": "Classic arcade shooter",
            "controls": "A=Left, D=Right, F=Fire"
        },
        "basketball_pong_v3": {
            "name": "Basketball Pong",
            "description": "Basketball version of Pong",
            "controls": "W=Up, S=Down, F=Fire, D=Fire Right, A=Fire Left"
        },
        "volleyball_pong_v3": {
            "name": "Volleyball Pong",
            "description": "Volleyball version of Pong",
            "controls": "W=Up, S=Down, F=Fire, D=Fire Right, A=Fire Left"
        },
        "foozpong_v3": {
            "name": "Foozpong",
            "description": "Football version of Pong",
            "controls": "W=Up, S=Down, F=Fire, D=Fire Right, A=Fire Left"
        },
        "quadrapong_v4": {
            "name": "Quadrapong",
            "description": "Four-player Pong",
            "controls": "W=Up, S=Down, F=Fire, D=Fire Right, A=Fire Left"
        },
        "joust_v3": {
            "name": "Joust",
            "description": "Medieval flying combat",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "mario_bros_v3": {
            "name": "Mario Bros",
            "description": "Classic platformer",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "maze_craze_v3": {
            "name": "Maze Craze",
            "description": "Navigate through mazes",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "othello_v3": {
            "name": "Othello",
            "description": "Classic board game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "video_checkers_v4": {
            "name": "Video Checkers",
            "description": "Classic checkers game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "warlords_v3": {
            "name": "Warlords",
            "description": "Castle defense game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "combat_plane_v2": {
            "name": "Combat: Plane",
            "description": "Aerial combat simulation",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "combat_tank_v2": {
            "name": "Combat: Tank",
            "description": "Tank warfare game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "flag_capture_v2": {
            "name": "Flag Capture",
            "description": "Capture the flag game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "surround_v2": {
            "name": "Surround",
            "description": "Snake-like game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "space_war_v2": {
            "name": "Space War",
            "description": "Space combat game",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "entombed_competitive_v3": {
            "name": "Entombed: Competitive",
            "description": "Competitive maze exploration",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        },
        "entombed_cooperative_v3": {
            "name": "Entombed: Cooperative",
            "description": "Cooperative maze exploration",
            "controls": "W=Up, S=Down, A=Left, D=Right, F=Fire"
        }
    }
    return env_info.get(env_id, {
        "name": env_id.replace("_v3", "").title(),
        "description": "PettingZoo Atari Environment",
        "controls": "WASD=Movement, F=Action"
    })

def play_pettingzoo_game(model_path: str, env_id: str) -> Dict:
    """Main game loop for PettingZoo Atari environment using AEC"""
    try:
        # Import PettingZoo Atari environments
        import pettingzoo.atari
        import supersuit as ss
        
        # Dynamic import of the environment
        env_module = getattr(pettingzoo.atari, env_id)
        env = env_module.env(render_mode="rgb_array")
        
        # Apply supersuit wrappers for preprocessing
        env = ss.max_observation_v0(env, 2)
        env = ss.frame_skip_v0(env, 4)
        env = ss.color_reduction_v0(env, mode="B")
        env = ss.resize_v1(env, x_size=84, y_size=84)
        env = ss.frame_stack_v1(env, 4)
        env = ss.agent_indicator_v0(env, type_only=False)
        
        # Reset environment
        env.reset(seed=42)
        
        # Get agent names
        possible_agents = env.possible_agents
        ai_agent_name = possible_agents[0]  # AI plays first agent
        human_agent_name = possible_agents[1] if len(possible_agents) > 1 else possible_agents[0]  # Human plays second agent
        
        # Load AI agent
        action_space = env.action_space(ai_agent_name).n
        agent = load_agent(model_path, action_space)
        
        # Game variables
        human_score = 0
        ai_score = 0
        game_frames = 0
        max_frames = 10000  # Prevent infinite games
        
        # Get environment info
        env_info = get_environment_info(env_id)
        key_mappings = get_keyboard_mappings(env_id)
        
        # Create window for display with optimized settings
        window_name = f"{env_info['name']} - You vs AI (PettingZoo)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        # Optimize for low latency
        cv2.setWindowProperty(window_name, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        cv2.setWindowProperty(window_name, cv2.WND_PROP_TOPMOST, 1)
        
        # Display game info
        print(f"Playing {env_info['name']} (PettingZoo)")
        print(f"Description: {env_info['description']}")
        print(f"Controls: {env_info['controls']}")
        print(f"AI Agent: {ai_agent_name}")
        print(f"Human Agent: {human_agent_name}")
        print("Press Q to quit")
        print("-" * 50)
        
        # AEC Game Loop (Agent Environment Cycle)
        for agent_name in env.agent_iter():
            # Get current observation, reward, termination, truncation, info
            obs, reward, terminated, truncated, info = env.last()
            
            # Check if game is done
            if terminated or truncated:
                env.step(None)  # Required for AEC
                continue
            
            # Render frame
            frame = env.render()
            
            # Display info on frame
            cv2.putText(frame, f"AI ({ai_agent_name}): {ai_score} | You ({human_agent_name}): {human_score}", 
                       (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(frame, f"Current Agent: {agent_name}", 
                       (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            cv2.putText(frame, "Q=Quit", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)
            
            cv2.imshow(window_name, frame)
            
            # Get human input (real-time) - optimized for low latency
            key = cv2.waitKey(1) & 0xFF  # 1ms delay for maximum responsiveness
            
            # Handle quit
            if key == ord('q'):
                print("Game ended by user")
                cv2.destroyAllWindows()
                break
            
            # Determine action based on current agent
            if agent_name == ai_agent_name:
                # AI agent's turn
                with torch.no_grad():
                    action, _, _ = agent.get_action(obs, deterministic=True)
                    action = action.cpu().item()
                ai_score += reward if reward else 0
            else:
                # Human agent's turn
                if key in key_mappings:
                    action = key_mappings[key]
                else:
                    action = 0  # NOOP if no key pressed
                human_score += reward if reward else 0
            
            # Take step in environment (AEC handles the flow)
            env.step(action)
            
            game_frames += 1
            if game_frames >= max_frames:
                break
        
        cv2.destroyAllWindows()
        env.close()
        
        # Determine winner
        if human_score > ai_score:
            winner = "human"
        elif ai_score > human_score:
            winner = "ai"
        else:
            winner = "draw"
        
        # Return results
        return {
            "human_score": human_score,
            "ai_score": ai_score,
            "winner": winner,
            "game_frames": game_frames,
            "environment": env_id,
            "environment_name": env_info["name"],
            "ai_agent": ai_agent_name,
            "human_agent": human_agent_name
        }
        
    except Exception as e:
        print(f"Error in game: {e}")
        return {
            "human_score": 0,
            "ai_score": 0,
            "winner": "error",
            "error": str(e),
            "environment": env_id
        }

def main():
    """Main function"""
    if len(sys.argv) < 3:
        print("Usage: python ale_play_script.py <model_path> <env_id>")
        print("Supported PettingZoo Atari environments:")
        supported_envs = [
            "pong_v3", "boxing_v2", "tennis_v3", "ice_hockey_v2", "double_dunk_v3", 
            "wizard_of_wor_v3", "space_invaders_v2", "basketball_pong_v3", "volleyball_pong_v3",
            "foozpong_v3", "quadrapong_v4", "joust_v3", "mario_bros_v3", "maze_craze_v3",
            "othello_v3", "video_checkers_v4", "warlords_v3", "combat_plane_v2", "combat_tank_v2",
            "flag_capture_v2", "surround_v2", "space_war_v2", "entombed_competitive_v3", "entombed_cooperative_v3"
        ]
        for env in supported_envs:
            print(f"  - {env}")
        sys.exit(1)
    
    model_path = sys.argv[1]
    env_id = sys.argv[2]
    
    # Validate environment
    supported_envs = [
        "pong_v3", "boxing_v2", "tennis_v3", "ice_hockey_v2", "double_dunk_v3", 
        "wizard_of_wor_v3", "space_invaders_v2", "basketball_pong_v3", "volleyball_pong_v3",
        "foozpong_v3", "quadrapong_v4", "joust_v3", "mario_bros_v3", "maze_craze_v3",
        "othello_v3", "video_checkers_v4", "warlords_v3", "combat_plane_v2", "combat_tank_v2",
        "flag_capture_v2", "surround_v2", "space_war_v2", "entombed_competitive_v3", "entombed_cooperative_v3"
    ]
    if env_id not in supported_envs:
        print(f"Unsupported environment: {env_id}")
        print(f"Supported: {supported_envs}")
        sys.exit(1)
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        sys.exit(1)
    
    # Play the game
    result = play_pettingzoo_game(model_path, env_id)
    
    # Print results
    print("\n" + "="*50)
    print("GAME OVER!")
    print("="*50)
    print(f"Environment: {result.get('environment_name', env_id)}")
    print(f"Your Score: {result['human_score']}")
    print(f"AI Score: {result['ai_score']}")
    print(f"Winner: {result['winner'].upper()}")
    print(f"Game Frames: {result['game_frames']}")
    
    if 'error' in result:
        print(f"Error: {result['error']}")
    
    # Output JSON for programmatic use
    print(json.dumps(result))

if __name__ == "__main__":
    main()

