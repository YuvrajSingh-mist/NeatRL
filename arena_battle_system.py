#!/usr/bin/env python3
"""
NeatRL Arena Battle System - Simplified Version
Handles real-time battles between RL models
"""

import asyncio
import json
import logging
import os
import sys
import time
import uuid
from typing import Dict, List, Optional
import websockets
from websockets.server import serve
import gymnasium as gym
import numpy as np
import subprocess
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ArenaBattleSystem:
    def __init__(self):
        self.active_battles: Dict[str, 'Battle'] = {}
        self.websocket_connections: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.running = True
    
    async def start_server(self, host: str = "0.0.0.0", port: int = 8765):
        """Start the WebSocket server"""
        logger.info(f"Starting Arena Battle System on {host}:{port}")
        
        async with serve(self.handle_websocket, host, port):
            await asyncio.Future()
    
    async def handle_websocket(self, websocket, path):
        """Handle WebSocket connections"""
        try:
            battle_id = path.strip('/')
            if not battle_id:
                await websocket.close(1008, "Invalid battle ID")
                return
            
            self.websocket_connections[battle_id] = websocket
            logger.info(f"WebSocket connected for battle {battle_id}")
            
            await websocket.send(json.dumps({
                "type": "connection_established",
                "battle_id": battle_id
            }))
            
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self.handle_battle_message(battle_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON received")
                    
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"WebSocket disconnected for battle {battle_id}")
        finally:
            if battle_id in self.websocket_connections:
                del self.websocket_connections[battle_id]
    
    async def handle_battle_message(self, battle_id: str, data: Dict):
        """Handle battle messages"""
        message_type = data.get("type")
        
        if message_type == "start_battle":
            await self.start_battle(battle_id, data)
        elif message_type == "end_battle":
            await self.end_battle(battle_id)
    
    async def start_battle(self, battle_id: str, data: Dict):
        """Start a new battle"""
        try:
            player_model_path = data.get("player_model_path")
            opponent_model_path = data.get("opponent_model_path")
            env_id = data.get("env_id", "CartPole-v1")
            
            battle = Battle(
                battle_id=battle_id,
                player_model_path=player_model_path,
                opponent_model_path=opponent_model_path,
                env_id=env_id,
                arena_system=self
            )
            
            self.active_battles[battle_id] = battle
            asyncio.create_task(battle.run())
            
            logger.info(f"Battle {battle_id} started")
            
        except Exception as e:
            logger.error(f"Error starting battle {battle_id}: {e}")
    
    async def end_battle(self, battle_id: str):
        """End a battle"""
        if battle_id in self.active_battles:
            battle = self.active_battles[battle_id]
            await battle.end()
            del self.active_battles[battle_id]
    
    async def send_battle_update(self, battle_id: str, data: Dict):
        """Send battle update"""
        if battle_id in self.websocket_connections:
            try:
                websocket = self.websocket_connections[battle_id]
                await websocket.send(json.dumps(data))
            except Exception as e:
                logger.error(f"Error sending battle update: {e}")

class Battle:
    def __init__(self, battle_id: str, player_model_path: str, opponent_model_path: str, 
                 env_id: str, arena_system: ArenaBattleSystem):
        self.battle_id = battle_id
        self.player_model_path = player_model_path
        self.opponent_model_path = opponent_model_path
        self.env_id = env_id
        self.arena_system = arena_system
        
        self.env = None
        self.running = False
        self.player_score = 0.0
        self.opponent_score = 0.0
        self.current_step = 0
        self.max_steps = 500
    
    async def run(self):
        """Run the battle"""
        try:
            await self.arena_system.send_battle_update(self.battle_id, {
                "type": "battle_started",
                "battle_id": self.battle_id,
                "env_id": self.env_id
            })
            
            # Initialize environment
            self.env = gym.make(self.env_id)
            observation, info = self.env.reset()
            
            self.running = True
            
            # Battle loop
            while self.running and self.current_step < self.max_steps:
                # Get actions from both models
                player_action = await self.get_model_action(self.player_model_path, observation)
                opponent_action = await self.get_model_action(self.opponent_model_path, observation)
                
                # Execute step (use player action for environment)
                observation, reward, done, truncated, info = self.env.step(player_action)
                
                # Update scores
                self.player_score += reward
                self.opponent_score += -reward
                
                self.current_step += 1
                
                # Send update every 10 steps
                if self.current_step % 10 == 0:
                    await self.arena_system.send_battle_update(self.battle_id, {
                        "type": "battle_step",
                        "step": self.current_step,
                        "player_score": self.player_score,
                        "opponent_score": self.opponent_score
                    })
                
                if done:
                    break
                
                await asyncio.sleep(0.1)  # 10 FPS
            
            # Finish battle
            await self.finish_battle()
            
        except Exception as e:
            logger.error(f"Error in battle {self.battle_id}: {e}")
    
    async def get_model_action(self, model_path: str, observation) -> int:
        """Get action from model"""
        try:
            # Create temporary file to run model
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                with open(model_path, 'r') as source:
                    f.write(source.read())
                temp_path = f.name
            
            # Run model
            cmd = ["python", temp_path, self.env_id]
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=5.0)
            
            # Clean up
            os.unlink(temp_path)
            
            if process.returncode == 0:
                result = json.loads(stdout.decode().strip())
                return int(result.get("action", 0))
            else:
                return 0
                
        except Exception as e:
            logger.error(f"Error getting model action: {e}")
            return 0
    
    async def end(self):
        """End the battle"""
        self.running = False
        if self.env:
            self.env.close()
    
    async def finish_battle(self):
        """Finish the battle"""
        try:
            # Determine winner
            if self.player_score > self.opponent_score:
                winner = "player"
            elif self.opponent_score > self.player_score:
                winner = "opponent"
            else:
                winner = "draw"
            
            await self.arena_system.send_battle_update(self.battle_id, {
                "type": "battle_finished",
                "player_score": self.player_score,
                "opponent_score": self.opponent_score,
                "winner": winner,
                "total_steps": self.current_step
            })
            
            logger.info(f"Battle {self.battle_id} finished. Winner: {winner}")
            
        except Exception as e:
            logger.error(f"Error finishing battle {self.battle_id}: {e}")

async def main():
    """Main entry point"""
    arena_system = ArenaBattleSystem()
    
    try:
        await arena_system.start_server()
    except KeyboardInterrupt:
        logger.info("Shutdown requested")
    finally:
        arena_system.running = False

if __name__ == "__main__":
    asyncio.run(main())
