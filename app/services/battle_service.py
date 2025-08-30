import asyncio
import json
import logging
import subprocess
import tempfile
import os
from typing import Dict, Optional, Tuple
from sqlalchemy.orm import Session
from app.models.submission import Battle, BattleResult, UserModel
from app.core.supabase import supabase_client
from app.core.config import settings
from app.api.rlarena import update_battle_status
import uuid

logger = logging.getLogger(__name__)

class BattleService:
    def __init__(self):
        self.temp_dir = tempfile.mkdtemp()
    
    async def execute_battle(self, battle_id: str, db: Session) -> Dict:
        """Execute a battle between two models"""
        try:
            # Get battle details
            battle = db.query(Battle).filter(Battle.id == battle_id).first()
            if not battle:
                raise Exception("Battle not found")
            
            # Update status to processing
            battle.status = "processing"
            db.commit()
            
            # Notify via WebSocket
            await update_battle_status(battle_id, "processing", db=db)
            
            # Get model details
            player_model = db.query(UserModel).filter(UserModel.id == battle.player_model_id).first()
            opponent_model = db.query(UserModel).filter(UserModel.id == battle.opponent_model_id).first()
            
            if not player_model or not opponent_model:
                raise Exception("One or both models not found")
            
            # Download model files
            player_file = await self._download_model_file(player_model.file_path)
            opponent_file = await self._download_model_file(opponent_model.file_path)
            
            # Execute battle
            result = await self._run_battle(
                battle_id, 
                player_file, 
                opponent_file, 
                battle.env_id,
                player_model,
                opponent_model
            )
            
            # Save battle result
            battle_result = BattleResult(
                id=str(uuid.uuid4()),
                battle_id=battle_id,
                player_score=result["player_score"],
                opponent_score=result["opponent_score"],
                winner=result["winner"],
                duration_seconds=result["duration_seconds"],
                battle_log=json.dumps(result["battle_log"])
            )
            
            db.add(battle_result)
            
            # Update model statistics
            await self._update_model_stats(
                player_model, opponent_model, result["winner"], db
            )
            
            # Update battle status
            battle.status = "completed"
            db.commit()
            
            # Notify via WebSocket
            await update_battle_status(battle_id, "completed", result, db=db)
            
            logger.info(f"Battle {battle_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error executing battle {battle_id}: {str(e)}")
            
            # Update battle status to failed
            if battle:
                battle.status = "failed"
                db.commit()
                await update_battle_status(battle_id, "failed", {"error": str(e)}, db=db)
            
            raise
    
    async def _download_model_file(self, file_path: str) -> str:
        """Download model file from Supabase Storage"""
        try:
            # Download file from Supabase
            response = supabase_client.storage.from_(settings.supabase_bucket).download(file_path)
            
            # Save to temporary file
            temp_file = os.path.join(self.temp_dir, f"{uuid.uuid4()}.py")
            with open(temp_file, 'wb') as f:
                f.write(response)
            
            return temp_file
            
        except Exception as e:
            logger.error(f"Error downloading model file {file_path}: {str(e)}")
            raise
    
    async def _run_battle(
        self, 
        battle_id: str, 
        player_file: str, 
        opponent_file: str, 
        env_id: str,
        player_model: UserModel,
        opponent_model: UserModel
    ) -> Dict:
        """Run the actual battle between two models"""
        try:
            # Create a simple battle environment
            # This is a simplified implementation - in a real scenario, you'd want
            # a more sophisticated battle system that can handle different environments
            
            # For now, we'll simulate a battle by running both models and comparing scores
            player_score = await self._evaluate_model(player_file, env_id, "player")
            opponent_score = await self._evaluate_model(opponent_file, env_id, "opponent")
            
            # Determine winner
            if player_score > opponent_score:
                winner = "player"
            elif opponent_score > player_score:
                winner = "opponent"
            else:
                winner = "draw"
            
            # Create battle log
            battle_log = {
                "battle_id": battle_id,
                "environment": env_id,
                "player_model": {
                    "id": player_model.id,
                    "name": player_model.model_name,
                    "algorithm": player_model.algorithm,
                    "user_id": player_model.user_id,
                    "score": player_score
                },
                "opponent_model": {
                    "id": opponent_model.id,
                    "name": opponent_model.model_name,
                    "algorithm": opponent_model.algorithm,
                    "user_id": opponent_model.user_id,
                    "score": opponent_score
                },
                "winner": winner,
                "timestamp": str(uuid.uuid4())
            }
            
            return {
                "player_score": player_score,
                "opponent_score": opponent_score,
                "winner": winner,
                "duration_seconds": 30.0,  # Simulated duration
                "battle_log": battle_log
            }
            
        except Exception as e:
            logger.error(f"Error running battle: {str(e)}")
            raise
    
    async def _evaluate_model(self, model_file: str, env_id: str, role: str) -> float:
        """Evaluate a single model and return its score"""
        try:
            # Run the model in a subprocess
            # This is a simplified evaluation - in production, you'd want proper isolation
            cmd = ["python", model_file, env_id]
            
            # Run with timeout
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=60)
            except asyncio.TimeoutError:
                process.kill()
                raise Exception(f"Model evaluation timed out for {role}")
            
            if process.returncode != 0:
                raise Exception(f"Model evaluation failed for {role}: {stderr.decode()}")
            
            # Parse the JSON output
            output = stdout.decode().strip()
            try:
                result = json.loads(output)
                return float(result.get("score", 0.0))
            except json.JSONDecodeError:
                raise Exception(f"Invalid JSON output from {role} model")
                
        except Exception as e:
            logger.error(f"Error evaluating model {role}: {str(e)}")
            # Return a default score on error
            return 0.0
    
    async def _update_model_stats(
        self, 
        player_model: UserModel, 
        opponent_model: UserModel, 
        winner: str, 
        db: Session
    ):
        """Update model statistics after a battle"""
        try:
            # Update player model stats
            player_model.total_battles += 1
            if winner == "player":
                player_model.wins += 1
            elif winner == "opponent":
                player_model.losses += 1
            else:  # draw
                player_model.draws += 1
            
            if player_model.total_battles > 0:
                player_model.win_rate = player_model.wins / player_model.total_battles
            
            # Update opponent model stats
            opponent_model.total_battles += 1
            if winner == "opponent":
                opponent_model.wins += 1
            elif winner == "player":
                opponent_model.losses += 1
            else:  # draw
                opponent_model.draws += 1
            
            if opponent_model.total_battles > 0:
                opponent_model.win_rate = opponent_model.wins / opponent_model.total_battles
            
            db.commit()
            
        except Exception as e:
            logger.error(f"Error updating model stats: {str(e)}")
            raise

# Global instance
battle_service = BattleService()
