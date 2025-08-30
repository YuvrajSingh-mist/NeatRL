import os
import subprocess
import tempfile
import shutil
import asyncio
import logging
from typing import Optional, Dict, Any
from pathlib import Path

logger = logging.getLogger(__name__)

class HumanVsAIService:
    """Service for handling human vs AI battles in the RL Arena"""
    
    def __init__(self, arena_container_name: str = "neatrl_arena_1"):
        self.arena_container_name = arena_container_name
        self.temp_dir = Path(tempfile.mkdtemp())
        
    async def execute_human_vs_ai_battle(
        self,
        env_id: str,
        model_path: str,
        opponent_model_id: Optional[str] = None,
        opponent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute a human vs AI battle in the arena container
        
        Args:
            env_id: PettingZoo Atari environment ID
            model_path: Path to the AI model file
            opponent_model_id: ID of the opponent model (optional)
            opponent_name: Name of the opponent (optional)
            
        Returns:
            Dictionary containing battle results
        """
        try:
            # Download model file to temp directory
            local_model_path = await self._download_model_file(model_path)
            
            # Execute the battle
            result = await self._run_pettingzoo_battle(
                env_id, local_model_path, opponent_model_id, opponent_name
            )
            
            # Cleanup
            await self.cleanup()
            
            return result
            
        except Exception as e:
            logger.error(f"Error in human vs AI battle: {e}")
            await self.cleanup()
            raise
    
    async def _download_model_file(self, model_path: str) -> str:
        """Download model file from storage to local temp directory"""
        try:
            # For now, assume model_path is a local path
            # In production, this would download from Supabase Storage
            if os.path.exists(model_path):
                local_path = self.temp_dir / "model.pth"
                shutil.copy2(model_path, local_path)
                return str(local_path)
            else:
                raise FileNotFoundError(f"Model file not found: {model_path}")
        except Exception as e:
            logger.error(f"Error downloading model file: {e}")
            raise
    
    async def _run_pettingzoo_battle(
        self,
        env_id: str,
        model_path: str,
        opponent_model_id: Optional[str] = None,
        opponent_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run the PettingZoo battle script in the arena container"""
        try:
            # Prepare command to run the battle script
            cmd = [
                "docker", "exec", self.arena_container_name,
                "python", "/app/ale_play_script.py",
                "--env_id", env_id,
                "--model_path", model_path
            ]
            
            if opponent_model_id:
                cmd.extend(["--opponent_model_id", opponent_model_id])
            if opponent_name:
                cmd.extend(["--opponent_name", opponent_name])
            
            # Execute the command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                logger.error(f"Battle script failed: {stderr.decode()}")
                raise RuntimeError(f"Battle script failed: {stderr.decode()}")
            
            # Parse results from stdout
            result = self._parse_battle_results(stdout.decode())
            
            return result
            
        except Exception as e:
            logger.error(f"Error running PettingZoo battle: {e}")
            raise
    
    def _parse_battle_results(self, output: str) -> Dict[str, Any]:
        """Parse battle results from script output"""
        # This is a simplified parser - in production, you'd want more robust parsing
        lines = output.strip().split('\n')
        result = {
            "ai_score": 0,
            "human_score": 0,
            "winner": "unknown",
            "duration": 0,
            "opponent_info": {}
        }
        
        for line in lines:
            if "AI Score:" in line:
                try:
                    result["ai_score"] = int(line.split(":")[1].strip())
                except:
                    pass
            elif "Human Score:" in line:
                try:
                    result["human_score"] = int(line.split(":")[1].strip())
                except:
                    pass
            elif "Winner:" in line:
                result["winner"] = line.split(":")[1].strip()
            elif "Duration:" in line:
                try:
                    result["duration"] = float(line.split(":")[1].strip())
                except:
                    pass
        
        return result
    
    async def cleanup(self):
        """Clean up temporary files and resources"""
        try:
            if self.temp_dir.exists():
                shutil.rmtree(self.temp_dir)
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    async def update_battle_results(
        self,
        battle_id: str,
        results: Dict[str, Any]
    ) -> bool:
        """Update battle results in the database"""
        try:
            # This would update the database with battle results
            # For now, just log the results
            logger.info(f"Battle {battle_id} completed: {results}")
            return True
        except Exception as e:
            logger.error(f"Error updating battle results: {e}")
            return False
