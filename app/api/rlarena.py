from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.submission import Battle, BattleResult, UserModel
from app.services.human_vs_ai_service import HumanVsAIService
from app.core.supabase import supabase_client
from app.core.config import settings
import uuid
import json
import logging
import asyncio
from typing import List, Optional, Dict
from datetime import datetime
import random

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rlarena", tags=["RL Arena"])

# Store active WebSocket connections for real-time battle updates
active_connections: Dict[str, WebSocket] = {}

@router.post("/battle")
async def create_battle(
    player_model_id: str,
    opponent_model_id: Optional[str] = None,
    env_id: str = "CartPole-v1",
    db: Session = Depends(get_db)
):
    """Create a new battle between two models"""
    try:
        # Get player model
        player_model = db.query(UserModel).filter(UserModel.id == player_model_id).first()
        if not player_model:
            raise HTTPException(status_code=404, detail="Player model not found")
        
        # If no opponent specified, select a random public model
        if not opponent_model_id:
            available_models = db.query(UserModel).filter(
                UserModel.is_public == True,
                UserModel.id != player_model_id,
                UserModel.env_id == env_id
            ).all()
            
            if not available_models:
                raise HTTPException(status_code=404, detail="No available opponent models found")
            
            opponent_model = random.choice(available_models)
            opponent_model_id = opponent_model.id
        else:
            opponent_model = db.query(UserModel).filter(UserModel.id == opponent_model_id).first()
            if not opponent_model:
                raise HTTPException(status_code=404, detail="Opponent model not found")
        
        # Create battle record
        battle_id = str(uuid.uuid4())
        battle = Battle(
            id=battle_id,
            player_model_id=player_model_id,
            opponent_model_id=opponent_model_id,
            env_id=env_id,
            status="pending"
        )
        
        db.add(battle)
        db.commit()
        db.refresh(battle)
        
        logger.info(f"Battle created: {battle_id} between {player_model_id} and {opponent_model_id}")
        
        return JSONResponse(content={
            "battle_id": battle_id,
            "player_model": {
                "id": player_model.id,
                "name": player_model.model_name,
                "algorithm": player_model.algorithm,
                "user_id": player_model.user_id
            },
            "opponent_model": {
                "id": opponent_model.id,
                "name": opponent_model.model_name,
                "algorithm": opponent_model.algorithm,
                "user_id": opponent_model.user_id
            },
            "env_id": env_id,
            "status": "pending"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating battle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create battle: {str(e)}")

@router.post("/human-vs-ai")
async def create_human_vs_ai_battle(
    opponent_model_id: Optional[str] = None,
    env_id: str = "Boxing-v4",
    db: Session = Depends(get_db)
):
    """Create a human vs AI battle in ALE environment"""
    try:
        # Validate PettingZoo Atari environment
        pettingzoo_envs = [
            "pong_v3", "boxing_v2", "tennis_v3", "ice_hockey_v2", "double_dunk_v3", 
            "wizard_of_wor_v3", "space_invaders_v2", "basketball_pong_v3", "volleyball_pong_v3",
            "foozpong_v3", "quadrapong_v4", "joust_v3", "mario_bros_v3", "maze_craze_v3",
            "othello_v3", "video_checkers_v4", "warlords_v3", "combat_plane_v2", "combat_tank_v2",
            "flag_capture_v2", "surround_v2", "space_war_v2", "entombed_competitive_v3", "entombed_cooperative_v3"
        ]
        if env_id not in pettingzoo_envs:
            raise HTTPException(status_code=400, detail=f"Invalid PettingZoo Atari environment. Supported: {pettingzoo_envs}")
        
        # If no opponent specified, select a random public model for this environment
        if not opponent_model_id:
            available_models = db.query(UserModel).filter(
                UserModel.is_public == True,
                UserModel.env_id == env_id
            ).all()
            
            if not available_models:
                raise HTTPException(status_code=404, detail=f"No available AI models found for {env_id}")
            
            opponent_model = random.choice(available_models)
            opponent_model_id = opponent_model.id
        else:
            opponent_model = db.query(UserModel).filter(UserModel.id == opponent_model_id).first()
            if not opponent_model:
                raise HTTPException(status_code=404, detail="Opponent model not found")
        
        # Create battle record for human vs AI
        battle_id = str(uuid.uuid4())
        battle = Battle(
            id=battle_id,
            player_model_id="human",  # Special identifier for human player
            opponent_model_id=opponent_model_id,
            env_id=env_id,
            status="pending"
        )
        
        db.add(battle)
        db.commit()
        db.refresh(battle)
        
        logger.info(f"Human vs AI battle created: {battle_id} in {env_id} against {opponent_model_id}")
        
        # Execute the battle using the service
        try:
            human_vs_ai_service = HumanVsAIService()
            
            # Get the model file path (this would come from the database in production)
            model_path = f"/app/models/{opponent_model_id}.pth"  # Placeholder path
            
            # Execute the battle
            battle_results = await human_vs_ai_service.execute_human_vs_ai_battle(
                env_id=env_id,
                model_path=model_path,
                opponent_model_id=opponent_model_id,
                opponent_name=opponent_model.model_name
            )
            
            # Update battle results in database
            await human_vs_ai_service.update_battle_results(battle_id, battle_results)
            
            # Update battle status
            battle.status = "completed"
            db.commit()
            
            return JSONResponse(content={
                "battle_id": battle_id,
                "battle_type": "human_vs_ai",
                "environment": env_id,
                "opponent_model": {
                    "id": opponent_model.id,
                    "name": opponent_model.model_name,
                    "algorithm": opponent_model.algorithm,
                    "user_id": opponent_model.user_id,
                    "description": opponent_model.description
                },
                "keyboard_controls": get_keyboard_controls(env_id),
                "battle_results": battle_results,
                "status": "completed"
            })
            
        except Exception as e:
            logger.error(f"Error executing human vs AI battle: {str(e)}")
            battle.status = "failed"
            db.commit()
            raise HTTPException(status_code=500, detail=f"Failed to execute battle: {str(e)}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error creating human vs AI battle: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create battle: {str(e)}")

def get_keyboard_controls(env_id: str) -> Dict:
    """Get keyboard controls for different ALE environments"""
    controls = {
        "boxing_v3": {
            "description": "Boxing - Punch and move to defeat your opponent",
            "controls": {
                "W": "Move Up",
                "S": "Move Down", 
                "A": "Move Left",
                "D": "Move Right",
                "F": "Punch",
                "G": "Block",
                "Q": "Quit Game"
            }
        },
        "pong_v3": {
            "description": "Pong - Move your paddle to hit the ball",
            "controls": {
                "W": "Move Up",
                "S": "Move Down",
                "F": "Fire",
                "D": "Fire Right", 
                "A": "Fire Left",
                "Q": "Quit Game"
            }
        },
        "tennis": {
            "description": "Tennis - Hit the ball over the net",
            "controls": {
                "W": "Move Up",
                "S": "Move Down",
                "A": "Move Left", 
                "D": "Move Right",
                "F": "Hit Ball",
                "Q": "Quit Game"
            }
        },
        "volleyball": {
            "description": "Volleyball - Spike and block the ball",
            "controls": {
                "W": "Move Up",
                "S": "Move Down",
                "A": "Move Left",
                "D": "Move Right", 
                "F": "Jump/Spike",
                "G": "Block",
                "Q": "Quit Game"
            }
        },
        "warlords": {
            "description": "Warlords - Defend your castle and attack enemies",
            "controls": {
                "W": "Move Up",
                "S": "Move Down",
                "A": "Move Left",
                "D": "Move Right",
                "F": "Fire",
                "Q": "Quit Game"
            }
        },
        "joust": {
            "description": "Joust - Fly and fight on your ostrich",
            "controls": {
                "W": "Fly Up",
                "S": "Fly Down",
                "A": "Move Left",
                "D": "Move Right",
                "F": "Attack",
                "Q": "Quit Game"
            }
        },
        "combat": {
            "description": "Combat - Tank warfare with multiple weapons",
            "controls": {
                "W": "Move Forward",
                "S": "Move Backward",
                "A": "Turn Left",
                "D": "Turn Right",
                "F": "Fire",
                "G": "Change Weapon",
                "Q": "Quit Game"
            }
        }
    }
    
    return controls.get(env_id, {
        "description": "ALE Environment",
        "controls": {
            "WASD": "Movement",
            "F": "Action",
            "Q": "Quit Game"
        }
    })

@router.get("/battles")
async def list_battles(
    user_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List battles"""
    try:
        query = db.query(Battle)
        
        if user_id:
            # Get battles where user's models participated
            user_models = db.query(UserModel.id).filter(UserModel.user_id == user_id).all()
            user_model_ids = [m.id for m in user_models]
            query = query.filter(
                (Battle.player_model_id.in_(user_model_ids)) |
                (Battle.opponent_model_id.in_(user_model_ids))
            )
        
        if status:
            query = query.filter(Battle.status == status)
            
        battles = query.order_by(Battle.created_at.desc()).limit(limit).all()
        
        return JSONResponse(content={
            "battles": [
                {
                    "id": battle.id,
                    "player_model_id": battle.player_model_id,
                    "opponent_model_id": battle.opponent_model_id,
                    "env_id": battle.env_id,
                    "status": battle.status,
                    "created_at": battle.created_at.isoformat(),
                    "completed_at": battle.completed_at.isoformat() if battle.completed_at else None
                }
                for battle in battles
            ]
        })
        
    except Exception as e:
        logger.error(f"Error listing battles: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list battles: {str(e)}")

@router.get("/battles/{battle_id}")
async def get_battle(battle_id: str, db: Session = Depends(get_db)):
    """Get battle details and results"""
    try:
        battle = db.query(Battle).filter(Battle.id == battle_id).first()
        if not battle:
            raise HTTPException(status_code=404, detail="Battle not found")
        
        # Get battle result if completed
        result = None
        if battle.status == "completed":
            battle_result = db.query(BattleResult).filter(BattleResult.battle_id == battle_id).first()
            if battle_result:
                result = {
                    "player_score": battle_result.player_score,
                    "opponent_score": battle_result.opponent_score,
                    "winner": battle_result.winner,
                    "duration_seconds": battle_result.duration_seconds,
                    "battle_log": json.loads(battle_result.battle_log) if battle_result.battle_log else None
                }
        
        return JSONResponse(content={
            "id": battle.id,
            "player_model_id": battle.player_model_id,
            "opponent_model_id": battle.opponent_model_id,
            "env_id": battle.env_id,
            "status": battle.status,
            "created_at": battle.created_at.isoformat(),
            "completed_at": battle.completed_at.isoformat() if battle.completed_at else None,
            "result": result
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting battle {battle_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get battle: {str(e)}")

@router.get("/rankings")
async def get_rankings(
    env_id: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """Get model rankings based on battle performance"""
    try:
        query = db.query(UserModel).filter(UserModel.is_public == True)
        
        if env_id:
            query = query.filter(UserModel.env_id == env_id)
        
        # Order by win rate, then by total battles
        models = query.order_by(
            UserModel.win_rate.desc(),
            UserModel.total_battles.desc()
        ).limit(limit).all()
        
        return JSONResponse(content={
            "rankings": [
                {
                    "rank": i + 1,
                    "id": model.id,
                    "model_name": model.model_name,
                    "algorithm": model.algorithm,
                    "env_id": model.env_id,
                    "user_id": model.user_id,
                    "total_battles": model.total_battles,
                    "wins": model.wins,
                    "losses": model.losses,
                    "draws": model.draws,
                    "win_rate": model.win_rate
                }
                for i, model in enumerate(models)
            ]
        })
        
    except Exception as e:
        logger.error(f"Error getting rankings: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get rankings: {str(e)}")

@router.websocket("/ws/{battle_id}")
async def websocket_battle_updates(websocket: WebSocket, battle_id: str):
    """WebSocket endpoint for real-time battle updates"""
    await websocket.accept()
    active_connections[battle_id] = websocket
    
    try:
        while True:
            # Keep connection alive and wait for messages
            data = await websocket.receive_text()
            # Echo back for now - in real implementation, this would handle battle commands
            await websocket.send_text(f"Message received: {data}")
    except WebSocketDisconnect:
        if battle_id in active_connections:
            del active_connections[battle_id]
    except Exception as e:
        logger.error(f"WebSocket error for battle {battle_id}: {str(e)}")
        if battle_id in active_connections:
            del active_connections[battle_id]

async def update_battle_status(battle_id: str, status: str, result: Optional[Dict] = None, db: Session = None):
    """Update battle status and notify connected clients"""
    try:
        if db:
            battle = db.query(Battle).filter(Battle.id == battle_id).first()
            if battle:
                battle.status = status
                if status == "completed":
                    battle.completed_at = datetime.utcnow()
                db.commit()
        
        # Notify WebSocket clients
        if battle_id in active_connections:
            websocket = active_connections[battle_id]
            message = {
                "type": "battle_update",
                "battle_id": battle_id,
                "status": status,
                "result": result
            }
            await websocket.send_text(json.dumps(message))
            
    except Exception as e:
        logger.error(f"Error updating battle status: {str(e)}")
