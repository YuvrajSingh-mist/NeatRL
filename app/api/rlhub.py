from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from app.db.session import get_db
from app.models.submission import UserModel
from app.core.supabase import supabase_client
from app.core.config import settings
import uuid
import json
import logging
from typing import List, Optional
from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/rlhub", tags=["RL Hub"])

@router.post("/upload")
async def upload_model(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    description: Optional[str] = Form(None),
    algorithm: str = Form(...),
    env_id: str = Form(...),
    user_id: str = Form(...),
    is_public: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Upload a trained RL model to the RL Hub"""
    try:
        # Validate file type
        if not file.filename.endswith('.py'):
            raise HTTPException(status_code=400, detail="Only Python (.py) files are allowed")
        
        # Generate unique ID for the model
        model_id = str(uuid.uuid4())
        
        # Upload file to Supabase Storage
        file_path = f"models/{model_id}/{file.filename}"
        
        # Read file content
        file_content = await file.read()
        
        # Upload to Supabase Storage
        storage_response = supabase_client.storage.from_(settings.supabase_bucket).upload(
            path=file_path,
            file=file_content,
            file_options={"content-type": "text/plain"}
        )
        
        if not storage_response:
            raise HTTPException(status_code=500, detail="Failed to upload file to storage")
        
        # Create model record in database
        db_model = UserModel(
            id=model_id,
            user_id=user_id,
            model_name=model_name,
            description=description,
            algorithm=algorithm,
            env_id=env_id,
            file_path=file_path,
            is_public=is_public
        )
        
        db.add(db_model)
        db.commit()
        db.refresh(db_model)
        
        logger.info(f"Model uploaded successfully: {model_id} by user {user_id}")
        
        return JSONResponse(content={
            "id": model_id,
            "model_name": model_name,
            "algorithm": algorithm,
            "env_id": env_id,
            "status": "uploaded"
        })
        
    except Exception as e:
        logger.error(f"Error uploading model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

@router.get("/models")
async def list_models(
    env_id: Optional[str] = None,
    algorithm: Optional[str] = None,
    user_id: Optional[str] = None,
    limit: int = 50,
    db: Session = Depends(get_db)
):
    """List available models in the RL Hub"""
    try:
        query = db.query(UserModel).filter(UserModel.is_public == True)
        
        if env_id:
            query = query.filter(UserModel.env_id == env_id)
        if algorithm:
            query = query.filter(UserModel.algorithm == algorithm)
        if user_id:
            query = query.filter(UserModel.user_id == user_id)
            
        models = query.order_by(UserModel.created_at.desc()).limit(limit).all()
        
        return JSONResponse(content={
            "models": [
                {
                    "id": model.id,
                    "model_name": model.model_name,
                    "description": model.description,
                    "algorithm": model.algorithm,
                    "env_id": model.env_id,
                    "user_id": model.user_id,
                    "created_at": model.created_at.isoformat(),
                    "total_battles": model.total_battles,
                    "wins": model.wins,
                    "losses": model.losses,
                    "draws": model.draws,
                    "win_rate": model.win_rate
                }
                for model in models
            ]
        })
        
    except Exception as e:
        logger.error(f"Error listing models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")

@router.get("/models/{model_id}")
async def get_model(model_id: str, db: Session = Depends(get_db)):
    """Get details of a specific model"""
    try:
        model = db.query(UserModel).filter(UserModel.id == model_id).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found")
            
        return JSONResponse(content={
            "id": model.id,
            "model_name": model.model_name,
            "description": model.description,
            "algorithm": model.algorithm,
            "env_id": model.env_id,
            "user_id": model.user_id,
            "created_at": model.created_at.isoformat(),
            "updated_at": model.updated_at.isoformat(),
            "total_battles": model.total_battles,
            "wins": model.wins,
            "losses": model.losses,
            "draws": model.draws,
            "win_rate": model.win_rate
        })
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model: {str(e)}")

@router.delete("/models/{model_id}")
async def delete_model(model_id: str, user_id: str, db: Session = Depends(get_db)):
    """Delete a model (only by the owner)"""
    try:
        model = db.query(UserModel).filter(
            UserModel.id == model_id,
            UserModel.user_id == user_id
        ).first()
        
        if not model:
            raise HTTPException(status_code=404, detail="Model not found or not owned by user")
        
        # Delete from storage
        try:
            supabase_client.storage.from_(settings.supabase_bucket).remove([model.file_path])
        except Exception as e:
            logger.warning(f"Failed to delete file from storage: {str(e)}")
        
        # Delete from database
        db.delete(model)
        db.commit()
        
        logger.info(f"Model deleted: {model_id} by user {user_id}")
        
        return JSONResponse(content={"status": "deleted"})
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")
