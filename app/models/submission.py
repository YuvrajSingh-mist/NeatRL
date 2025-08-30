
from sqlalchemy import Column, String, Float, DateTime, Text, Integer, ForeignKey, Boolean
from sqlalchemy.orm import relationship
from datetime import datetime
from app.db.base import Base

class Submission(Base):
    __tablename__ = "submissions"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    env_id = Column(String, index=True)
    algorithm = Column(String, index=True)
    score = Column(Float, nullable=True)
    duration_seconds = Column(Float, nullable=True)  # Actual evaluation duration
    created_at = Column(DateTime, default=datetime.utcnow)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    error = Column(String, nullable=True)

class LeaderboardEntry(Base):
    __tablename__ = "leaderboard_entries"

    id = Column(String, primary_key=True, index=True)  # UUID
    submission_id = Column(String, index=True)
    user_id = Column(String, index=True)
    env_id = Column(String, index=True)
    algorithm = Column(String, index=True)
    score = Column(Float, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

class UserModel(Base):
    """Model for RL Hub - stores user uploaded models"""
    __tablename__ = "user_models"
    
    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, index=True)
    model_name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    algorithm = Column(String, nullable=False)
    env_id = Column(String, nullable=False)
    file_path = Column(String, nullable=False)  # Path to model file in storage
    is_public = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    total_battles = Column(Integer, default=0)
    wins = Column(Integer, default=0)
    losses = Column(Integer, default=0)
    draws = Column(Integer, default=0)
    win_rate = Column(Float, default=0.0)

class Battle(Base):
    """Model for RL Arena - stores battle information"""
    __tablename__ = "battles"
    
    id = Column(String, primary_key=True, index=True)
    player_model_id = Column(String, ForeignKey("user_models.id"), nullable=False)
    opponent_model_id = Column(String, ForeignKey("user_models.id"), nullable=False)
    env_id = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, processing, completed, failed
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)
    
    # Relationships
    player_model = relationship("UserModel", foreign_keys=[player_model_id])
    opponent_model = relationship("UserModel", foreign_keys=[opponent_model_id])

class BattleResult(Base):
    """Model for RL Arena - stores battle results"""
    __tablename__ = "battle_results"
    
    id = Column(String, primary_key=True, index=True)
    battle_id = Column(String, ForeignKey("battles.id"), nullable=False)
    player_score = Column(Float, nullable=False)
    opponent_score = Column(Float, nullable=False)
    winner = Column(String, nullable=True)  # "player", "opponent", "draw"
    duration_seconds = Column(Float, nullable=True)
    battle_log = Column(Text, nullable=True)  # JSON string of battle details
    created_at = Column(DateTime, default=datetime.utcnow)
    
    # Relationship
    battle = relationship("Battle")
