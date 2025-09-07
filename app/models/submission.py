
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


