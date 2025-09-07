from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.db.session import get_db
from app.models import Submission
from datetime import datetime, timedelta

router = APIRouter()

@router.get("/evaluation-metrics")
def get_evaluation_metrics(db: Session = Depends(get_db)):
    """Get evaluation metrics from database"""
    
    # Get counts for different time periods
    now = datetime.utcnow()
    one_hour_ago = now - timedelta(hours=1)
    one_day_ago = now - timedelta(days=1)
    one_week_ago = now - timedelta(weeks=1)
    
    # Total counts
    total_started = db.query(func.count(Submission.id)).scalar()
    total_completed = db.query(func.count(Submission.id)).filter(Submission.status == "completed").scalar()
    total_failed = db.query(func.count(Submission.id)).filter(Submission.status == "failed").scalar()
    
    # Recent counts (last hour)
    recent_started = db.query(func.count(Submission.id)).filter(Submission.created_at >= one_hour_ago).scalar()
    recent_completed = db.query(func.count(Submission.id)).filter(
        Submission.status == "completed", 
        Submission.created_at >= one_hour_ago
    ).scalar()
    recent_failed = db.query(func.count(Submission.id)).filter(
        Submission.status == "failed", 
        Submission.created_at >= one_hour_ago
    ).scalar()
    
    # Daily counts
    daily_started = db.query(func.count(Submission.id)).filter(Submission.created_at >= one_day_ago).scalar()
    daily_completed = db.query(func.count(Submission.id)).filter(
        Submission.status == "completed", 
        Submission.created_at >= one_day_ago
    ).scalar()
    daily_failed = db.query(func.count(Submission.id)).filter(
        Submission.status == "failed", 
        Submission.created_at >= one_day_ago
    ).scalar()
    
    # Weekly counts
    weekly_started = db.query(func.count(Submission.id)).filter(Submission.created_at >= one_week_ago).scalar()
    weekly_completed = db.query(func.count(Submission.id)).filter(
        Submission.status == "completed", 
        Submission.created_at >= one_week_ago
    ).scalar()
    weekly_failed = db.query(func.count(Submission.id)).filter(
        Submission.status == "failed", 
        Submission.created_at >= one_week_ago
    ).scalar()
    
    # By environment
    env_stats = db.query(
        Submission.env_id,
        func.count(Submission.id).label('total'),
        func.count(Submission.id).filter(Submission.status == "completed").label('completed'),
        func.count(Submission.id).filter(Submission.status == "failed").label('failed')
    ).group_by(Submission.env_id).all()
    
    env_breakdown = {}
    for env_id, total, completed, failed in env_stats:
        env_breakdown[env_id] = {
            "total": total,
            "completed": completed,
            "failed": failed
        }
    
    return {
        "total": {
            "started": total_started,
            "completed": total_completed,
            "failed": total_failed
        },
        "recent": {
            "started": recent_started,
            "completed": recent_completed,
            "failed": recent_failed
        },
        "daily": {
            "started": daily_started,
            "completed": daily_completed,
            "failed": daily_failed
        },
        "weekly": {
            "started": weekly_started,
            "completed": weekly_completed,
            "failed": weekly_failed
        },
        "by_environment": env_breakdown,
        "timestamp": now.isoformat()
    }

@router.get("/stats/{stat_type}")
def get_stat(stat_type: str, db: Session = Depends(get_db)):
    """Get specific statistics for frontend display"""
    
    if stat_type == "submissions":
        total_submissions = db.query(func.count(Submission.id)).scalar()
        return {"count": total_submissions or 0}
    
    elif stat_type == "completed":
        completed_submissions = db.query(func.count(Submission.id)).filter(Submission.status == "completed").scalar()
        return {"count": completed_submissions or 0}
    
    elif stat_type == "failed":
        failed_submissions = db.query(func.count(Submission.id)).filter(Submission.status == "failed").scalar()
        return {"count": failed_submissions or 0}
    
    elif stat_type == "success-rate":
        total = db.query(func.count(Submission.id)).scalar() or 0
        completed = db.query(func.count(Submission.id)).filter(Submission.status == "completed").scalar() or 0
        if total > 0:
            rate = round((completed / total) * 100, 1)
        else:
            rate = 0
        return {"count": f"{rate}%"}
    
    else:
        return {"count": 0}
