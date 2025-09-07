from fastapi import APIRouter, UploadFile, File, Depends, HTTPException, Form
from sqlalchemy.orm import Session
from typing import Optional
from app.db.session import get_db
from app.models import Submission
from app.core.celery import evaluate_submission_task
from app.core.client import supabase_client
from app.core.config import settings
from app.core.metrics import (
    SUBMISSIONS_RECEIVED_TOTAL,
    SUBMISSIONS_UPLOAD_BYTES_TOTAL,
    SUBMISSIONS_VALIDATION_FAILURES_TOTAL,
)
from app.core.real_metrics import real_metrics
from typing import Set
import uuid
import logging
import io
import tarfile
import os

logger = logging.getLogger(__name__)
router = APIRouter()


# Heuristic list of heavy environments (MuJoCo and similar)
HEAVY_ENVS: Set[str] = {
    # Gymnasium MuJoCo classics
    "Ant-v4", "HalfCheetah-v4", "Hopper-v4", "Humanoid-v4", "Walker2d-v4",
    # Older gym IDs that might still appear
    "Ant-v3", "HalfCheetah-v3", "Hopper-v3", "Humanoid-v3", "Walker2d-v3",
    # Bullet / other physics that are heavy
    "AntBulletEnv-v0", "HumanoidBulletEnv-v0", "Walker2DBulletEnv-v0",
}


def _select_queue_for_env(env_id: str) -> str:
    """Return the Celery queue for a given environment.

    - Heavy environments go to the 'heavy' queue (low concurrency worker)
    - Others use the default 'celery' queue (higher concurrency worker)
    """
    try:
        _id = (env_id or "").strip()
        return "heavy" if _id in HEAVY_ENVS or "mujoco" in _id.lower() else "celery"
    except Exception:
        return "celery"

@router.post("/submit/")
@router.post("/submit")
async def submit_rl_script(
    # Backward-compat single file
    file: UploadFile | None = File(None),
    # New: multiple files support
    files: list[UploadFile] | None = File(None),
    main_file: str | None = Form(None),
    env_id: str = Form("CartPole-v1"),
    algorithm: str = Form("Custom"),
    name: str = Form("anonymous"),
    client_id: str | None = Form(None),
    db: Session = Depends(get_db)
):
    """
    Submit RL agent code.

    - If `files` provided: bundle as a tar archive and upload as <submission_id>.tar.
      Additional support files of any type are allowed, but at least one `.py` must be included.
      The chosen `main_file` (must be a `.py`) will be included as submission.py inside the archive.
    - Else if `file` provided: legacy single-script upload as <submission_id>.py.
    """
    # Create or accept a client-provided submission ID (UUID)
    try:
        if client_id:
            _ = uuid.UUID(client_id)
            submission_id = str(client_id)
        else:
            submission_id = str(uuid.uuid4())
    except Exception:
        logger.warning(f"Invalid client_id provided: {client_id}")
        raise HTTPException(400, "client_id must be a valid UUID")

    has_many = bool(files and len(files) > 0)
    has_one = bool(file is not None)
    if not has_many and not has_one:
        SUBMISSIONS_VALIDATION_FAILURES_TOTAL.labels(reason="no_file").inc()
        real_metrics.record_validation_failure("no_file")
        raise HTTPException(400, "Please upload at least one Python file")

    # Multi-file path
    if has_many:
        # Accept any file types but require at least one Python file among them
        safe_names = []
        for f in files:
            if not f.filename:
                logger.warning("Empty filename in upload", extra={"client_id": client_id})
                SUBMISSIONS_VALIDATION_FAILURES_TOTAL.labels(reason="empty_filename").inc()
                real_metrics.record_validation_failure("empty_filename")
                raise HTTPException(400, "Invalid file in upload")
            safe_names.append(os.path.basename(f.filename))
        if not any(name.lower().endswith('.py') for name in safe_names):
            SUBMISSIONS_VALIDATION_FAILURES_TOTAL.labels(reason="no_py").inc()
            real_metrics.record_validation_failure("no_py")
            raise HTTPException(400, "At least one Python (.py) file is required when uploading multiple files")

        # Choose main file
        chosen_main = None
        if main_file:
            cand = os.path.basename(main_file)
            if cand in safe_names:
                chosen_main = cand
            else:
                logger.warning(
                    f"main_file not in uploaded files: {main_file}",
                    extra={"client_id": client_id},
                )
                raise HTTPException(400, "main_file must match one of the uploaded file names")
        else:
            # Explicitly require main_file when multiple files are uploaded
            SUBMISSIONS_VALIDATION_FAILURES_TOTAL.labels(reason="missing_main").inc()
            real_metrics.record_validation_failure("missing_main")
            raise HTTPException(400, "main_file is required when uploading multiple files")
        if not chosen_main.lower().endswith('.py'):
            SUBMISSIONS_VALIDATION_FAILURES_TOTAL.labels(reason="main_not_py").inc()
            real_metrics.record_validation_failure("main_not_py")
            raise HTTPException(400, "main_file must be a Python (.py) file")

        # Build tar archive in-memory; include all files at root
        try:
            # Read all files exactly once and keep in-memory bytes
            file_bytes_map: dict[str, bytes] = {}
            total_bytes = 0
            now_ts = int(__import__("time").time())
            for f in files:
                content = await f.read()
                file_bytes_map[os.path.basename(f.filename)] = content
                try:
                    total_bytes += len(content) if content is not None else 0
                except Exception:
                    pass

            tar_stream = io.BytesIO()
            with tarfile.open(fileobj=tar_stream, mode='w') as tar:
                # Add uploaded files
                for name, content in file_bytes_map.items():
                    info = tarfile.TarInfo(name=name)
                    info.size = len(content)
                    info.mtime = now_ts
                    info.mode = 0o444
                    tar.addfile(info, io.BytesIO(content))
                # Add alias for main file as submission.py if needed
                if chosen_main != "submission.py":
                    mf_content = file_bytes_map.get(chosen_main)
                    if mf_content is None:
                        raise Exception(f"main_file {chosen_main} not found in upload set")
                    info = tarfile.TarInfo(name="submission.py")
                    info.size = len(mf_content)
                    info.mtime = now_ts
                    info.mode = 0o444
                    tar.addfile(info, io.BytesIO(mf_content))

            tar_stream.seek(0)
            result = supabase_client.storage.from_(settings.SUPABASE_BUCKET).upload(
                f"{submission_id}.tar",
                tar_stream.getvalue(),
                file_options={"content-type": "application/x-tar", "upsert": "true"}
            )
            if hasattr(result, 'error') and result.error:
                raise Exception(str(result.error))
            logger.info(
                f"Uploaded {submission_id}.tar to Supabase (main={chosen_main})",
                extra={
                    "submission_id": submission_id,
                    "env_id": env_id,
                    "algorithm": algorithm,
                    "user_name": name,
                    "client_id": client_id,
                },
            )
            SUBMISSIONS_RECEIVED_TOTAL.labels(mode="multi").inc()
            SUBMISSIONS_UPLOAD_BYTES_TOTAL.inc(total_bytes)
        except Exception as e:
            logger.error(
                f"Failed to upload tar to Supabase {submission_id}: {str(e)}",
                extra={"submission_id": submission_id, "env_id": env_id, "algorithm": algorithm, "user_name": name},
            )
            raise HTTPException(500, "Failed to save submission bundle")

    # Legacy single-file path
    else:
        if not file.filename.endswith(".py"):
            logger.warning(
                f"Invalid file type submitted: {file.filename}",
                extra={"client_id": client_id},
            )
            SUBMISSIONS_VALIDATION_FAILURES_TOTAL.labels(reason="not_py").inc()
            real_metrics.record_validation_failure("not_py")
            raise HTTPException(400, "Only Python scripts allowed")
        try:
            content = await file.read()
            result = supabase_client.storage.from_(settings.SUPABASE_BUCKET).upload(
                f"{submission_id}.py",
                content,
                file_options={"content-type": "text/plain", "upsert": "true"}
            )
            if hasattr(result, 'error') and result.error:
                raise Exception(str(result.error))
            logger.info(
                f"Uploaded {submission_id}.py to Supabase",
                extra={
                    "submission_id": submission_id,
                    "env_id": env_id,
                    "algorithm": algorithm,
                    "user_name": name,
                    "client_id": client_id,
                },
            )
            SUBMISSIONS_RECEIVED_TOTAL.labels(mode="single").inc()
            try:
                SUBMISSIONS_UPLOAD_BYTES_TOTAL.inc(len(content) if content is not None else 0)
            except Exception:
                pass
        except Exception as e:
            logger.error(
                f"Failed to upload to Supabase {submission_id}: {str(e)}",
                extra={"submission_id": submission_id, "env_id": env_id, "algorithm": algorithm, "user_name": name},
            )
            raise HTTPException(500, "Failed to save submission")

    # Create DB record
    try:
        submission = Submission(
            id=submission_id,
            user_id=name,
            env_id=env_id,
            algorithm=algorithm,
            status="pending"
        )
        db.add(submission)
        db.commit()
        db.refresh(submission)
    except Exception as e:
        logger.error(
            f"Database error creating submission {submission_id}: {str(e)}",
            extra={"submission_id": submission_id, "env_id": env_id, "algorithm": algorithm, "user_name": name},
        )
        raise HTTPException(500, "Failed to create submission record")

    # Queue async evaluation
    logger.info(
        f"Queuing evaluation for submission {submission_id}",
        extra={
            "submission_id": submission_id,
            "env_id": env_id,
            "algorithm": algorithm,
            "user_name": name,
        },
    )
    # Route task based on environment weight
    queue_name = _select_queue_for_env(env_id)
    try:
        evaluate_submission_task.apply_async(
            kwargs={
                "submission_id": submission_id,
                "name": name,
                "env_id": env_id,
                "algorithm": algorithm,
            },
            queue=queue_name,
        )
    except Exception:
        # Fallback to default queue if broker lacks route
        evaluate_submission_task.delay(submission_id)

    return {
        "id": submission_id,
        "status": "queued",
        "env_id": env_id,
        "algorithm": algorithm
    }

@router.get("/results/{submission_id}")
def get_evaluation_results(submission_id: str, db: Session = Depends(get_db)):
    submission = db.query(Submission).get(submission_id)
    if not submission:
        logger.warning(
            f"Submission not found: {submission_id}",
            extra={"submission_id": submission_id},
        )
        raise HTTPException(404, "Submission not found")
    
    return {
        "id": submission.id,
        "status": submission.status,
        "score": submission.score,
        "env_id": submission.env_id,
        "algorithm": submission.algorithm,
        "created_at": submission.created_at.isoformat() if submission.created_at else None,
        "error": submission.error
    }

@router.get("/submissions")
def get_submissions(
    user_id: Optional[str] = None,
    limit: int = 10,
    offset: int = 0,
    db: Session = Depends(get_db)
):
    """Get submissions, optionally filtered by user_id"""
    query = db.query(Submission)
    
    if user_id:
        query = query.filter(Submission.user_id == user_id)
    
    submissions = query.order_by(Submission.created_at.desc()).offset(offset).limit(limit).all()
    
    return {
        "submissions": [
            {
                "id": sub.id,
                "user_id": sub.user_id,
                "status": sub.status,
                "score": sub.score,
                "env_id": sub.env_id,
                "algorithm": sub.algorithm,
                "created_at": sub.created_at.isoformat() if sub.created_at else None,
                "error": sub.error
            }
            for sub in submissions
        ],
        "count": len(submissions)
    }