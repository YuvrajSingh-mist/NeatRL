
from fastapi import FastAPI
from fastapi import Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, PlainTextResponse
from app.middleware.real_metrics import RealMetricsMiddleware
from app.api import submissions, leaderboard
from app.api import alerts
from app.api import visitor
from app.api import metrics_endpoint, prometheus_metrics
from app.db.session import init_db
from app.core.config import settings
from app.services.leaderboard import redis_leaderboard
from app.core.metrics import init_fastapi_instrumentation, OVERALL_SYSTEM_HEALTH, DATABASE_HEALTH, REDIS_HEALTH, CELERY_WORKER_HEALTH, SUPABASE_STORAGE_HEALTH
from app.core.logging_config import setup_logging
import logging
import os
import asyncio
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

# Configure logging (JSON)
setup_logging()

app = FastAPI(
    title="NeatRL",
    description="A Reinforcement Learning evaluation platform for testing and comparing trained agents",
    version="1.0.0"
)

## Sentry removed; using Loki/Promtail for logs


# Expose Prometheus /metrics immediately (not only on startup)
try:
    init_fastapi_instrumentation(app)
except Exception as _e:
    logging.getLogger(__name__).exception("Prometheus metrics init failed", extra={"error": str(_e)})

# Custom metrics endpoint that includes both FastAPI and custom metrics
@app.get("/metrics")
async def metrics():
    """Custom metrics endpoint that combines FastAPI metrics with our custom health metrics"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        # Generate metrics from prometheus_client (includes our custom metrics)
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to generate metrics: {str(e)}")
        return Response(content="", status_code=500)

# CORS for cross-origin frontend
_cors_origins_env = os.getenv("CORS_ORIGINS", "*")
_cors_origins = [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add REAL metrics middleware
app.add_middleware(RealMetricsMiddleware)


# Initialize database, Redis, and metrics
@app.on_event("startup")
async def startup_event():
    # Initialize database (non-fatal)
    try:
        init_db()
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.exception("DB init skipped due to error", extra={"error": str(e)})
    logger = logging.getLogger(__name__)
    logger.info("Database initialized successfully")
    
    # Logs: Structured JSON to stdout, collected by Promtail -> Loki

    # Prometheus /metrics already exposed at import time

    # Initialize Redis leaderboard
    try:
        redis_leaderboard.connect()
        logger.info("Redis leaderboard connected successfully")
        # Backfill persistent entries and warm Redis
        redis_leaderboard.sync_from_submissions()
        redis_leaderboard.warm_redis_from_db()
    except Exception as e:
        logger.exception("Failed to initialize Redis leaderboard", extra={"error": str(e)})
        logger.info("Will use database fallback for leaderboard")

    # Start background task to refresh unique visitor gauges
    try:
        async def _refresh_loop():
            while True:
                try:
                    visitor.refresh_unique_visitor_metrics()
                except Exception as e:
                    logging.getLogger(__name__).debug("visitor_metrics_refresh_loop_error", extra={"error": str(e)})
                await asyncio.sleep(30)
        asyncio.create_task(_refresh_loop())
        logger.info("Visitor metrics refresh loop started")
    except Exception as e:
        logging.getLogger(__name__).warning("failed_to_start_refresh_loop", extra={"error": str(e)})

    # Start background task to monitor all system components
    try:
        async def _system_health_loop():
            import datetime
            from app.core.metrics import check_overall_system_health
            while True:
                try:
                    health_status = check_overall_system_health()
                    if not health_status["overall"]:
                        logging.getLogger(__name__).error(
                            "system_health_check_failed",
                            extra={
                                "health_status": health_status,
                                "timestamp": datetime.datetime.utcnow().isoformat(),
                            }
                        )
                    else:
                        logging.getLogger(__name__).info(
                            "system_health_check_passed",
                            extra={
                                "health_status": health_status,
                                "timestamp": datetime.datetime.utcnow().isoformat(),
                            }
                        )
                except Exception as e:
                    logging.getLogger(__name__).error("system_health_check_failed", extra={"error": str(e)})
                await asyncio.sleep(15)  # Check every 15 seconds
        asyncio.create_task(_system_health_loop())
        logger.info("System health monitoring loop started")
    except Exception as e:
        logging.getLogger(__name__).warning("failed_to_start_system_health_loop", extra={"error": str(e)})

    # Seed health gauges immediately so dashboards don't show "No data"
    try:
        from app.core.metrics import check_overall_system_health
        _ = check_overall_system_health()
    except Exception as e:
        logging.getLogger(__name__).debug("initial_health_seed_failed", extra={"error": str(e)})


@app.middleware("http")
async def request_logging_middleware(request: Request, call_next):
    import time
    from uuid import uuid4
    logger = logging.getLogger("request")
    start = time.perf_counter()
    request_id = str(uuid4())
    client = request.client.host if request.client else "-"

    # Extract visitor token (JWT) if present
    visitor_id = None
    try:
        import jwt
        token = request.cookies.get("visitor_token") or request.headers.get("X-Visitor-Token")
        if token:
            try:
                payload = jwt.decode(
                    token,
                    settings.VISITOR_JWT_SECRET,
                    algorithms=["HS256"],
                    audience=settings.VISITOR_JWT_AUDIENCE,
                    issuer=settings.VISITOR_JWT_ISSUER,
                    options={"verify_exp": True},
                )
                visitor_id = payload.get("sub")
            except Exception as e:
                logging.getLogger(__name__).debug("request_middleware_jwt_decode_failed", extra={"error": str(e)})
                visitor_id = None
    except Exception as e:
        logging.getLogger(__name__).debug("request_middleware_token_extraction_failed", extra={"error": str(e)})
        visitor_id = None

    try:
        response = await call_next(request)
        duration_ms = int((time.perf_counter() - start) * 1000)
        extra = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": getattr(response, "status_code", 0),
            "duration_ms": duration_ms,
            "client": client,
            "visitor_id": visitor_id,
        }
        logger.info("request_completed", extra=extra)
        return response
    except Exception:
        duration_ms = int((time.perf_counter() - start) * 1000)
        extra = {
            "request_id": request_id,
            "method": request.method,
            "path": request.url.path,
            "status_code": 500,
            "duration_ms": duration_ms,
            "client": client,
            "visitor_id": visitor_id,
        }
        logger.exception("request_failed", extra=extra)
        raise

# Include API routes
app.include_router(submissions.router, prefix="/api", tags=["submissions"])
app.include_router(leaderboard.router, prefix="/api/leaderboard", tags=["leaderboard"])
app.include_router(alerts.router, prefix="/api", tags=["alerts"])
app.include_router(visitor.router, prefix="/api", tags=["visitor"])
app.include_router(metrics_endpoint.router, prefix="/api", tags=["metrics"])
app.include_router(prometheus_metrics.router, prefix="/api", tags=["prometheus"])

# --- Basic SEO endpoints ---
@app.get("/robots.txt", response_class=PlainTextResponse)
def robots_txt():
    lines = [
        "User-agent: *",
        "Allow: /",
        "Sitemap: {}/sitemap.xml".format(settings.PUBLIC_BASE_URL.rstrip("/")) if settings.PUBLIC_BASE_URL else "",
    ]
    text = "\n".join([l for l in lines if l]) + "\n"
    return PlainTextResponse(content=text, media_type="text/plain; charset=utf-8")


@app.get("/sitemap.xml")
def sitemap_xml():
    import datetime
    base = settings.PUBLIC_BASE_URL.rstrip("/") if settings.PUBLIC_BASE_URL else ""
    now = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    # Known public routes for sitemap
    urls = []
    if base:
        urls.append({"loc": base + "/", "changefreq": "hourly", "priority": "0.8"})
    # API documentation could be indexed if enabled (FastAPI docs routes). Add conservative entries.

    xml_urls = []
    for u in urls:
        xml_urls.append(
            f"<url><loc>{u['loc']}</loc><lastmod>{now}</lastmod><changefreq>{u['changefreq']}</changefreq><priority>{u['priority']}</priority></url>"
        )
    body = (
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"\
        "<urlset xmlns=\"http://www.sitemaps.org/schemas/sitemap/0.9\">"\
        + "".join(xml_urls) + "</urlset>"
    )
    return Response(content=body, media_type="application/xml")

@app.get("/health")
def health_check():
    """Liveness + Readiness: verify core dependencies (DB, Redis).

    Returns JSON with overall status and component statuses. If any component
    check fails, status is "unhealthy".
    """
    import datetime
    from sqlalchemy import text
    import redis as _redis
    logger = logging.getLogger(__name__)

    statuses: dict[str, str] = {}
    # DB check
    try:
        from app.db.session import engine
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        statuses["database"] = "ok"
    except Exception as e:
        statuses["database"] = f"error: {e}"
    # Redis (leaderboard) check
    try:
        r = _redis.from_url(settings.REDIS_URL, socket_timeout=2)
        r.ping()
        statuses["redis"] = "ok"
    except Exception as e:
        statuses["redis"] = f"error: {e}"

    # Celery broker check (may use a different Redis DB)
    try:
        broker = _redis.from_url(getattr(settings, "CELERY_BROKER_URL", settings.REDIS_URL), socket_timeout=2)
        broker.ping()
        statuses["broker"] = "ok"
    except Exception as e:
        statuses["broker"] = f"error: {e}"

    # Celery worker(s) check via control ping
    try:
        from app.core.celery import celery_app
        
        # Check 1: Basic ping with shorter timeout for blackbox
        pongs = celery_app.control.ping(timeout=1.0)
        if not isinstance(pongs, list) or len(pongs) == 0:
            statuses["celery_workers"] = "error: no workers responding"
        else:
            # For blackbox health checks, just use ping to be fast
            statuses["celery_workers"] = f"ok: {len(pongs)} workers responding"
    except Exception as e:
        statuses["celery_workers"] = f"error: {e}"

    # Supabase Storage check (bucket access)
    try:
        from app.core.client import supabase_client
        bucket = settings.SUPABASE_BUCKET
        # list may vary by SDK version; try a minimal call
        try:
            supabase_client.storage.from_(bucket).list(path="", limit=1)
        except TypeError:
            # older SDK signature
            supabase_client.storage.from_(bucket).list()
        statuses["storage"] = "ok"
    except Exception as e:
        statuses["storage"] = f"error: {e}"

    healthy = all(v.startswith("ok") for v in statuses.values())
    
    # Log health status for monitoring
    if not healthy:
        logger.error(
            "health_check_failed",
            extra={
                "status": "unhealthy",
                "components": statuses,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
        )
    else:
        logger.info(
            "health_check_passed",
            extra={
                "status": "healthy",
                "components": statuses,
                "timestamp": datetime.datetime.utcnow().isoformat(),
            }
        )
    
    return {
        "status": "healthy" if healthy else "unhealthy",
        "components": statuses,
        "timestamp": datetime.datetime.utcnow().isoformat(),
    }





