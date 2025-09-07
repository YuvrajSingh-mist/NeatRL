
# Worker service Dockerfile
FROM python:3.10-slim

# Create non-root user
RUN useradd -m appuser
WORKDIR /home/appuser

# Copy minimal worker requirements (no heavy ML)
COPY requirements.worker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /home/appuser/app

# Switch to non-root user
USER appuser

# Command will be set in docker-compose
RUN mkdir -p /home/appuser/prom_metrics && chown -R appuser:appuser /home/appuser/prom_metrics
ENV PROMETHEUS_MULTIPROC_DIR=/home/appuser/prom_metrics
