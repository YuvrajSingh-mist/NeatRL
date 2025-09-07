
# Worker service Dockerfile
FROM python:3.10-slim

# Install Docker CLI for Docker-in-Docker functionality
RUN apt-get update && apt-get install -y \
    docker.io \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user and add to staff group (GID 20 to match Docker socket)
RUN useradd -m appuser && \
    groupadd -g 20 staff 2>/dev/null || true && \
    usermod -aG staff appuser && \
    usermod -aG docker appuser
WORKDIR /home/appuser

# Copy minimal worker requirements (no heavy ML)
COPY requirements.worker.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /home/appuser/app

# Create prometheus metrics directory with proper permissions
RUN mkdir -p /home/appuser/prom_metrics && chown -R appuser:appuser /home/appuser/prom_metrics
ENV PROMETHEUS_MULTIPROC_DIR=/home/appuser/prom_metrics

# Switch to non-root user
USER appuser

# Command will be set in docker-compose
