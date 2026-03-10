# Base Image:
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install UV from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# Working directory
WORKDIR /app

# Install Python dependencies with UV (much faster than pip)
COPY src/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source code
COPY src/ .

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 9999

CMD ["python", "app.py"]