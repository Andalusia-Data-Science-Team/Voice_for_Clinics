# Base Image
FROM python:3.11-slim

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    build-essential \
    ca-certificates \
    curl \
    gnupg \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# Install the Microsoft ODBC Driver used by patient-history database access.
RUN curl -fsSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -fsSL https://packages.microsoft.com/config/debian/12/prod.list \
        -o /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# Install UV
RUN pip install uv

# Working directory
WORKDIR /app

# Install Python dependencies with UV (much faster than pip)
COPY src/requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy source code
COPY src/ .

# Create persistent directories before switching to non-root user
RUN mkdir -p /app/uploads /app/static /app/logs

# Non-root user for security
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 9999

CMD ["python", "app.py"]