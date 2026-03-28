FROM python:3.10-slim

WORKDIR /app

# Install git for openenv package installation
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install the OpenEnv package and project dependencies
RUN pip install --no-cache-dir -e ".[baseline]"

EXPOSE 8000

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
