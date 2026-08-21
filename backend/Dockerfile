FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# rag/vectorstore is meant to be a mounted volume (see docker-compose.yml)
# so the ingested index survives container rebuilds; make sure the
# directory exists even before ingestion has run.
RUN mkdir -p /app/rag/vectorstore

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
