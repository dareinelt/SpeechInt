# SpeechInt gateway – the HTTP API in front of whisper.cpp and llama.cpp.
#
# The image contains only the FastAPI app; the two model servers run as their
# own containers (see docker-compose.yml), so the gateway stays small and can
# be restarted or updated without touching the models.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# curl is only needed for the container healthcheck.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Unprivileged runtime user: the service only reads its configuration from the
# environment and never writes to disk.
RUN useradd --system --create-home --uid 10001 speechint
USER speechint

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
  CMD curl -fsS http://localhost:8080/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--proxy-headers"]
