# Multi-stage: build tools and the fastembed model download stay in the
# builder stage; only the venv, the pre-baked model cache, and app code
# make it into the final image. requirements/locked-base.txt only — NOT
# eval.txt/dev.txt/locked.txt (that one's a freeze of the full dev
# environment: sklearn, pandas, matplotlib, pytest, all of which have no
# business in a serving container — see TOLLGATE.md §1's own "deliberately
# absent" table).

FROM python:3.12-slim AS builder

WORKDIR /build

# build-essential covers any package here that doesn't ship a prebuilt
# manylinux wheel for this Python/platform combo. Discarded after this
# stage — never copied into the runtime image.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements/locked-base.txt .
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r locked-base.txt

# Bake the embedding model into the image at BUILD time, not on the first
# real request — TOLLGATE.md Appendix A: "First request takes 30s / Model
# downloading at runtime / Bake it into the Docker image." Model name
# matches EMBEDDING_MODEL in .env.example; if that changes, this line
# needs to change with it — nothing enforces the two stay in sync.
ENV FASTEMBED_CACHE_PATH=/opt/fastembed-cache
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2', cache_dir='/opt/fastembed-cache')"

FROM python:3.12-slim AS runtime

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY --from=builder /opt/fastembed-cache /opt/fastembed-cache
ENV PATH="/opt/venv/bin:$PATH"
ENV FASTEMBED_CACHE_PATH=/opt/fastembed-cache
ENV PYTHONUNBUFFERED=1

# App code only — no data/, no eval/, no tests/, no .env (also blocked by
# .dockerignore even if this COPY pattern ever changes).
COPY app/ ./app/
COPY migrations/ ./migrations/

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/healthz')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
