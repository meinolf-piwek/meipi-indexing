# Filesystem watcher for meipi-indexing (CPU image, no CUDA/DALI).
# Thumbnails are disabled by default; enable only with a GPU-capable custom build.

FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MEIPI_CONFIG_ENV=/etc/meipi/config.env \
    MEIPI_NO_THUMBS=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libexif12 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.MD LICENSE ./
COPY src ./src
COPY docker/requirements-watcher.txt ./docker/requirements-watcher.txt
COPY docker/config.env ./docker/config.env
COPY docker/ensure_pool.py ./docker/ensure_pool.py

RUN pip install --upgrade pip \
    && pip install -r docker/requirements-watcher.txt \
    && pip install --no-deps .

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

RUN mkdir -p /etc/meipi \
    && cp docker/config.env /etc/meipi/config.env

ENTRYPOINT ["/entrypoint.sh"]
