#!/bin/sh
set -eu

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

if [ "${MEIPI_ENSURE_TABLES:-0}" = "1" ]; then
  echo "Ensuring database tables exist..."
  meipi-index create-tables
fi

if [ -z "${MEIPI_POOL_ID:-}" ]; then
  if [ -z "${MEIPI_POOL_NAME:-}" ] || [ -z "${MEIPI_POOL_ROOTPATH:-}" ]; then
    echo "Set MEIPI_POOL_ID or both MEIPI_POOL_NAME and MEIPI_POOL_ROOTPATH." >&2
    exit 1
  fi
  echo "Resolving datapool ${MEIPI_POOL_NAME}..."
  MEIPI_POOL_ID="$(python /app/docker/ensure_pool.py)"
fi

WATCH_PATH="${MEIPI_WATCH_PATH:-.}"
DEBOUNCE="${MEIPI_DEBOUNCE:-1.0}"

set -- meipi-index watch --pool-id "${MEIPI_POOL_ID}" --debounce "${DEBOUNCE}"

if [ "${MEIPI_INITIAL_SCAN:-0}" = "1" ]; then
  set -- "$@" --initial-scan
fi

if [ "${MEIPI_NO_THUMBS:-1}" = "1" ]; then
  set -- "$@" --no-thumbs
fi

set -- "$@" "${WATCH_PATH}"

echo "Starting watcher for pool ${MEIPI_POOL_ID} at ${WATCH_PATH}..."
exec "$@"
