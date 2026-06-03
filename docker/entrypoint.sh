#!/bin/sh
set -eu

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

: "${MEIPI_POOL_ID:?Set MEIPI_POOL_ID to an existing datapool id}"

WATCH_PATH="${MEIPI_WATCH_PATH:-.}"
DEBOUNCE="${MEIPI_DEBOUNCE:-1.0}"

set -- meipi-index watch --pool-id "${MEIPI_POOL_ID}" --debounce "${DEBOUNCE}"

if [ "${MEIPI_NO_THUMBS:-0}" = "1" ]; then
  set -- "$@" --no-thumbs
fi

set -- "$@" "${WATCH_PATH}"

echo "Watching pool ${MEIPI_POOL_ID} at ${WATCH_PATH}..."
exec "$@"
