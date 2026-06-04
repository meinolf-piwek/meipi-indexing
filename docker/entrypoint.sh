#!/bin/sh
set -eu

if [ "$#" -gt 0 ]; then
  exec "$@"
fi

: "${IND_WATCH_POOL_ID:?Set IND_WATCH_POOL_ID to an existing datapool id}"

WATCH_PATH="${IND_WATCH_PATH:-.}"
DEBOUNCE="${IND_WATCH_DEBOUNCE:-1.0}"

set -- meipi-index watch --pool-id "${IND_WATCH_POOL_ID}" --debounce "${DEBOUNCE}"

if [ "${IND_WATCH_NO_THUMBS:-0}" = "1" ]; then
  set -- "$@" --no-thumbs
fi

set -- "$@" "${WATCH_PATH}"

echo "Watching pool ${IND_WATCH_POOL_ID} at ${WATCH_PATH}..."
exec "$@"
