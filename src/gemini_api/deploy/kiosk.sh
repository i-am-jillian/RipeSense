#!/usr/bin/env bash
# Full-screen browser on the 3.5" display, once the server answers.
URL="${RIPESENSE_URL:-http://localhost:8000/}"
for _ in $(seq 60); do curl -fs "$URL" >/dev/null && break; sleep 1; done
unclutter -idle 0.5 -root >/dev/null 2>&1 &
BROWSER=$(command -v chromium-browser || command -v chromium)
exec "$BROWSER" --kiosk --app="$URL" --window-size=480,320 --window-position=0,0 \
  --noerrdialogs --disable-infobars --disable-session-crashed-bubble --check-for-update-interval=31536000 \
  --overscroll-history-navigation=0 --disable-pinch
