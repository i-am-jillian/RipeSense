#!/usr/bin/env bash
# Full-screen Chromium on the 480x320 screen, pointing at the local app.
URL=${1:-http://localhost:5000/}
until curl -fs "$URL" >/dev/null; do sleep 1; done
BROWSER=$(command -v chromium-browser || command -v chromium)
command -v unclutter >/dev/null && unclutter -idle 0.5 -root &
exec "$BROWSER" --kiosk --app="$URL" --window-size=480,320 --window-position=0,0 \
  --noerrdialogs --disable-infobars --disable-session-crashed-bubble --overscroll-history-navigation=0 \
  --check-for-update-interval=31536000 --password-store=basic
