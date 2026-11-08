#!/bin/sh
export DISPLAY=:0
export XAUTHORITY=/home/ieee/.Xauthority
export MOZ_ENABLE_WAYLAND=0
export LIBGL_ALWAYS_SOFTWARE=1
for attempt in $(seq 1 30); do
    curl -fs http://127.0.0.1:8000/ >/dev/null && break
    sleep 1
done
xset s off
xset -dpms
xset s noblank
exec firefox --no-remote --profile /home/ieee/.mozilla/ripesense-demo --kiosk http://127.0.0.1:8000/
