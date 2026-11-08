#!/usr/bin/env bash
# One-time setup on the Raspberry Pi 5 (Raspberry Pi OS Bookworm, 64-bit).
# Run from the project folder:   bash deploy/install.sh
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"

echo "== system packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-opencv python3-numpy python3-lgpio i2c-tools \
  git cmake build-essential pkg-config libssl-dev libusb-1.0-0-dev libudev-dev unclutter
sudo apt-get install -y chromium-browser || sudo apt-get install -y chromium

echo "== enable I2C and SPI"
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

echo "== Python environment (uses the system OpenCV and lgpio)"
python3 -m venv --system-site-packages .venv
. .venv/bin/activate
pip install --upgrade pip
pip install -e ".[pi,extras]"
[ -f config.toml ] || cp config.example.toml config.toml

echo "== service (starts the server at boot)"
sed -e "s|__USER__|$USER|g" -e "s|__ROOT__|$ROOT|g" deploy/ripesense.service | sudo tee /etc/systemd/system/ripesense.service >/dev/null
sudo touch /etc/ripesense.env    # put GEMINI_API_KEY=... here (optional)
sudo systemctl daemon-reload
sudo systemctl enable ripesense.service

echo "== kiosk browser at login"
mkdir -p ~/.config/autostart
sed "s|__ROOT__|$ROOT|g" deploy/ripesense-kiosk.desktop > ~/.config/autostart/ripesense-kiosk.desktop
chmod +x deploy/kiosk.sh

cat <<EOF

Done. Next:
  1. Screen driver: follow the Hosyond tutorial, reboot, check the 3.5" screen shows the desktop.
  2. RealSense:     bash deploy/build_librealsense.sh      (about an hour on a Pi 5)
  3. Check parts:   .venv/bin/python -m ripesense doctor
  4. Calibrate:     .venv/bin/python -m ripesense calibrate
  5. Start:         sudo systemctl start ripesense   (then reboot to test the kiosk)
EOF
