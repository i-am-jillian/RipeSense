#!/usr/bin/env bash
# One-time Pi 5 setup (Raspberry Pi OS Bookworm 64-bit). Run from the repo root:
#   bash deploy/setup_pi.sh
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)

echo "== apt packages"
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev python3-gpiozero python3-lgpio i2c-tools \
  git cmake build-essential pkg-config libusb-1.0-0-dev libudev-dev libssl-dev \
  chromium-browser unclutter || sudo apt-get install -y chromium

echo "== enable I2C + SPI"
sudo raspi-config nonint do_i2c 0
sudo raspi-config nonint do_spi 0

echo "== python venv (system site-packages for gpiozero/lgpio)"
python3 -m venv --system-site-packages .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt -r requirements-pi.txt

echo "== systemd service"
sed "s|__USER__|$USER|g; s|__REPO__|$REPO|g" deploy/ripesense.service | sudo tee /etc/systemd/system/ripesense.service >/dev/null
[ -f .env ] || cp .env.example .env
sudo systemctl daemon-reload
sudo systemctl enable ripesense.service

echo "== kiosk autostart"
mkdir -p ~/.config/autostart
sed "s|__REPO__|$REPO|g" deploy/ripesense-kiosk.desktop > ~/.config/autostart/ripesense-kiosk.desktop
chmod +x deploy/kiosk.sh

cat <<MSG

Done. Next:
  1. bash deploy/build_librealsense.sh      (≈45–90 min on a Pi 5; start it now)
  2. Screen: see deploy/SCREEN.md, then reboot
  3. Check sensors:  i2cdetect -y 1   → 39 (AS7341) and 48 (ADS1115) or 77 (BME680)
  4. Set gas.type in config.yaml (mq | bme680), then: sudo systemctl start ripesense
  5. Browse to http://$(hostname -I | awk '{print $1}'):5000/collect from your laptop
MSG
