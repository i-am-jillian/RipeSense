#!/usr/bin/env bash
# Build Intel librealsense with Python bindings on the Pi 5, using the RSUSB backend
# (no kernel patches needed). Takes about an hour; start it the night before.
# Usage: bash deploy/build_librealsense.sh        (LRS_VERSION=v2.55.1 by default)
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$PWD"
VER="${LRS_VERSION:-v2.55.1}"
JOBS="${JOBS:-3}"            # use 2 on a 4 GB Pi, or add swap
PY="$ROOT/.venv/bin/python"
[ -x "$PY" ] || { echo "run deploy/install.sh first"; exit 1; }

mkdir -p "$HOME/src" && cd "$HOME/src"
[ -d librealsense ] || git clone --depth 1 --branch "$VER" https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger

mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$PY" -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_UNIT_TESTS=OFF
make -j"$JOBS"
sudo make install
sudo ldconfig

# make "import pyrealsense2" work inside the project's venv
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
SO_DIR=$(dirname "$(find "$PWD" -name 'pyrealsense2*.so' | head -n1)")
echo "$SO_DIR" > "$SITE/pyrealsense2-build.pth"
"$PY" - <<'EOF'
import pyrealsense2 as rs
devs = [d.get_info(rs.camera_info.name) for d in rs.context().query_devices()]
print("pyrealsense2 OK; cameras found:", devs or "none (plug into a blue USB 3 port)")
EOF
