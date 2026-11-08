#!/usr/bin/env bash
# Build librealsense + Python bindings on a Pi 5 with the RSUSB backend
# (user-space USB, no kernel patches). Run after setup_pi.sh.
#   LRS_TAG=v2.55.1 JOBS=3 bash deploy/build_librealsense.sh
# If this tag doesn't build on your OS, try the newest release tag.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO=$(pwd)
TAG=${LRS_TAG:-v2.55.1}
JOBS=${JOBS:-3}     # 4 GB Pi: use 2 to avoid running out of RAM
PY=$REPO/.venv/bin/python

mkdir -p ~/src && cd ~/src
[ -d librealsense ] || git clone --depth 1 --branch "$TAG" https://github.com/IntelRealSense/librealsense.git
cd librealsense
sudo cp config/99-realsense-libusb.rules /etc/udev/rules.d/
sudo udevadm control --reload-rules && sudo udevadm trigger
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DFORCE_RSUSB_BACKEND=ON -DBUILD_PYTHON_BINDINGS=ON \
  -DPYTHON_EXECUTABLE="$PY" -DBUILD_EXAMPLES=OFF -DBUILD_GRAPHICAL_EXAMPLES=OFF -DBUILD_WITH_OPENMP=OFF \
  -DCHECK_FOR_UPDATES=OFF
make -j"$JOBS"
sudo make install
sudo ldconfig

# Put the bindings into the venv.
SITE=$("$PY" -c 'import site; print(site.getsitepackages()[0])')
find . -name 'pyrealsense2*.so*' -exec cp -v {} "$SITE/" \;
find . -name 'pyrsutils*.so*' -exec cp -v {} "$SITE/" \; || true
"$PY" -c 'import pyrealsense2 as rs; print("pyrealsense2 OK, devices:", len(rs.context().devices))'
echo "Now: cd $REPO && .venv/bin/python -m ripesense.hw.camera --test"
