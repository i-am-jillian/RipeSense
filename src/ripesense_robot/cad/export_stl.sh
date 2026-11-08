#!/usr/bin/env bash
# Export every printed part to cad/stl/ (needs OpenSCAD on PATH).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p stl
PARTS="cradle_insert camera_clamp cal_puck screen_clip magnet_block door_handle plate"
# All-printed fallback (no laser): bash cad/export_stl.sh --printed
[ "${1:-}" = "--printed" ] && PARTS="$PARTS shell_base shell_chamber shell_head shell_door shell_bottom shell_top"
for p in $PARTS; do
  echo "→ $p"
  openscad -D "part=\"$p\"" -o "stl/$p.stl" hybrid_assembly.scad
done
openscad -D 'part="assembly"' --imgsize=1200,1400 --camera=420,-520,420,73,70,150 -o assembly.png hybrid_assembly.scad || true
echo "Slice stl/plate.stl (or the single parts) in PrusaSlicer: PETG, 0.20 mm, supports on build plate only (cradle flange)."
