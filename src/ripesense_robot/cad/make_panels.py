#!/usr/bin/env python3
"""RipeSense laser-cut enclosure generator.

    python cad/make_panels.py            # writes cad/laser/sheet1.svg/.dxf, sheet2.*, parts/*, preview.png
    python cad/make_panels.py --t 2.8    # measured acrylic thickness

All geometry is parametric. Measure your acrylic first (sold as 3 mm, often
2.8–3.2 mm) and set --t, then regenerate. Cut a cardboard test of one corner
before cutting acrylic.

Layers / colours:  CUT = red hairline (#FF0000)   ENGRAVE = black fill (#000000)
Text is emitted as <text>/TEXT entities: convert to paths in LightBurn/Inkscape
(Path ▸ Object to Path) before engraving.

Coordinate system: X = width (left→right), Y = depth (back=0 → front), Z = height.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------- parameters
P = dict(
    t=3.0,            # sheet thickness (measure!)
    clear=0.10,       # added to each side of slots/notches for fit (tune to your laser's kerf)
    W=146.0,          # outer width
    D=140.0,          # outer depth (back face → front face of head/base fronts)
    H=310.0,          # outer height
    base_h=50.0,      # base compartment height (base front panel)
    chamber_h=188.0,  # floor top → ceiling underside (camera ~188 mm above floor)
    lid_inset=4.0,    # bottom/top lids sit this far inside so T-slot holes clear the edge
    door_w=145.0,
    cradle_d=64.6,    # floor hole for the printed cradle insert
    cam_win=(84.0, 20.0),
    cam_screw_dx=53.0,  # must match hybrid_assembly.scad
    screen_win=(74.0, 50.0),
    screen_pcb=(86.0, 56.0),  # Hosyond 3.5" PCB outline: clips grip its left/right edges
    screen_x0=20.0,   # window left edge, from the head front's left edge
    bx=0.0, by=0.0,   # nudge the cradle hole (and thus the AS7341) if the chip isn't centred
    sheet=(600.0, 300.0),
    gap=4.0,
)


# ------------------------------------------------------------------ geometry
@dataclass
class Panel:
    name: str
    outline: list                      # closed polygon [(x, y), ...]
    holes: list = field(default_factory=list)    # closed polygons
    circles: list = field(default_factory=list)  # (x, y, r)
    engrave: list = field(default_factory=list)  # ("circle", x, y, r) | ("ring", x, y, r_out, r_in) | ("text", x, y, size, str)
    qty: int = 1

    def bbox(self):
        xs = [p[0] for p in self.outline]
        ys = [p[1] for p in self.outline]
        return min(xs), min(ys), max(xs), max(ys)

    def normalized(self):
        x0, y0, _, _ = self.bbox()
        f = lambda pts: [(x - x0, y - y0) for x, y in pts]  # noqa: E731
        return Panel(self.name, f(self.outline), [f(h) for h in self.holes],
                     [(x - x0, y - y0, r) for x, y, r in self.circles],
                     [(e[0], e[1] - x0, e[2] - y0, *e[3:]) for e in self.engrave], self.qty)


def rect(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def tab(a, b, d):
    """Edge profile: a tab (d>0, outward) or notch (d<0, inward) from a to b along the edge."""
    return [(a, 0.0), (a, d), (b, d), (b, 0.0)]


def tslot(s, screw=3.2, nut_w=5.8, nut_h=2.6, nut_at=5.0, length=12.0):
    """Edge profile for a T-slot (screw + captive M3 nut) entering the panel at s."""
    h = screw / 2
    n = nut_w / 2
    return [(s - h, 0), (s - h, -nut_at), (s - n, -nut_at), (s - n, -nut_at - nut_h), (s - h, -nut_at - nut_h),
            (s - h, -length), (s + h, -length), (s + h, -nut_at - nut_h), (s + n, -nut_at - nut_h),
            (s + n, -nut_at), (s + h, -nut_at), (s + h, 0)]


def outline(w, h, bottom=(), right=(), top=(), left=()):
    """Walk a w×h rectangle counter-clockwise. Each edge gets a list of profiles
    whose points are (distance along edge, outward depth). Positions are given in
    panel coordinates: bottom/top by x, left/right by y."""
    pts = []

    def walk(origin, direction, normal, length, feats, reverse_pos):
        feats = sorted(feats, key=lambda f: f[0][0])
        if reverse_pos:   # convert panel coordinate → distance along the walking direction
            feats = [[(length - a, d) for a, d in reversed(f)] for f in reversed(feats)]
        pts.append(origin)
        for f in feats:
            for a, d in f:
                pts.append((origin[0] + direction[0] * a + normal[0] * d,
                            origin[1] + direction[1] * a + normal[1] * d))

    walk((0, 0), (1, 0), (0, -1), w, list(bottom), False)
    walk((w, 0), (0, 1), (1, 0), h, list(right), False)
    walk((w, h), (-1, 0), (0, 1), w, list(top), True)
    walk((0, h), (0, -1), (-1, 0), h, list(left), True)
    # Drop consecutive duplicates.
    out = [pts[0]]
    for p in pts[1:]:
        if abs(p[0] - out[-1][0]) > 1e-9 or abs(p[1] - out[-1][1]) > 1e-9:
            out.append(p)
    return out


def build(p=P):
    t, c = p["t"], p["clear"]
    W, D, H = p["W"], p["D"], p["H"]
    IW, ID = W - 2 * t, D - 2 * t                 # interior width / depth
    floor_z = p["base_h"] - t                     # floor occupies [floor_z, floor_z + t]
    ceil_z = p["base_h"] + p["chamber_h"]         # ceiling occupies [ceil_z, ceil_z + t]
    head_z = ceil_z + t                           # head front bottom
    head_h = H - head_z
    li = p["lid_inset"]
    bot_z, top_z = li, H - li - t                 # lid z ranges [bot_z, bot_z+t], [top_z, top_z+t]
    lid_hole_z = (bot_z + t / 2, top_z + t / 2)
    door_h = ceil_z + t - p["base_h"] - 1.0      # 1 mm clearance under the head front

    # Shared joint positions -------------------------------------------------
    rear_tabs = [((i + 0.5) * H / 5 - 12, (i + 0.5) * H / 5 + 12) for i in range(5)]
    front_tabs = [(p["base_h"] / 2 - 10, p["base_h"] / 2 + 10), (head_z + head_h / 2 - 12, head_z + head_h / 2 + 12)]
    hz_floor_side = [(ID * f - 10, ID * f + 10) for f in (0.25, 0.75)]   # along depth (from back inner face)
    hz_floor_back = [(IW * f - 10, IW * f + 10) for f in (0.3, 0.7)]     # along width (from left inner face)

    panels = []

    # Side walls (body u ∈ [t, D-t], tabs into back & fronts) -----------------
    wall = Panel("wall", outline(
        ID, H,
        left=[tab(a, b, t) for a, b in rear_tabs],
        right=[tab(a, b, t) for a, b in front_tabs]), qty=2)
    for z0 in (floor_z, ceil_z):
        for a, b in hz_floor_side:
            wall.holes.append(rect(a - c, z0 - c, b + c, z0 + t + c))
    for z in lid_hole_z:
        wall.circles.append((ID / 2, z, 1.7))
    for i in range(5):                                 # head vents
        z = head_z + 14 + i * 9
        if z + 4 < top_z - 3:
            wall.holes.append(rect(ID / 2 - 25, z, ID / 2 + 25, z + 4))
    wall.outline = [(x + t, y) for x, y in wall.outline]  # shift so tabs start at 0
    wall.holes = [[(x + t, y) for x, y in h] for h in wall.holes]
    wall.circles = [(x + t, y, r) for x, y, r in wall.circles]
    panels.append(wall)

    # Back ---------------------------------------------------------------------
    back = Panel("back", outline(
        W, H,
        left=[tab(a - c, b + c, -t) for a, b in rear_tabs],
        right=[tab(a - c, b + c, -t) for a, b in rear_tabs]))
    for z0 in (floor_z, ceil_z):
        for a, b in hz_floor_back:
            back.holes.append(rect(t + a - c, z0 - c, t + b + c, z0 + t + c))
    for z in lid_hole_z:
        back.circles.append((W / 2, z, 1.7))
    # Raspberry Pi 5: 58 × 49 mm hole pattern, board 85 × 56, centred in the head.
    pi_cz = head_z + (top_z - head_z) / 2
    bx0, bz0 = W / 2 - 42.5, pi_cz - 28
    for dx in (3.5, 61.5):
        for dz in (3.5, 52.5):
            back.circles.append((bx0 + dx, bz0 + dz, 1.35))
    # Wall keyholes 80 mm apart (screw head enters the Ø8, slides down the 4 mm slot).
    for kx in (W / 2 - 40, W / 2 + 40):
        kz = pi_cz - 12
        back.holes.append(keyhole(kx, kz))
    back.circles.append((W - 16, pi_cz, 6.0))         # USB-C power cable Ø12
    panels.append(back)

    # Head front -----------------------------------------------------------------
    hf = Panel("head_front", outline(
        W, head_h,
        left=[tab(a - head_z - c, b - head_z + c, -t) for a, b in front_tabs[1:]],
        right=[tab(a - head_z - c, b - head_z + c, -t) for a, b in front_tabs[1:]]))
    sw, sh = p["screen_win"]
    sx0 = p["screen_x0"]
    sz0 = (head_h - sh) / 2
    hf.holes.append(rect(sx0, sz0, sx0 + sw, sz0 + sh))
    scx = sx0 + sw / 2                                   # clip screws just outside the PCB edges
    clip_dx = p["screen_pcb"][0] / 2 + 4
    for x in (scx - clip_dx, scx + clip_dx):
        for z in (head_h / 2 - 15, head_h / 2 + 15):
            hf.circles.append((x, z, 1.7))
    tap_x = (scx + clip_dx + W - t) / 2 + 1
    hf.engrave.append(("ring", tap_x, head_h / 2 + 3, 11.0, 9.0))
    hf.engrave.append(("text", tap_x, head_h / 2 - 16, 5.0, "TAP"))
    hf.circles.append((W / 2, lid_hole_z[1] - head_z, 1.7))
    panels.append(hf)

    # Door (lift-off, overlays the chamber opening) --------------------------------
    door = Panel("door", rect(0, 0, p["door_w"], door_h))
    door.engrave.append(("text", p["door_w"] / 2, door_h * 0.42, 16.0, "RipeSense"))
    door.engrave.append(("text", p["door_w"] / 2, door_h * 0.42 - 12, 5.5, "fruit ripeness scanner"))
    panels.append(door)

    # Base front -----------------------------------------------------------------
    bf = Panel("base_front", outline(
        W, p["base_h"],
        left=[tab(a - c, b + c, -t) for a, b in front_tabs[:1]],
        right=[tab(a - c, b + c, -t) for a, b in front_tabs[:1]]))
    bf.circles.append((W / 2, lid_hole_z[0], 1.7))
    panels.append(bf)

    # Floor ------------------------------------------------------------------------
    def plate(name):
        return Panel(name, outline(
            IW, ID,
            bottom=[tab(a, b, t) for a, b in hz_floor_back],     # y=0 is the back edge
            left=[tab(a, b, t) for a, b in hz_floor_side],
            right=[tab(a, b, t) for a, b in hz_floor_side]))

    fl = plate("floor")
    cx, cy = IW / 2 + p["bx"], ID / 2 + p["by"]
    fl.circles.append((cx, cy, p["cradle_d"] / 2))
    fan = (25.0, 25.0)                                    # back-left
    fl.circles += grille(fan[0], fan[1], 17.0, 3.0)
    fl.circles += [(fan[0] + sx * 16, fan[1] + sy * 16, 1.7) for sx in (-1, 1) for sy in (-1, 1)]
    gas = (IW - 35.0, 25.0)                               # back-right
    fl.circles += grille(gas[0], gas[1], 11.0, 2.5)
    fl.holes += [rect(gas[0] + sx * 14 - 1.2, gas[1] + sy * 9 - 2.2, gas[0] + sx * 14 + 1.2, gas[1] + sy * 9 + 2.2)
                 for sx in (-1, 1) for sy in (-1, 1)]     # zip-tie slots
    fl.circles.append((IW - 10.0, 12.0, 5.0))             # cable hole → back-right corner
    panels.append(fl)

    # Ceiling ------------------------------------------------------------------------
    ce = plate("ceiling")
    cw, ch = p["cam_win"]
    ce.holes.append(rect(IW / 2 - cw / 2, ID / 2 - ch / 2, IW / 2 + cw / 2, ID / 2 + ch / 2))
    ce.circles += [(IW / 2 + s * p["cam_screw_dx"], ID / 2, 1.7) for s in (-1, 1)]  # on the camera centreline
    ce.circles.append((IW / 2 + 30, ID / 2 + ch / 2 + 10, 2.5))  # LED strip wire Ø5
    ce.circles.append((IW - 10.0, 12.0, 5.0))                     # cable hole, same as floor
    panels.append(ce)

    # Bottom / top lids (inside the walls, held by 4 T-slot M3 screws) -------------
    def lid(name):
        return Panel(name, outline(
            IW, ID,
            bottom=[tslot(IW / 2)], top=[tslot(IW / 2)],
            left=[tslot(ID / 2)], right=[tslot(ID / 2)]))

    panels.append(lid("bottom"))
    top = lid("top")
    for i in range(5):  # vents over the Pi (back of the head)
        y = 8 + i * 8
        top.holes.append(rect(IW / 2 - 35, y, IW / 2 + 35, y + 4))
    panels.append(top)

    # Sensor sled: AS7341 board (Adafruit 25.4×17.8, holes 20.3 apart) → cradle underside
    sled = Panel("sensor_sled", rect(0, 0, 50, 32))
    sled.circles += [(25 + s * 21, 16, 1.7) for s in (-1, 1)]
    sled.circles += [(25 + s * 10.16, 16 + 6.35, 1.35) for s in (-1, 1)]
    sled.holes.append(rect(20, 3, 30, 6))
    panels.append(sled)

    meta = dict(screen_clip_x=(scx - clip_dx, scx + clip_dx), tap_x=tap_x, floor_z=floor_z, ceil_z=ceil_z, head_z=head_z, head_h=head_h, door_h=door_h,
                bot_lid_z=bot_z, top_lid_z=top_z, interior=(IW, ID), cradle=(cx, cy),
                chamber_litres=IW * ID * p["chamber_h"] / 1e6)
    return panels, meta


def keyhole(x, z, big=4.0, slot=2.0, length=9.0, n=24):
    """Keyhole as one closed polygon: Ø8 circle with a 4 mm slot going up."""
    pts = []
    a0 = math.asin(slot / big)
    for i in range(n + 1):  # big circle, from the slot's right edge round the bottom to its left edge
        a = math.pi / 2 + a0 + (2 * math.pi - 2 * a0) * i / n
        pts.append((x + big * math.cos(a), z + big * math.sin(a)))
    top = z + length
    for i in range(n // 2 + 1):  # rounded top of the slot
        a = math.pi + math.pi * i / (n // 2)
        pts.append((x + slot * math.cos(a + math.pi), top - slot * math.sin(a + math.pi)))
    return pts


def grille(cx, cy, r, hole_d):
    """Hex grid of round holes of diameter hole_d inside a circle of radius r."""
    pitch = hole_d + 2.0
    out = []
    n = int(r // pitch) + 1
    for i in range(-n, n + 1):
        for j in range(-n, n + 1):
            x = cx + i * pitch + (pitch / 2 if j % 2 else 0)
            y = cy + j * pitch * 0.866
            if math.hypot(x - cx, y - cy) + hole_d / 2 <= r:
                out.append((x, y, hole_d / 2))
    return out


# ---------------------------------------------------------------------- nesting
@dataclass
class Placed:
    panel: Panel
    x: float
    y: float
    rot: bool


def transform(pt, pl: Placed, h):
    x, y = pt
    if pl.rot:  # rotate 90° CCW then shift into place
        x, y = h - y, x
    return x + pl.x, y + pl.y


def nest(panels, sheet, gap):
    """Shelf packing, largest first, rotation allowed. Returns list of sheets."""
    items = []
    for p in panels:
        for k in range(p.qty):
            items.append(p.normalized())
    items.sort(key=lambda p: -max(p.bbox()[2], p.bbox()[3]) * min(p.bbox()[2], p.bbox()[3]))
    SW, SH = sheet
    sheets = []
    for it in items:
        w, h = it.bbox()[2], it.bbox()[3]
        placed = False
        for sh in sheets:
            for rot in (True, False) if w < h else (False, True):
                pw, ph = (h, w) if rot else (w, h)
                pos = find_spot(sh["placed"], pw, ph, SW, SH, gap)
                if pos:
                    sh["placed"].append((Placed(it, pos[0], pos[1], rot), pw, ph))
                    placed = True
                    break
            if placed:
                break
        if not placed:
            rot = w < h
            pw, ph = (h, w) if rot else (w, h)
            if pw > SW - 2 * gap or ph > SH - 2 * gap:
                rot = not rot
                pw, ph = (h, w) if rot else (w, h)
            sheets.append({"placed": [(Placed(it, gap, gap, rot), pw, ph)]})
    return sheets


def find_spot(placed, pw, ph, SW, SH, gap):
    cands = [(gap, gap)] + [(pl.x + w + gap, pl.y) for pl, w, h in placed] + [(pl.x, pl.y + h + gap) for pl, w, h in placed]
    cands.sort(key=lambda c: (c[0], c[1]))
    for x, y in cands:
        if x + pw > SW - gap or y + ph > SH - gap:
            continue
        if all(x + pw + gap <= pl.x or pl.x + w + gap <= x or y + ph + gap <= pl.y or pl.y + h + gap <= y
               for pl, w, h in placed):
            return x, y
    return None


# ----------------------------------------------------------------------- writers
def svg_doc(items, w, h):
    """items: list of (Placed, ph). Geometry is y-up; SVG is y-down."""
    o = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}mm" height="{h}mm" viewBox="0 0 {w} {h}">',
         f'<!-- RipeSense panels. Red = cut, black = engrave. Sheet {w}x{h} mm -->',
         f'<rect x="0" y="0" width="{w}" height="{h}" fill="none" stroke="#0000FF" stroke-width="0.2" stroke-dasharray="4 4"/>']
    for pl, pw, ph in items:
        p = pl.panel
        hh = p.bbox()[3]
        f = lambda pt: transform(pt, pl, hh)  # noqa: E731
        flip = lambda pt: (round(pt[0], 3), round(h - pt[1], 3))  # noqa: E731
        o.append(f'<g id="{p.name}">')
        for poly in [p.outline] + p.holes:
            d = "M " + " L ".join(f"{x},{y}" for x, y in (flip(f(q)) for q in poly)) + " Z"
            o.append(f'<path d="{d}" fill="none" stroke="#FF0000" stroke-width="0.1"/>')
        for x, y, r in p.circles:
            X, Y = flip(f((x, y)))
            o.append(f'<circle cx="{X}" cy="{Y}" r="{r}" fill="none" stroke="#FF0000" stroke-width="0.1"/>')
        for e in p.engrave:
            X, Y = flip(f((e[1], e[2])))
            if e[0] == "ring":
                o.append(f'<path fill="#000000" fill-rule="evenodd" d="M {X - e[3]},{Y} a {e[3]},{e[3]} 0 1,0 {2 * e[3]},0 '
                         f'a {e[3]},{e[3]} 0 1,0 {-2 * e[3]},0 Z M {X - e[4]},{Y} a {e[4]},{e[4]} 0 1,0 {2 * e[4]},0 '
                         f'a {e[4]},{e[4]} 0 1,0 {-2 * e[4]},0 Z"/>')
            elif e[0] == "text":
                rot = f' transform="rotate(-90 {X} {Y})"' if pl.rot else ""
                o.append(f'<text x="{X}" y="{Y}" font-family="DejaVu Sans, Arial, sans-serif" font-weight="bold" '
                         f'font-size="{e[3]}" text-anchor="middle" dominant-baseline="middle" fill="#000000"{rot}>{e[4]}</text>')
        o.append('</g>')
    o.append('</svg>')
    return "\n".join(o)


def dxf_doc(items):
    """Minimal DXF R12 (LINE-free POLYLINE/CIRCLE/TEXT), mm, layers CUT and ENGRAVE."""
    o = ["0", "SECTION", "2", "HEADER", "9", "$INSUNITS", "70", "4", "0", "ENDSEC",
         "0", "SECTION", "2", "ENTITIES"]

    def poly(pts, layer):
        o.extend(["0", "POLYLINE", "8", layer, "66", "1", "10", "0.0", "20", "0.0", "30", "0.0", "70", "1"])
        for x, y in pts:
            o.extend(["0", "VERTEX", "8", layer, "10", f"{x:.4f}", "20", f"{y:.4f}", "30", "0.0"])
        o.extend(["0", "SEQEND", "8", layer])

    for pl, pw, ph in items:
        p = pl.panel
        hh = p.bbox()[3]
        f = lambda pt: transform(pt, pl, hh)  # noqa: E731
        for pg in [p.outline] + p.holes:
            poly([f(q) for q in pg], "CUT")
        for x, y, r in p.circles:
            X, Y = f((x, y))
            o.extend(["0", "CIRCLE", "8", "CUT", "10", f"{X:.4f}", "20", f"{Y:.4f}", "40", f"{r:.4f}"])
        for e in p.engrave:
            X, Y = f((e[1], e[2]))
            if e[0] == "ring":
                for r in (e[3], e[4]):
                    o.extend(["0", "CIRCLE", "8", "ENGRAVE", "10", f"{X:.4f}", "20", f"{Y:.4f}", "40", f"{r:.4f}"])
            elif e[0] == "text":
                o.extend(["0", "TEXT", "8", "ENGRAVE", "10", f"{X:.4f}", "20", f"{Y:.4f}", "40", f"{e[3]:.3f}",
                          "1", e[4], "50", "90" if pl.rot else "0", "72", "1", "73", "2",
                          "11", f"{X:.4f}", "21", f"{Y:.4f}"])
    o.extend(["0", "ENDSEC", "0", "EOF"])
    return "\n".join(o) + "\n"


def cut_length(items):
    L = 0.0
    for pl, _, _ in items:
        p = pl.panel
        for pg in [p.outline] + p.holes:
            L += sum(math.dist(pg[i], pg[(i + 1) % len(pg)]) for i in range(len(pg)))
        L += sum(2 * math.pi * r for _, _, r in p.circles)
    return L


def preview(sheets, SW, SH, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Polygon
    fig, axes = plt.subplots(len(sheets), 1, figsize=(10, 5.2 * len(sheets)))
    axes = [axes] if len(sheets) == 1 else axes
    for ax, sh in zip(axes, sheets):
        ax.add_patch(Polygon(rect(0, 0, SW, SH), fill=False, ec="#3b82f6", ls="--"))
        for pl, pw, ph in sh["placed"]:
            p = pl.panel
            hh = p.bbox()[3]
            f = lambda pt: transform(pt, pl, hh)  # noqa: E731
            ax.add_patch(Polygon([f(q) for q in p.outline], fc="#eee", ec="r", lw=0.6))
            for hpoly in p.holes:
                ax.add_patch(Polygon([f(q) for q in hpoly], fc="white", ec="r", lw=0.5))
            for x, y, r in p.circles:
                ax.add_patch(Circle(f((x, y)), r, fc="white", ec="r", lw=0.5))
            for e in p.engrave:
                X, Y = f((e[1], e[2]))
                if e[0] == "ring":
                    ax.add_patch(Circle((X, Y), e[3], fc="none", ec="k", lw=1.5))
                else:
                    ax.text(X, Y, e[4], ha="center", va="center", fontsize=e[3] * 0.9, rotation=90 if pl.rot else 0,
                            fontweight="bold")
            cx = pl.x + pw / 2
            cy = pl.y + ph / 2
            ax.text(cx, cy + (ph / 2 - 8 if ph > 60 else 0), p.name, ha="center", va="center", fontsize=7, color="#2563eb")
        ax.set_xlim(-5, SW + 5)
        ax.set_ylim(-5, SH + 5)
        ax.set_aspect("equal")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--t", type=float, help="measured sheet thickness, mm")
    ap.add_argument("--clear", type=float)
    ap.add_argument("--bx", type=float, help="shift cradle hole in X, mm")
    ap.add_argument("--by", type=float, help="shift cradle hole in Y, mm")
    ap.add_argument("--sheet", type=str, help="sheet size WxH, e.g. 600x300")
    ap.add_argument("--out", default=str(HERE / "laser"))
    a = ap.parse_args()
    p = dict(P)
    for k in ("t", "clear", "bx", "by"):
        if getattr(a, k) is not None:
            p[k] = getattr(a, k)
    if a.sheet:
        p["sheet"] = tuple(float(v) for v in a.sheet.lower().split("x"))

    panels, meta = build(p)
    out = Path(a.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    SW, SH = p["sheet"]
    sheets = nest(panels, p["sheet"], p["gap"])
    total = 0.0
    for i, sh in enumerate(sheets, 1):
        (out / f"sheet{i}.svg").write_text(svg_doc(sh["placed"], SW, SH))
        (out / f"sheet{i}.dxf").write_text(dxf_doc(sh["placed"]))
        total += cut_length(sh["placed"])
    for pnl in panels:
        n = pnl.normalized()
        w, h = n.bbox()[2], n.bbox()[3]
        items = [(Placed(n, 2, 2, False), w, h)]
        (out / "parts" / f"{pnl.name}.svg").write_text(svg_doc(items, w + 4, h + 4))
        (out / "parts" / f"{pnl.name}.dxf").write_text(dxf_doc(items))
    preview(sheets, SW, SH, out / "preview.png")
    sizes = {pn.name: [round(pn.normalized().bbox()[2], 1), round(pn.normalized().bbox()[3], 1), pn.qty] for pn in panels}
    (out / "panels.json").write_text(json.dumps({"params": p, "derived": meta, "sizes_mm": sizes}, indent=1))
    print(f"t={p['t']} mm → {len(sheets)} sheet(s) of {SW:.0f}×{SH:.0f}, {sum(pn.qty for pn in panels)} panels, "
          f"cut length ≈ {total / 1000:.1f} m, chamber {meta['chamber_litres']:.2f} L")
    for k, v in sizes.items():
        print(f"  {k:12s} {v[0]:6.1f} × {v[1]:6.1f} mm  ×{v[2]}")
    print(f"→ {out}/sheet*.svg|dxf, parts/, preview.png, panels.json")


if __name__ == "__main__":
    main()
