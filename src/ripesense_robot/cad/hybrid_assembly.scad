// RipeSense — printed parts + assembly preview for the laser-cut cabinet.
//
// Preview everything:     openscad cad/hybrid_assembly.scad
// Export one part:        openscad -D 'part="cradle_insert"' -o cad/stl/cradle_insert.stl cad/hybrid_assembly.scad
// Export all parts:       bash cad/export_stl.sh
//
// Print in PETG, 0.2 mm layers. Print the cradle and camera clamp in BLACK
// (stray light into the spectral sensor), the cal puck in WHITE.
// Dimensions must match cad/make_panels.py (t, W, D, H, base_h, chamber_h …).

part = "assembly";   // assembly | cradle_insert | camera_clamp | cal_puck | screen_clip | magnet_block | door_handle | plate
                     // all-printed fallback (no laser): printed_assembly | shell_base | shell_chamber | shell_head
                     //                                  | shell_door | shell_bottom | shell_top

$fn = 72;

// ---- shared with make_panels.py -------------------------------------------------
t = 3.0;            // acrylic thickness (measure!)
W = 146; D = 140; H = 310;
base_h = 50; chamber_h = 188;
floor_z = base_h - t;              // 47
ceil_z  = base_h + chamber_h;      // 238
head_z  = ceil_z + t;              // 241
IW = W - 2*t; ID = D - 2*t;        // interior 140 × 134
cradle_hole_d = 64.6;
cam_screw_dx = 53;   // clamp screws, ±mm from centre (clears M3 heads past the D435)

// ---- cradle insert --------------------------------------------------------------
cr_body_d   = cradle_hole_d - 0.4;  // slip fit in the floor hole
cr_flange_d = 76;
cr_top      = 8;                    // flange/dish block height above the floor
cr_below    = 15;                   // how far the body hangs below the floor top
dish_r      = 60;                   // spherical seat radius
dish_floor  = 2;                    // dish bottom height at the centre
aperture_d  = 10;                   // spectral window through to the AS7341
as_board    = [26.0, 18.6];         // Adafruit AS7341 board 25.4 × 17.8 + clearance
as_pocket   = 3.0;                  // board pocket depth from below
as_parts    = [20, 14, 2.2];        // relief for components on the board's face
led_d       = 5.2;                  // 5 mm LEDs, push-fit + hot glue
led_exit_r  = 14;                   // where the LED bores come out in the dish
led_tilt    = 20;                   // degrees inward from vertical
sled_dx     = 21;                   // laser-cut sled screw spacing (±21 mm)
insert_d    = 4.0;                  // M3 heat-set insert hole
insert_h    = 5.0;

function dish_z(r) = dish_floor + dish_r - sqrt(dish_r*dish_r - r*r);

module cradle_insert() {
    difference() {
        union() {
            translate([0, 0, -cr_below]) cylinder(d = cr_body_d, h = cr_below + 0.01);
            cylinder(d = cr_flange_d, h = cr_top);
        }
        // Spherical seat.
        translate([0, 0, dish_floor + dish_r]) sphere(r = dish_r, $fn = 160);
        // Aperture down to the sensor.
        translate([0, 0, -cr_below - 1]) cylinder(d = aperture_d, h = cr_below + dish_floor + 2);
        // AS7341 board pocket + component relief (from below).
        translate([-as_board[0]/2, -as_board[1]/2, -cr_below - 0.01]) cube([as_board[0], as_board[1], as_pocket]);
        translate([-as_parts[0]/2, -as_parts[1]/2, -cr_below + as_pocket - 0.01]) cube(as_parts);
        // Cable slot for the STEMMA QT lead.
        translate([as_board[0]/2 - 1, -4, -cr_below - 0.01]) cube([12, 8, as_pocket + 1]);
        // Sled screws: heat-set inserts from below.
        for (s = [-1, 1]) translate([s * sled_dx, 0, -cr_below - 0.01]) cylinder(d = insert_d, h = insert_h);
        // Four LED bores at 45°, 135°, 225°, 315°, tilted toward the aperture.
        for (a = [45, 135, 225, 315]) rotate([0, 0, a]) led_bore();
    }
}

module led_bore() {
    ez = dish_z(led_exit_r);
    len = (ez + cr_below) / cos(led_tilt) + 4;
    // Axis passes through the exit point, leaning inward (toward −x) going up.
    translate([led_exit_r, 0, ez])
        rotate([0, -led_tilt, 0])
            translate([0, 0, -len + 2]) cylinder(d = led_d, h = len);
}

// ---- camera clamp (RealSense lying lens-down on the ceiling) ------------------
// D435: 90 × 25 × 25.  D415: 99 × 20 × 23.  D455: 124 × 26 × 29 (change cam_screw_dx too).
cam = [90, 25, 25];
clamp_w = 16;      // strap width
clamp_t = 3;
foot = [10, clamp_w, 4];   // only foot[2] (thickness) is used

module camera_clamp() {
    span = cam[0] + 1.0;            // inside length
    hgt  = cam[2] + 0.4;            // inside height
    reach = cam_screw_dx + 6;       // feet extend to here
    difference() {
        union() {
            // Top strap + two side walls (an upside-down U over the camera).
            translate([-span/2 - clamp_t, -clamp_w/2, 0]) cube([span + 2*clamp_t, clamp_w, hgt + clamp_t]);
            // Feet on the ceiling, outside the side walls.
            for (s = [-1, 1]) translate([s > 0 ? span/2 : -reach, -clamp_w/2, 0]) cube([reach - span/2, clamp_w, foot[2]]);
        }
        translate([-span/2, -clamp_w/2 - 1, -0.01]) cube([span, clamp_w + 2, hgt]);
        for (s = [-1, 1]) translate([s * cam_screw_dx, 0, -1]) cylinder(d = 3.4, h = foot[2] + 2);
        // USB-C cable exit through one side wall.
        translate([span/2 - 1, -6, 4]) cube([clamp_t + 2, 12, 10]);
    }
}

// ---- white calibration puck (sits in the dish, covers the aperture) ----------
module cal_puck() {
    d = 40;
    intersection() {
        translate([0, 0, dish_floor + dish_r]) sphere(r = dish_r - 0.2, $fn = 160);
        translate([0, 0, dish_floor]) cylinder(d = d, h = 7);
    }
    // Grip knob.
    translate([0, 0, dish_floor + 6]) cylinder(d1 = 14, d2 = 10, h = 8);
}

// ---- screen clip (×2): screws from the front into heat-set inserts ----------
pcb_stack = 4.8;   // PCB + LCD thickness behind the acrylic; measure your screen
module screen_clip() {
    len = 44; w = 8; depth = pcb_stack + 3; lip = 6;
    difference() {
        union() {
            translate([-w/2, -len/2, 0]) cube([w, len, depth]);
            // Lip that presses the PCB edge onto the panel.
            translate([w/2 - 0.01, -len/2 + 6, pcb_stack]) cube([lip, len - 12, 3]);
        }
        for (z = [-15, 15]) translate([0, z, -0.01]) cylinder(d = insert_d, h = insert_h);
    }
}

// ---- magnet block (×4): chamber front corners, magnet flush with the door plane
mag_d = 6.2; mag_h = 3.2; mb = 12;
module magnet_block() {
    difference() {
        cube([mb, mb, mb]);
        // Magnet pocket on the front face (y = mb).
        translate([mb/2, mb + 0.01, mb/2]) rotate([90, 0, 0]) cylinder(d = mag_d, h = mag_h);
    }
}

// ---- door handle ---------------------------------------------------------------
module door_handle() {
    L = 60; stand = 12; th = 6; pad = [12, 14, 2];
    for (s = [-1, 1]) translate([s * (L/2 - pad[0]/2) - pad[0]/2, -pad[1]/2, 0]) cube(pad);
    hull() for (s = [-1, 1]) translate([s * (L/2 - pad[0]/2), 0, 0]) cylinder(d = th, h = stand);
    translate([-L/2 + pad[0]/2, -th/2, stand - th]) cube([L - pad[0], th, th]);
}

// ---- print plate: all parts laid out for one MK4S plate -----------------------
module plate() {
    translate([0, 0, cr_below]) cradle_insert();                     // flange up, prints on its base
    translate([-30, 80, 0]) camera_clamp();
    translate([70, 0, -dish_floor]) cal_puck();
    for (i = [0, 1]) translate([60 + i * 22, 60, 0]) screen_clip();
    for (i = [0:3]) translate([-80 + i * 16, -70, 0]) magnet_block();
    translate([40, -70, 0]) door_handle();
}

// ---- assembly preview ------------------------------------------------------------
module panel(sz, pos, c = [0.08, 0.08, 0.08, 0.95]) color(c) translate(pos) cube(sz);

module assembly() {
    // Laser-cut panels (simplified: no tabs).
    panel([t, D, H], [0, 0, 0]);                         // left wall
    panel([t, D, H], [W - t, 0, 0], [0.08, 0.08, 0.08, 0.25]); // right wall (ghosted)
    panel([W, t, H], [0, 0, 0]);                         // back
    panel([IW, ID, t], [t, t, floor_z]);                 // floor
    panel([IW, ID, t], [t, t, ceil_z]);                  // ceiling
    panel([W, t, H - head_z], [0, D - t, head_z]);       // head front
    panel([W, t, base_h], [0, D - t, 0]);                // base front
    panel([IW, ID, t], [t, t, 4]);                       // bottom lid
    panel([IW, ID, t], [t, t, H - 4 - t]);               // top lid
    // Door (ghosted so you can see inside).
    color([0.1, 0.1, 0.1, 0.25]) translate([0.5, D - t, base_h]) cube([145, t, ceil_z + t - base_h - 1]);

    // Printed parts in place.
    color("orange") translate([W/2, D/2, base_h]) cradle_insert();
    color("orange") translate([W/2, D/2, ceil_z + t]) camera_clamp();
    color("SteelBlue") translate([W/2 - cam[0]/2, D/2 - cam[1]/2, ceil_z + t]) cube(cam);  // RealSense
    for (x = [t, W - t - mb], z = [base_h, ceil_z - mb])
        color("orange") translate([x, D - t - mb, z]) magnet_block();
    color("orange") translate([W/2, D + 0, base_h + 120]) rotate([-90, 0, 0]) door_handle();
    // Fruit.
    color("gold") translate([W/2, D/2, base_h + dish_floor + 30]) scale([1, 0.85, 0.85]) sphere(r = 32);
    // Pi 5 on the back wall, screen behind the head front.
    color("green") translate([W/2 - 42.5, t + 10, head_z + 3]) cube([85, 1.6, 56]);
    color("RoyalBlue") translate([20 + 37 - 43, D - t - 4.8, head_z + (H - head_z)/2 - 28]) cube([86, 4.8, 56]);
}

// =====================================================================================
// ALL-PRINTED FALLBACK (handoff p.16: "no laser cutter free before Saturday").
// Same outer size and internal layout as the acrylic cabinet, split into three
// stacking sections that fit an MK4S bed (250 × 210 × 220). Sections bolt together
// through 4 vertical corner bosses (M3 × 16 + heat-set inserts). Same printed parts
// (cradle, clamp, clips, magnet blocks) fit unchanged. ~25–30 h of printing.
// =====================================================================================
sw = 2.4;                 // shell wall
boss_d = 8;               // corner boss
joint_h = 6;              // alignment lip height
function bosses() = [[sw + boss_d/2, sw + boss_d/2], [W - sw - boss_d/2, sw + boss_d/2],
                     [sw + boss_d/2, D - sw - boss_d/2 - 3], [W - sw - boss_d/2, D - sw - boss_d/2 - 3]];

module shell_box(h, open_front = false, front_from = 0, front_to = 0) {
    difference() {
        cube([W, D, h]);
        translate([sw, sw, -1]) cube([W - 2*sw, D - 2*sw, h + 2]);
        if (open_front) translate([sw, D - sw - 1, front_from]) cube([W - 2*sw, sw + 2, front_to - front_from]);
    }
    for (b = bosses()) translate([b[0], b[1], 0]) difference() {
        cylinder(d = boss_d, h = h);
        translate([0, 0, -1]) cylinder(d = insert_d, h = insert_h + 1);          // insert from below
        translate([0, 0, h - insert_h]) cylinder(d = 3.4, h = insert_h + 1);   // screw from above
    }
}

module plate_with(h = t) {   // integrated floor/ceiling
    translate([sw, sw, 0]) cube([W - 2*sw, D - 2*sw, h]);
}

// Base: fan + gas sensor + spectral under the floor. Bottom lid screws on.
module shell_base() {
    difference() {
        shell_box(base_h - t);
        translate([W/2, -1, 10]) rotate([-90, 0, 0]) cylinder(d = 12, h = sw + 2);   // cable/air hole
    }
}

// Chamber: integrated floor (cradle hole, fan + gas grilles) and ceiling (camera window).
module shell_chamber() {
    ch = chamber_h + 2*t;                    // floor plate + chamber + ceiling plate
    difference() {
        union() {
            shell_box(ch, true, t, ch - t);
            plate_with(t);
            translate([0, 0, ch - t]) plate_with(t);
        }
        // Floor features (same positions as make_panels.py floor, in world coordinates).
        translate([W/2, D/2, -1]) cylinder(d = cradle_hole_d, h = t + 2);
        for (g = [[t + 25, t + 25, 17], [t + IW - 35, t + 25, 11]])
            for (i = [-3:3], j = [-3:3]) {
                x = g[0] + i*5 + (j % 2 == 0 ? 0 : 2.5); y = g[1] + j*4.33;
                if (sqrt(pow(x - g[0], 2) + pow(y - g[1], 2)) + 1.5 <= g[2])
                    translate([x, y, -1]) cylinder(d = 3, h = t + 2);
            }
        translate([t + IW - 10, t + 12, -1]) cylinder(d = 10, h = ch + 2);        // cable path, bottom to top
        // Ceiling features.
        translate([W/2 - cam_win_[0]/2, D/2 - cam_win_[1]/2, ch - t - 1]) cube([cam_win_[0], cam_win_[1], t + 2]);
        for (s = [-1, 1]) translate([W/2 + s*cam_screw_dx, D/2, ch - t - 1]) cylinder(d = 3.4, h = t + 2);
        translate([W/2 + 30, D/2 + cam_win_[1]/2 + 10, ch - t - 1]) cylinder(d = 5, h = t + 2);
    }
    // Door stop ledge + magnet blocks are glued in as in the acrylic version.
}
cam_win_ = [84, 20];

// Head: screen window + TAP pocket in front, Pi on the back wall, vents.
module shell_head() {
    hh = H - head_z;
    difference() {
        shell_box(hh);
        translate([20, D - sw - 1, (hh - 50)/2]) cube([74, sw + 2, 50]);                        // screen window
        translate([124.5, D - 1.5 + 0.01, hh/2 + 3]) rotate([-90, 0, 0]) cylinder(d = 22, h = 2); // TAP pocket (thin wall)
        for (x = [W/2 - 42.5 + 3.5, W/2 - 42.5 + 61.5], z = [5, 54])                           // Pi 5 holes
            translate([x, -1, z]) rotate([-90, 0, 0]) cylinder(d = 2.7, h = sw + 2);
        translate([W - 16, -1, hh/2]) rotate([-90, 0, 0]) cylinder(d = 12, h = sw + 2);         // power
        for (kx = [W/2 - 40, W/2 + 40]) translate([kx, -1, hh/2 - 12]) rotate([-90, 0, 0]) {    // keyholes
            cylinder(d = 8, h = sw + 2); translate([-2, 0, 0]) cube([4, 9, sw + 2]);
        }
        for (i = [0:4]) translate([-1, D/2 - 25, 14 + i*9]) cube([W + 2, 50, 4]);               // side vents
    }
}

module shell_door() {
    dh = ceil_z + t - base_h - 1;
    difference() {
        cube([145, dh, 3]);
        translate([145/2, dh*0.42, 2.4]) linear_extrude(1) text("RipeSense", size = 14, halign = "center", valign = "center");
    }
}

module shell_lid(top = false) {
    difference() {
        translate([sw + 0.3, sw + 0.3, 0]) cube([W - 2*sw - 0.6, D - 2*sw - 0.6, 3]);
        for (b = bosses()) translate([b[0], b[1], -1]) cylinder(d = 3.4, h = 5);
        if (top) for (i = [0:4]) translate([W/2 - 35, 8 + i*8, -1]) cube([70, 4, 5]);
    }
}

module printed_assembly() {
    color([0.12, 0.12, 0.12]) shell_base();
    color([0.18, 0.18, 0.18]) translate([0, 0, base_h - t]) shell_chamber();
    color([0.12, 0.12, 0.12]) translate([0, 0, head_z]) shell_head();
    color([0.1, 0.1, 0.1, 0.3]) translate([0.5, D, base_h]) rotate([90, 0, 0]) shell_door();
    color("orange") translate([W/2, D/2, base_h]) cradle_insert();
    color("gold") translate([W/2, D/2, base_h + dish_floor + 30]) scale([1, 0.85, 0.85]) sphere(r = 32);
}

if (part == "assembly") assembly();
else if (part == "printed_assembly") printed_assembly();
else if (part == "shell_base") shell_base();
else if (part == "shell_chamber") shell_chamber();
else if (part == "shell_head") shell_head();
else if (part == "shell_door") shell_door();
else if (part == "shell_bottom") shell_lid(false);
else if (part == "shell_top") shell_lid(true);
else if (part == "cradle_insert") cradle_insert();
else if (part == "camera_clamp") camera_clamp();
else if (part == "cal_puck") cal_puck();
else if (part == "screen_clip") screen_clip();
else if (part == "magnet_block") magnet_block();
else if (part == "door_handle") door_handle();
else if (part == "plate") plate();
