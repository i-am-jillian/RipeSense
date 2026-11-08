#!/usr/bin/env python3
"""Solve the xArm eye-in-hand calibration from samples.json (OpenCV 4.x calibrateHandEye; the Pi's
OpenCV 5 wheel lacks it).  usage: python handeye_solve.py samples.json [out_dir]"""
import json, math, os, sys
import cv2, numpy as np

# UFactory's value for their official D435 bracket (xyz m, rpy rad), for comparison
UF_EEF_TO_COLOR = [0.067052239, -0.0311387575, 0.021611456, -0.004202176, -0.00848499, 1.5898775]


def rpy_to_R(r, p, y):
    """xArm convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def R_to_rpy(R):
    p = math.asin(-max(-1.0, min(1.0, R[2, 0])))
    return math.atan2(R[2, 1], R[2, 2]), p, math.atan2(R[1, 0], R[0, 0])


def T(R, t):
    M = np.eye(4); M[:3, :3] = R; M[:3, 3] = np.ravel(t); return M


def angle_deg(Ra, Rb):
    return math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(Ra.T @ Rb) - 1) / 2))))



def solve(samples):
    Rg, tg, Rt, tt = [], [], [], []
    for s in samples:
        x, y, z, r, p, yw = s['arm']                       # mm, rad
        Rg.append(rpy_to_R(r, p, yw)); tg.append(np.array([x, y, z]) / 1000.0)
        Rt.append(cv2.Rodrigues(np.array(s['rvec']))[0]); tt.append(np.array(s['tvec']))
    results = {}
    for name, m in [('TSAI', cv2.CALIB_HAND_EYE_TSAI), ('PARK', cv2.CALIB_HAND_EYE_PARK),
                    ('HORAUD', cv2.CALIB_HAND_EYE_HORAUD), ('DANIILIDIS', cv2.CALIB_HAND_EYE_DANIILIDIS)]:
        R, t = cv2.calibrateHandEye(Rg, tg, Rt, tt, method=m)
        X = T(R, t)                                          # camera -> flange (cam2gripper)
        # consistency: board pose in the base frame should be identical for every sample
        boards = [T(Rg[i], tg[i]) @ X @ T(Rt[i], tt[i]) for i in range(len(samples))]
        pos = np.array([b[:3, 3] for b in boards]) * 1000
        spread_mm = float(np.linalg.norm(pos - pos.mean(0), axis=1).mean())
        rot_spread = float(np.mean([angle_deg(boards[0][:3, :3], b[:3, :3]) for b in boards[1:]])) if len(boards) > 1 else 0
        results[name] = dict(xyz_m=[float(v) for v in t.ravel()], rpy_rad=[float(v) for v in R_to_rpy(R)],
                             board_pos_spread_mm=round(spread_mm, 2), board_rot_spread_deg=round(rot_spread, 2),
                             board_in_base_mm=[round(float(v), 1) for v in pos.mean(0)])
    best = min(results, key=lambda k: results[k]['board_pos_spread_mm'])
    uf = T(rpy_to_R(*UF_EEF_TO_COLOR[3:]), UF_EEF_TO_COLOR[:3])
    b = results[best]; Xb = T(rpy_to_R(*b['rpy_rad']), b['xyz_m'])
    diff_mm = float(np.linalg.norm((Xb[:3, 3] - uf[:3, 3]) * 1000)); diff_deg = angle_deg(Xb[:3, :3], uf[:3, :3])
    print('\n==== hand-eye result (camera -> flange), %d samples ====' % len(samples))
    for k, v in results.items():
        print('%-10s xyz_mm %s  rpy_deg %s  | board spread %.1f mm / %.2f deg%s' % (
            k, [round(1000 * a, 1) for a in v['xyz_m']], [round(math.degrees(a), 2) for a in v['rpy_rad']],
            v['board_pos_spread_mm'], v['board_rot_spread_deg'], '   <-- best' if k == best else ''))
    print('UFactory bracket   xyz_mm %s  rpy_deg %s' % ([round(1000 * a, 1) for a in UF_EEF_TO_COLOR[:3]],
                                                       [round(math.degrees(a), 2) for a in UF_EEF_TO_COLOR[3:]]))
    print('difference best vs UFactory: %.1f mm, %.2f deg' % (diff_mm, diff_deg))
    with open(OUT + '/result.yaml', 'w') as f:
        f.write('# camera(color optical) -> flange, xArm rpy convention, from %d samples\n' % len(samples))
        f.write('method: %s\nEULER_EEF_TO_COLOR_OPT: [%s]\n' % (best, ', '.join('%.6f' % v for v in b['xyz_m'] + b['rpy_rad'])))
        f.write('board_pos_spread_mm: %.2f\nboard_rot_spread_deg: %.2f\n' % (b['board_pos_spread_mm'], b['board_rot_spread_deg']))
        f.write('diff_vs_ufactory_mm: %.1f\ndiff_vs_ufactory_deg: %.2f\n' % (diff_mm, diff_deg))
        f.write('all_methods: %s\n' % json.dumps(results))
    print('saved', OUT + '/result.yaml')


if __name__ == '__main__':
    OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.abspath(sys.argv[1]))
    solve(json.load(open(sys.argv[1])))
