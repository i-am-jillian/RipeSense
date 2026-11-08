#!/usr/bin/env python3
"""Automatic recovery after a failed mission step: clear the arm error; if a fruit is still in the gripper,
pull it in, lift it and lower it into the bin (it was being picked for the bin); then go to the start pose.
Exit 0 = recovered and ready, 1 = could not recover (mission must stop)."""
import json, math, os, sys, time, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from xarm.wrapper import XArmAPI
    a = XArmAPI('192.168.1.232'); time.sleep(0.3)
def log(*m): print('   [recover]', *m, flush=True)
def wait(what):
    t0 = time.time(); time.sleep(0.4)
    while a.state == 1 and time.time() - t0 < 60: time.sleep(0.1)
    if a.error_code: raise RuntimeError('error %s during %s' % (a.error_code, what))
def line(x, y, z, r, p, yw, what):
    c = a.set_position(x=x, y=y, z=z, roll=r, pitch=p, yaw=yw, speed=30, mvacc=200, wait=False)
    if c:                                             # straight line not possible -> same target in joint space
        log('%s: straight move rejected (code %s), trying a joint move' % (what, c))
        a.clean_error(); a.set_state(0); time.sleep(0.3)
        ik, j = a.get_inverse_kinematics([x, y, z, r, p, yw], input_is_radian=False, return_is_radian=False)
        if ik != 0 or a.set_servo_angle(angle=list(j[:6]), speed=15, mvacc=100, wait=False): raise RuntimeError(what + ' rejected')
    wait(what)
def joints(j, what):
    if a.set_servo_angle(angle=list(j), speed=20, mvacc=150, wait=False): raise RuntimeError(what + ' rejected')
    wait(what)
try:
    log('arm state %s, error %s' % (a.state, a.error_code))
    a.clean_error(); a.clean_warn(); a.motion_enable(True); a.set_mode(0); a.set_state(0); time.sleep(0.6)
    if a.error_code: raise RuntimeError('error %s does not clear' % a.error_code)
    code, g = a.get_gripper_position(); holding = code == 0 and g is not None and 40 < g < 800
    x, y, z, r, p, yw = a.position
    log('gripper %s -> %s' % (g, 'HOLDING a fruit' if holding else 'empty'))
    if z < 60:                                         # low (near the table): straight up first
        try: line(x, y, z + 60, r, p, yw, 'lift'); z += 60
        except RuntimeError as e: log('lift skipped (%s)' % e); a.clean_error(); a.set_state(0); time.sleep(0.4)
    k = min(1.0, 450 / max(1.0, math.hypot(x, y)))
    if k < 1.0: line(x * k, y * k, z, r, p, yw, 'pull in'); x, y = x * k, y * k
    if z < 200: line(x, y, 200, r, p, yw, 'rise')
    if holding:
        BIN = sys.argv[1] if len(sys.argv) > 1 else 'ripe_bin'      # spoiled phase passes spoiled_bin
        b = json.load(open(os.path.expanduser('~/drop_poses.json')))[BIN]; bx, by, bz, br, bp, byw = b['pose']
        from above import above_joints
        above = above_joints(a, b['joints'], 120)
        if not above: raise RuntimeError('no pose above the %s' % BIN)
        joints(above[:6], 'above bin'); line(bx, by, bz, br, bp, byw, 'down into bin')
        a.set_gripper_speed(1500); a.set_gripper_position(850, wait=True); time.sleep(0.8)
        log('fruit released in the %s' % BIN.replace('_', ' ')); line(bx, by, bz + 120, br, bp, byw, 'up out of bin')
    c, home = a.get_inverse_kinematics([263.7, 36.7, 459, 180, 0, 0], input_is_radian=False, return_is_radian=False)
    joints(home[:6], 'start pose')
    log('RECOVERED: back at the start pose, error %s' % a.error_code); a.disconnect(); sys.exit(0)
except Exception as e:  # noqa: BLE001
    log('RECOVERY FAILED: %s' % e); a.disconnect(); sys.exit(1)
