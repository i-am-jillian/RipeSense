#!/usr/bin/env python3
"""Rail check v2: clear the error and IMMEDIATELY command 'stay where you are', then watch. Watchdog halts on drift."""
import sys, time, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from xarm.wrapper import XArmAPI
    a = XArmAPI('192.168.1.232'); time.sleep(0.3)
R = lambda: a.get_linear_track_registers()[1]
def halt(why):
    a.set_linear_track_stop(); time.sleep(0.3); a.set_linear_track_enable(False)
    print('HALTED:', why, R(), flush=True); a.disconnect(); sys.exit(1)
def watch(sec, ref):
    t0 = time.time()
    while time.time() - t0 < sec:
        time.sleep(0.2); r = R()
        if r['error']: halt('rail error %s' % r['error'])
        if abs(r['pos'] - ref) > 3: halt('drifted %s -> %s mm' % (ref, r['pos']))
    return R()
def goto(t, timeout=40):
    print('goto %d ->' % t, a.set_linear_track_pos(t, speed=50, wait=False), flush=True); t0 = time.time()
    while time.time() - t0 < timeout:
        time.sleep(0.3); r = R()
        if r['error']: halt('rail error %s on the way to %d' % (r['error'], t))
        if abs(r['pos'] - t) <= 2: time.sleep(0.5); return R()
    halt('did not reach %d' % t)
r0 = R(); print('start', r0, flush=True)
p = r0['pos']
a.set_linear_track_speed(30)
a.clean_linear_track_error()
print('hold-here ->', a.set_linear_track_pos(p, speed=30, wait=False), flush=True)
r = watch(3.0, p); print('3 s holding still:', r, flush=True)
print('homing ->', a.set_linear_track_back_origin(wait=False), flush=True); t0 = time.time()
while time.time() - t0 < 60:
    time.sleep(0.3); r = R()
    if r['error']: halt('rail error %s while homing' % r['error'])
    if r['on_zero'] and abs(r['pos']) <= 2: break
else: halt('homing timeout')
print('HOMED', R(), flush=True)
print('at', goto(100), flush=True)
r = watch(2.0, R()['pos']); print('holding at 100:', r['pos'], flush=True)
print('RAIL OK', flush=True); a.disconnect()
