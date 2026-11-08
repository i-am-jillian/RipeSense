#!/usr/bin/env python3
"""Careful linear-rail test with a watchdog: any uncommanded motion or rail error -> stop + disable."""
import sys, time, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from xarm.wrapper import XArmAPI
    a = XArmAPI('192.168.1.232'); time.sleep(0.3)
R = lambda: a.get_linear_track_registers()[1]
def halt(why):
    a.set_linear_track_stop(); time.sleep(0.3); a.set_linear_track_enable(False)
    print('HALTED:', why, R(), flush=True); a.disconnect(); sys.exit(1)
def watch(target=None, timeout=60, still=False):
    t0 = time.time(); last = R()['pos']; stable = 0
    while time.time() - t0 < timeout:
        time.sleep(0.3); r = R()
        if r['error']: halt('rail error %s' % r['error'])
        if still and abs(r['pos'] - last) > 3: halt('moved by itself (%s -> %s mm)' % (last, r['pos']))
        if target is not None and abs(r['pos'] - target) <= 2: return r
        if target == 'zero' and r['on_zero']: return r
        last = r['pos']
    if still: return R()
    halt('did not reach %s in %ss' % (target, timeout))
print('start', R(), flush=True)
a.clean_linear_track_error(); time.sleep(0.5)
r = R(); print('after clear', r, flush=True)
if r['error']: halt('error does not clear')
print('enable ->', a.set_linear_track_enable(True), flush=True)
r = watch(still=True, timeout=3); print('3 s after enable (should not move):', r, flush=True)
a.set_linear_track_speed(50)
print('homing ->', a.set_linear_track_back_origin(wait=False), flush=True)
r = watch('zero', timeout=60); print('HOMED:', r, flush=True)
for tgt in (100, 200):
    print('goto %d ->' % tgt, a.set_linear_track_pos(tgt, speed=50, wait=False), flush=True)
    r = watch(tgt, timeout=30); print('  at', r['pos'], 'mm, error', r['error'], flush=True)
    time.sleep(1); r2 = R(); print('  1 s later (holding):', r2['pos'], 'mm', flush=True)
print('RAIL TEST PASSED', flush=True)
a.disconnect()
