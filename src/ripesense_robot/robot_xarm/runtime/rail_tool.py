#!/usr/bin/env python3
"""Linear rail helper.  rail_tool.py home | status | goto <mm> [speed]"""
import sys, time
from xarm.wrapper import XArmAPI
a = XArmAPI('192.168.1.232'); time.sleep(0.4)
def st():
    r = a.get_linear_track_registers()[1]
    print('rail pos %s mm  status %s  error %s  enabled %s  on_zero %s | arm state %s err %s' % (
        r['pos'], r['status'], r['error'], r['is_enabled'], r['on_zero'], a.state, a.error_code), flush=True)
    return r
cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
try:
    st()
    if cmd in ('home', 'goto'):
        if a.state == 4 or a.error_code:
            a.clean_error(); a.clean_warn(); a.motion_enable(True); a.set_mode(0); a.set_state(0); time.sleep(0.5)
        print('enable', a.set_linear_track_enable(True), flush=True)
        sp = int(sys.argv[3]) if len(sys.argv) > 3 else 100
        print('speed', a.set_linear_track_speed(sp), flush=True)
    if cmd == 'home':
        t0 = time.time(); print('back_origin', a.set_linear_track_back_origin(wait=True), '%.1fs' % (time.time() - t0), flush=True)
        st()
    if cmd == 'goto':
        t0 = time.time(); print('goto', a.set_linear_track_pos(int(float(sys.argv[2])), wait=True, timeout=120), '%.1fs' % (time.time() - t0), flush=True)
        st()
finally:
    a.disconnect()
