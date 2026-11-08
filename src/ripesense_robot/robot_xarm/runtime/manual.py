#!/usr/bin/env python3
"""Hand-guiding (manual / teach) mode and saving drop-off poses.
  manual.py on               # arm becomes hand-movable (mode 2)
  manual.py save <name>      # store the current pose as <name> in ~/drop_poses.json (stays in manual mode)
  manual.py off              # back to normal position control (mode 0)"""
import json, os, sys, time, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from xarm.wrapper import XArmAPI
    a = XArmAPI('192.168.1.232'); time.sleep(0.3)
F = os.path.expanduser('~/drop_poses.json')
cmd = sys.argv[1] if len(sys.argv) > 1 else 'status'
if cmd == 'on':
    a.clean_warn(); a.clean_error(); a.motion_enable(True)
    a.set_mode(2); a.set_state(0); time.sleep(0.5)
elif cmd == 'off':
    a.set_mode(0); a.set_state(0); time.sleep(0.5)
elif cmd == 'save':
    name = sys.argv[2]
    poses = json.load(open(F)) if os.path.exists(F) else {}
    poses[name] = {'pose': [round(v, 2) for v in a.position], 'joints': [round(v, 3) for v in a.angles[:6]], 'saved': time.strftime('%Y-%m-%d %H:%M:%S')}
    json.dump(poses, open(F, 'w'), indent=1)
    print('saved', name, poses[name])
print('mode', a.mode, 'state', a.state, 'error', a.error_code, 'pose', [round(v) for v in a.position])
a.disconnect()
