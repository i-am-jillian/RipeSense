#!/usr/bin/env python3
"""One JSON line: arm, gripper, rail, model and job status (read only, no motion)."""
import json, os, time, glob
out = {'t': time.time()}
try:
    out['job'] = open('/tmp/job.status').read().split()
except Exception:
    out['job'] = None
md = os.path.expanduser('~/models/robot_fruit_model.onnx')
out['model'] = {'mtime': os.path.getmtime(md), 'classes': len(json.load(open(os.path.expanduser('~/models/robot_fruit_model_classes.json'))))}
out['busy'] = os.system('flock -n /tmp/job.lock true') != 0
out['moving_job'] = bool(os.popen('pgrep -f "^/home/sujal/ripesense_env/bin/python /home/sujal/(row_search|pick_all)\\.py"').read().strip())
out['autopilot'] = bool(os.popen('pgrep -f "^bash /home/sujal/autopilot\\.sh"').read().strip())
out['autopilot_stopping'] = os.path.exists('/tmp/autopilot.stop')
out['live_view'] = bool(os.popen('pgrep -f "^/home/sujal/ripesense_env/bin/python /home/sujal/live_view\\.py"').read().strip())
try:
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        from xarm.wrapper import XArmAPI
        a = XArmAPI('192.168.1.232', do_not_open=False); time.sleep(0.25)
    out['arm'] = {'connected': a.connected, 'state': a.state, 'error': a.error_code, 'warn': a.warn_code,
                  'pose': [round(v, 1) for v in a.position]}
    try:
        out['rail'] = a.get_linear_track_registers()[1]
    except Exception as e:  # noqa: BLE001
        out['rail'] = {'err': str(e)}
    code, g = a.get_gripper_position(); out['gripper'] = g if code == 0 else None
    a.disconnect()
except Exception as e:  # noqa: BLE001
    out['arm'] = {'connected': False, 'err': str(e)}
print(json.dumps(out, default=str))
