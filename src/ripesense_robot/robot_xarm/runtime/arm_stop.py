#!/usr/bin/env python3
"""STOP button: halt the arm's motion now (state 4 clears the motion queue), stop the rail, then the job
script is killed by the caller.  The arm is left powered and is set ready again so the next job can run."""
import time, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    from xarm.wrapper import XArmAPI
    a = XArmAPI('192.168.1.232'); time.sleep(0.2)
a.set_state(4)
try: a.set_linear_track_stop()
except Exception: pass
time.sleep(0.6); a.set_state(0); time.sleep(0.3)
print('arm halted, state', a.state, 'error', a.error_code); a.disconnect()
