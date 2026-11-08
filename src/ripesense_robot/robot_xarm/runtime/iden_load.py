# Payload identification, runs detached on the Pi; log -> /tmp/iden_load.log
import time, math
from xarm.wrapper import XArmAPI
a = XArmAPI("192.168.1.232"); time.sleep(0.3)
a.clean_warn(); a.clean_error(); a.motion_enable(True); a.set_mode(0); a.set_state(0); time.sleep(0.5)
print("start: state", a.state, "error", a.error_code, "| joints", [round(v, 1) for v in a.angles[:6]], flush=True)
a.set_gripper_enable(True); a.set_gripper_position(850, wait=True)
t0 = time.time()
res = a.iden_tcp_load()
print("iden_tcp_load ->", res, "(%.0f s)" % (time.time() - t0), flush=True)
if res[0] == 0:
    mass, com = res[1][0], res[1][1:4]
    print("IDENTIFIED payload %.3f kg, centre of gravity %s mm" % (mass, [round(v, 1) for v in com]), flush=True)
    print("set_tcp_load:", a.set_tcp_load(mass, com)); time.sleep(0.5)
    print("save_conf:", a.save_conf()); time.sleep(0.5)
    print("payload now:", a.tcp_load, flush=True)
print("final: mode", a.mode, "state", a.state, "error", a.error_code, "| joints", [round(v, 1) for v in a.angles[:6]], flush=True)
print("DONE", flush=True)
a.disconnect()
