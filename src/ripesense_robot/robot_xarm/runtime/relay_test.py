#!/usr/bin/env python3
from gpiozero import LED
import time

PIN = 17   # BCM GPIO27 = physical pin 13
relay = LED(PIN)

print("Testing GPIO%d (physical pin 13)" % PIN)
print("Watch/listen to the relay + measure pin 13 to GND\n")

try:
    for i in range(5):
        print("Cycle %d: HIGH (relay should click ON, pin reads ~3.3V)" % (i+1))
        relay.on()
        time.sleep(2)

        print("Cycle %d: LOW  (relay should click OFF, pin reads 0V)" % (i+1))
        relay.off()
        time.sleep(2)
finally:
    relay.off()
    relay.close()
    print("\nDone. GPIO released.")
