# Hosyond 3.5" 480×320 SPI screen on a Pi 5

These boards are ILI9486 displays with an XPT2046 (ADS7846-compatible) touch
controller, wired like the Waveshare 3.5" (A). The vendor `LCD-show` scripts
target older Pis/OS images and often break a Pi 5 on Bookworm, so don't run them.

**Most reliable path on Bookworm (X11 + fbdev):**

1. `sudo raspi-config` → Advanced → Wayland → **X11**.
2. Add to `/boot/firmware/config.txt`:
   ```
   dtparam=spi=on
   dtoverlay=piscreen,speed=16000000,rotate=90
   ```
   (`rotate=270` if upside down.) The touch controller comes up as an ADS7846 input.
3. Reboot. `ls /dev/fb*` should show `/dev/fb1` (or `/dev/fb0` if no HDMI).
4. Point X at the SPI framebuffer: create `/usr/share/X11/xorg.conf.d/99-spi-screen.conf`:
   ```
   Section "Device"
     Identifier "spi"
     Driver "fbdev"
     Option "fbdev" "/dev/fb1"
   EndSection
   ```
5. If touches are mirrored/rotated, add a `TransformationMatrix` for the ADS7846
   device in `/usr/share/X11/xorg.conf.d/40-libinput.conf` (e.g. `0 1 0 -1 0 1 0 0 1` for 90°).

**Fallback (from the handoff):** if the driver fights you, don't lose hours —
open `http://<pi-ip>:5000/` full-screen on a phone or laptop. The UI is the
same 480×320 page. The physical TTP223 tap pad still starts scans.

The screen sits on a 40-pin ribbon, not directly on the header, so I2C
(GPIO 2/3) and the GPIO pins in `config.yaml` stay free. The defaults there
(5, 6, 12, 13, 16, 26) already avoid the Waveshare-(A)-style pins: SPI0
(GPIO 7–11), touch IRQ GPIO 17, RST GPIO 24, DC GPIO 25. If your board's
silkscreen differs, check it before wiring.
