# Banana touchscreen demo

Installed on the Raspberry Pi at `/home/ieee/ripesense-demo`.
The original sensor pipeline remains available; this lightweight demo runs separately
with `python3 -m ripesense.demo`, using Flask and Segno without the sensor/ML dependencies.

## Present it

1. Power the Pi and allow Wi-Fi and the desktop to start. The kiosk opens automatically.
2. Place your slightly green banana in the demonstration position.
3. Tap **Scan my fruit** on the fruit-neutral welcome screen. The 16-second sequence identifies the fruit (always banana in this demo), then shows camera, spectrum, gas, and result stages. A generated interior-chamber image illustrates the simulated view.
4. The result says **Slightly green**, firm texture, mild sweetness, and **1–2 days to ideal**.
   This is a scenario estimate for sweeter fresh eating at room temperature, not time to spoilage.
5. Tap **Get recipe · QR** when ready. Gemini writes a recipe for using this banana now.
6. Scan with a phone connected to UB Wi-Fi. Phone access from eduroam to this Pi on UB_Devices
   was confirmed by the user during setup.
7. Tap **Result**, then **Scan again** to repeat. Each scan requests a recipe from Gemini.

The screen and recipe page identify the ripeness analysis as a simulation. It does not
inspect a real banana or certify food safety. No real camera or sensors are used in this mode.

## Gemini and QR

The private API key is read on the Pi from `~/.config/ripesense/gemini.key` (mode 600).
It is never sent to the browser or embedded in a QR. The verified default model is
`gemini-2.5-flash`, called through Google's structured JSON REST endpoint:
https://ai.google.dev/api/generate-content

Gemini generates the recipe text; Segno generates the QR containing a link to its page.
If credentials, quota, or networking fail, the recipe is visibly labeled **Built-in fallback**.
Recipe pages persist across server restarts in `demo-data/`. Existing QR URLs depend on
the Pi retaining its IP address. New QR URLs use its current network address.
Phones need network access to the Pi; cellular-only access is not provided.

## Operation

```bash
# Connect from the Mac
ssh -i ~/.ssh/raspberrypi_ieee_setup ieee@10.84.194.127

# Server status / restart (resets scan screen; saved recipes remain)
systemctl status ripesense-demo
sudo systemctl restart ripesense-demo

# Launch kiosk manually if closed
sh ~/ripesense-demo/deploy/demo-kiosk.sh

# Shut down before unplugging
sudo poweroff
```

Server: `/etc/systemd/system/ripesense-demo.service`.
Desktop launch: `~/.config/autostart/ripesense-demo.desktop`.
Firefox uses a separate profile at `~/.mozilla/ripesense-demo`.

## Validation performed

- Four automated tests on the Pi: scan progression and persistence, QR/page endpoints,
  rejection of remote/cross-origin scan requests, recipe validation and fallback behavior.
- Real Gemini request produced **Warm Cinnamon Bananas** with recipe quantities and steps.
- Inspected the actual 480×320 kiosk screenshot; the large QR fits the screen.
- User scanned the QR on a phone and confirmed the recipe opens.

Run the demo tests with:
`python3 -m unittest discover -s tests -p test_demo.py -v`
