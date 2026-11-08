#include <Wire.h>
#include <Adafruit_AS7341.h>

Adafruit_AS7341 sensor;
// Fixed for dark, white and EVERY fruit scan. After changing these, recalibrate.
constexpr uint8_t ATIME_VALUE = 99;
constexpr uint16_t ASTEP_VALUE = 999;
constexpr as7341_gain_t GAIN_VALUE = AS7341_GAIN_16X;
constexpr uint16_t GAIN_MULTIPLIER = 16;
// Default: stable external white lamp, sensor LED off. Optional fixed breakout LED.
constexpr bool USE_BREAKOUT_LED = true;
constexpr uint16_t LED_CURRENT_MA = 4;
uint32_t sequenceNumber = 0;
bool ready = false;

const as7341_color_channel_t visibleChannels[] = {
  AS7341_CHANNEL_415nm_F1, AS7341_CHANNEL_445nm_F2,
  AS7341_CHANNEL_480nm_F3, AS7341_CHANNEL_515nm_F4,
  AS7341_CHANNEL_555nm_F5, AS7341_CHANNEL_590nm_F6,
  AS7341_CHANNEL_630nm_F7, AS7341_CHANNEL_680nm_F8
};

bool readRegister(uint8_t address, uint8_t &value) {
  Wire.beginTransmission(AS7341_I2CADDR_DEFAULT);
  Wire.write(address);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(uint8_t(AS7341_I2CADDR_DEFAULT), uint8_t(1)) != 1) return false;
  value = Wire.read();
  return true;
}
bool writeRegister(uint8_t address, uint8_t value) {
  Wire.beginTransmission(AS7341_I2CADDR_DEFAULT);
  Wire.write(address); Wire.write(value);
  return Wire.endTransmission() == 0;
}
void errorRow(const char *code) {
  Serial.print("ERR,"); Serial.print(sequenceNumber++);
  Serial.print(','); Serial.print(millis()); Serial.print(','); Serial.println(code);
}
void reportLED() {
  uint8_t config = 0, led = 0;
  const bool bankOK = sensor.setBank(true);
  const bool readOK = bankOK && readRegister(AS7341_CONFIG, config) && readRegister(AS7341_LED, led);
  const bool restoreOK = sensor.setBank(false);
  if (!readOK || !restoreOK) { errorRow("LED_READBACK_FAILED"); return; }
  Serial.print("# LED readback: control_enabled="); Serial.print((config & 0x08) != 0);
  Serial.print(" active="); Serial.print((led & 0x80) != 0);
  Serial.print(" current_ma="); Serial.println((led & 0x7F) * 2 + 4);
}
void header() {
  Serial.println("kind,seq,millis,F1,F2,F3,F4,F5,F6,F7,F8,gain,atime,astep,led_ma,saturated");
}
void setup() {
  Serial.begin(115200);
  const uint32_t started = millis();
  while (!Serial && millis() - started < 5000) delay(10);
  Wire.begin();  // UNO R4 main SDA/SCL header (Wire), not its separate Qwiic bus.
  header();
  if (!sensor.begin()) { errorRow("SENSOR_NOT_FOUND_AT_0x39"); return; }
  if (!sensor.setATIME(ATIME_VALUE) || !sensor.setASTEP(ASTEP_VALUE) ||
      !sensor.setGain(GAIN_VALUE) || !sensor.setLEDCurrent(LED_CURRENT_MA) ||
      !sensor.enableLED(USE_BREAKOUT_LED)) {
    errorRow("CONFIG_WRITE_FAILED"); return;
  }
  if (sensor.getATIME() != ATIME_VALUE || sensor.getASTEP() != ASTEP_VALUE ||
      sensor.getGain() != GAIN_VALUE) { errorRow("CONFIG_VERIFY_FAILED"); return; }
  // Latch saturation across BOTH SMUX integrations, including analog saturation.
  // INTENAB.ASIEN enables STATUS.ASAT; STATUS is write-one-to-clear (datasheet).
  uint8_t interruptEnable;
  if (!readRegister(AS7341_INTENAB, interruptEnable) ||
      !writeRegister(AS7341_INTENAB, interruptEnable | 0x80)) {
    errorRow("SATURATION_CONFIG_FAILED"); return;
  }
  ready = true;
}
void loop() {
  if (!ready) { errorRow("NOT_READY_CHECK_WIRING_AND_RESET"); delay(1000); return; }
  if (sequenceNumber % 10 == 0) { header(); reportLED(); } // Allows attaching without resetting board.
  if (!writeRegister(AS7341_STATUS, 0x80)) { errorRow("STATUS_CLEAR_FAILED"); delay(500); return; }
  if (!sensor.readAllChannels()) { errorRow("READ_ALL_CHANNELS_FAILED"); delay(500); return; }
  uint8_t status;
  if (!readRegister(AS7341_STATUS, status)) { errorRow("STATUS_READ_FAILED"); delay(500); return; }
  const uint32_t integrationCounts = (uint32_t(ATIME_VALUE) + 1) * (uint32_t(ASTEP_VALUE) + 1);
  const uint32_t ceiling = integrationCounts < 65535UL ? integrationCounts : 65535UL;
  bool saturated = (status & 0x80) != 0;
  for (uint8_t i = 0; i < 8; ++i)
    if (sensor.getChannel(visibleChannels[i]) >= ceiling * 0.98f) saturated = true;
  Serial.print("DATA,"); Serial.print(sequenceNumber++); Serial.print(','); Serial.print(millis());
  // Named channels avoid accidentally treating duplicate clear/NIR slots as F5/F6.
  for (uint8_t i = 0; i < 8; ++i) { Serial.print(','); Serial.print(sensor.getChannel(visibleChannels[i])); }
  Serial.print(','); Serial.print(GAIN_MULTIPLIER);
  Serial.print(','); Serial.print(ATIME_VALUE);
  Serial.print(','); Serial.print(ASTEP_VALUE);
  Serial.print(','); Serial.print(USE_BREAKOUT_LED ? LED_CURRENT_MA : 0);
  Serial.print(','); Serial.println(saturated ? 1 : 0);
  delay(100);
}
