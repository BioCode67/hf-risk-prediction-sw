// Device packet parsers against hand-built packets from the published specs.
// Run: node --experimental-strip-types web/scripts/check_ble_parsers.mjs
import assert from "node:assert/strict";
import { float32, parseBerryMed, parseBloodPressure, parseHeartRate, parsePc60fw, parsePlxContinuous, parseThermometer, sfloat } from "../src/lib/ble.ts";

const dv = (bytes) => new DataView(new Uint8Array(bytes).buffer);
// SFLOAT 97 = mantissa 97, exponent 0 -> 0x0061; 36.6 = 366 x 10^-1 -> 0xF16E
assert.equal(sfloat(0x0061), 97);
assert.ok(Math.abs(sfloat(0xf16e) - 36.6) < 1e-9);
assert.ok(Number.isNaN(sfloat(0x07ff)));
assert.ok(Math.abs(float32(0xff00016e) - 36.6) < 1e-9);
// Heart rate: uint8 and uint16 forms
assert.deepEqual(parseHeartRate(dv([0x00, 72])), { pulse: 72 });
assert.deepEqual(parseHeartRate(dv([0x01, 0x2c, 0x01])), { pulse: 300 });
// PLX continuous: flags, SpO2 97, PR 72
assert.deepEqual(parsePlxContinuous(dv([0x00, 0x61, 0x00, 0x48, 0x00])), { spo2: 97, pulse: 72 });
// Blood pressure mmHg 120/80, MAP 93, pulse 70 (flags: pulse present)
const bp = parseBloodPressure(dv([0x04, 120, 0, 80, 0, 93, 0, 70, 0]));
assert.deepEqual(bp, { sbp: 120, dbp: 80, pulse: 70 });
// Thermometer Celsius 36.6
assert.ok(Math.abs(parseThermometer(dv([0x00, 0x6e, 0x01, 0x00, 0xff])).temperature - 36.6) < 1e-9);
// BerryMed: two frames, second wins; PR 130 needs bit6 of byte2 (130 = 0x82 -> 0x02 | bit7)
assert.deepEqual(parseBerryMed(new Uint8Array([0x85, 0x30, 0x05, 72, 98, 0x86, 0x31, 0x45, 0x02, 95])), { pulse: 130, spo2: 95 });
assert.deepEqual(parseBerryMed(new Uint8Array([0x85, 0x30, 0x05, 127, 127])), { pulse: 127 }); // SpO2 127 = no finger
// PC-60FW
assert.deepEqual(parsePc60fw(new Uint8Array([0x00, 0xaa, 0x55, 0x0f, 0x08, 0x01, 96, 80, 0, 0])), { spo2: 96, pulse: 80 });
console.log("BLE parsers OK");
