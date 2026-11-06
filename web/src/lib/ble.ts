/**
 * Read consumer vital-sign devices straight from the browser (Web Bluetooth).
 *
 * Chrome / Edge on a laptop, served over https or localhost. Each connector asks
 * the user to pick a device (a browser permission prompt — Web Bluetooth cannot
 * scan silently), subscribes to its measurement characteristic and calls
 * `onReading` with whatever vitals that packet carries.
 *
 * Standard Bluetooth SIG profiles first — any device that implements them works:
 *   Heart Rate 0x180D · Pulse Oximeter 0x1822 · Blood Pressure 0x1810 · Health Thermometer 0x1809
 * Plus two inexpensive oximeters whose protocols are publicly documented:
 *   BerryMed BM1000 series (github.com/zh2x/BCI_Protocol) · Viatom/Wellue PC-60FW.
 *
 * Blood pressure and thermometers send one value when a measurement finishes
 * (an *indication*), not a stream: connect first, then take the measurement.
 */

import type { Vital } from "./live-features";

export type DeviceKind = "heart_rate" | "plx" | "berrymed" | "pc60fw" | "blood_pressure" | "thermometer";

export interface DeviceSpec {
  kind: DeviceKind;
  label: string;
  hint: string;
  vitals: Vital[];
}

export const DEVICES: DeviceSpec[] = [
  { kind: "heart_rate", label: "심박계 (표준)", hint: "Polar H10 등 표준 심박 서비스 기기", vitals: ["pulse"] },
  { kind: "berrymed", label: "산소포화도 (BerryMed)", hint: "BM1000 계열 · 공개 프로토콜", vitals: ["spo2", "pulse"] },
  { kind: "pc60fw", label: "산소포화도 (PC-60FW)", hint: "Viatom/Wellue · 공개 디코더 기반", vitals: ["spo2", "pulse"] },
  { kind: "plx", label: "산소포화도 (표준)", hint: "표준 Pulse Oximeter 서비스 기기", vitals: ["spo2", "pulse"] },
  { kind: "blood_pressure", label: "혈압계 (표준)", hint: "연결 후 측정 버튼 → 결과 1회 전송", vitals: ["sbp", "dbp", "pulse"] },
  { kind: "thermometer", label: "체온계 (표준)", hint: "연결 후 측정 → 결과 1회 전송", vitals: ["temperature"] },
];

type Values = Partial<Record<Vital, number>>;

// --- IEEE-11073 number formats used by the SIG health profiles ---------------

/** 16-bit SFLOAT: 4-bit signed exponent, 12-bit signed mantissa. */
export function sfloat(raw: number): number {
  if ([0x07ff, 0x0800, 0x07fe, 0x0802, 0x0801].includes(raw)) return NaN; // NaN, NRes, ±INF, reserved
  let mantissa = raw & 0x0fff;
  let exponent = raw >> 12;
  if (mantissa >= 0x0800) mantissa -= 0x1000;
  if (exponent >= 0x08) exponent -= 0x10;
  return mantissa * 10 ** exponent;
}

/** 32-bit FLOAT: 8-bit signed exponent, 24-bit signed mantissa. */
export function float32(raw: number): number {
  let mantissa = raw & 0x00ffffff;
  let exponent = (raw >> 24) & 0xff;
  if (mantissa === 0x007fffff) return NaN;
  if (mantissa >= 0x800000) mantissa -= 0x1000000;
  if (exponent >= 0x80) exponent -= 0x100;
  return mantissa * 10 ** exponent;
}

// --- Packet parsers (pure, unit-testable) ------------------------------------

export function parseHeartRate(view: DataView): Values {
  const flags = view.getUint8(0);
  const bpm = flags & 0x01 ? view.getUint16(1, true) : view.getUint8(1);
  return bpm > 0 ? { pulse: bpm } : {};
}

export function parsePlxContinuous(view: DataView): Values {
  return { spo2: sfloat(view.getUint16(1, true)), pulse: sfloat(view.getUint16(3, true)) };
}

export function parseBloodPressure(view: DataView): Values {
  const flags = view.getUint8(0);
  const kpa = flags & 0x01;
  const scale = kpa ? 7.50062 : 1;
  let offset = 7; // flags(1) + systolic, diastolic, MAP (2 each)
  if (flags & 0x02) offset += 7; // timestamp
  const out: Values = {
    sbp: sfloat(view.getUint16(1, true)) * scale,
    dbp: sfloat(view.getUint16(3, true)) * scale,
  };
  if (flags & 0x04 && view.byteLength >= offset + 2) out.pulse = sfloat(view.getUint16(offset, true));
  return out;
}

export function parseThermometer(view: DataView): Values {
  const flags = view.getUint8(0);
  const value = float32(view.getUint32(1, true));
  return { temperature: flags & 0x01 ? ((value - 32) * 5) / 9 : value };
}

/**
 * BerryMed BCI protocol: 5-byte frames, the first byte has bit 7 set.
 * pulse = byte3 | (byte2 bit6 << 1); SpO2 = byte4; 127/255 mean "no reading".
 */
export function parseBerryMed(bytes: Uint8Array): Values {
  let latest: Values = {};
  for (let i = 0; i + 4 < bytes.length; i++) {
    if (!(bytes[i] & 0x80)) continue;
    const frame = bytes.subarray(i, i + 5);
    if (frame.slice(1).some((b) => b & 0x80)) continue; // not aligned on a frame
    const pulse = frame[3] | ((frame[2] & 0x40) << 1);
    const spo2 = frame[4];
    latest = {
      ...(pulse > 0 && pulse < 255 ? { pulse } : {}),
      ...(spo2 > 0 && spo2 <= 100 ? { spo2 } : {}),
    };
    i += 4;
  }
  return latest;
}

/** PC-60FW: frames start AA 55 0F 08 01, then SpO2, pulse rate. */
export function parsePc60fw(bytes: Uint8Array): Values {
  for (let i = 0; i + 6 < bytes.length; i++) {
    if (bytes[i] === 0xaa && bytes[i + 1] === 0x55 && bytes[i + 2] === 0x0f && bytes[i + 3] === 0x08 && bytes[i + 4] === 0x01) {
      const spo2 = bytes[i + 5];
      const pulse = bytes[i + 6];
      return {
        ...(spo2 > 0 && spo2 <= 100 ? { spo2 } : {}),
        ...(pulse > 0 && pulse < 255 ? { pulse } : {}),
      };
    }
  }
  return {};
}

// --- Connection ----------------------------------------------------------------

const UUID = {
  berrymedService: "49535343-fe7d-4ae5-8fa9-9fafd205e455",
  berrymedNotify: "49535343-1e4d-4bd9-ba61-23c647249616",
  nordicUart: "6e400001-b5a3-f393-e0a9-e50e24dcca9e",
  nordicUartTx: "6e400003-b5a3-f393-e0a9-e50e24dcca9e",
};

interface Plan {
  service: string | number;
  characteristic: string | number;
  filters: Record<string, unknown>[];
  parse: (view: DataView) => Values;
}

const PLANS: Record<DeviceKind, Plan> = {
  heart_rate: { service: "heart_rate", characteristic: "heart_rate_measurement", filters: [{ services: ["heart_rate"] }], parse: parseHeartRate },
  plx: { service: 0x1822, characteristic: 0x2a5f, filters: [{ services: [0x1822] }], parse: parsePlxContinuous },
  blood_pressure: { service: "blood_pressure", characteristic: "blood_pressure_measurement", filters: [{ services: ["blood_pressure"] }], parse: parseBloodPressure },
  thermometer: { service: "health_thermometer", characteristic: "temperature_measurement", filters: [{ services: ["health_thermometer"] }], parse: parseThermometer },
  berrymed: {
    service: UUID.berrymedService,
    characteristic: UUID.berrymedNotify,
    filters: [{ services: [UUID.berrymedService] }, { namePrefix: "BerryMed" }],
    parse: (view) => parseBerryMed(new Uint8Array(view.buffer, view.byteOffset, view.byteLength)),
  },
  pc60fw: {
    service: UUID.nordicUart,
    characteristic: UUID.nordicUartTx,
    filters: [{ namePrefix: "PC-60" }, { namePrefix: "OxySmart" }, { services: [UUID.nordicUart] }],
    parse: (view) => parsePc60fw(new Uint8Array(view.buffer, view.byteOffset, view.byteLength)),
  },
};

export function bluetoothAvailable(): boolean {
  return typeof navigator !== "undefined" && "bluetooth" in navigator;
}

export interface Connection {
  name: string;
  disconnect: () => void;
}

/** Prompt for a device of `kind`, subscribe, and stream parsed values. */
export async function connectDevice(
  kind: DeviceKind,
  onReading: (values: Values) => void,
  onDisconnect: () => void,
): Promise<Connection> {
  if (!bluetoothAvailable()) {
    throw new Error("이 브라우저는 Web Bluetooth를 지원하지 않습니다. 노트북의 Chrome 또는 Edge를 사용해 주세요.");
  }
  const plan = PLANS[kind];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const bluetooth = (navigator as any).bluetooth;
  const device = await bluetooth.requestDevice({ filters: plan.filters, optionalServices: [plan.service] });
  device.addEventListener("gattserverdisconnected", onDisconnect);
  const server = await device.gatt.connect();
  const service = await server.getPrimaryService(plan.service);
  const characteristic = await service.getCharacteristic(plan.characteristic);
  characteristic.addEventListener("characteristicvaluechanged", (event: Event) => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    const view: DataView = (event.target as any).value;
    try {
      const values = plan.parse(view);
      if (Object.keys(values).length) onReading(values);
    } catch {
      // A malformed packet is dropped; the next one usually parses.
    }
  });
  await characteristic.startNotifications();
  return { name: device.name ?? "기기", disconnect: () => device.gatt.connected && device.gatt.disconnect() };
}
