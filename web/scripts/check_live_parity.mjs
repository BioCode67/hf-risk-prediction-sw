// Python export == TypeScript port, on the parity cases baked into model.json.
// Run: node --experimental-strip-types web/scripts/check_live_parity.mjs
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { binReadings, liveFeatures, VITALS } from "../src/lib/live-features.ts";
import { contributions, predictRisk } from "../src/lib/live-model.ts";

const path = fileURLToPath(new URL("../public/live/model.json", import.meta.url));
const model = JSON.parse(readFileSync(path, "utf8"));
let failed = 0;
for (const c of model.parity) {
  const readings = c.readings.map((r) => ({ t: r.t, values: Object.fromEntries(VITALS.map((v) => [v, r[v]])) }));
  const f = liveFeatures(binReadings(readings, 10), model.features, model.window_steps, model.baseline_steps);
  let worst = 0;
  f.row.forEach((v, i) => {
    const want = c.features[i];
    const both = want === null ? (Number.isFinite(v) ? Infinity : 0) : Math.abs(v - want);
    worst = Math.max(worst, both);
  });
  const risk = predictRisk(model, f.row);
  const top = contributions(model, f.row).slice(0, 3).map((x) => x.feature).join(", ");
  const ok = worst < 1e-6 && Math.abs(risk - c.risk) < 1e-4;
  if (!ok) failed++;
  console.log(`${ok ? "OK  " : "FAIL"} ${c.kind.padEnd(10)} feature max|Δ|=${worst.toExponential(1)} risk ts=${risk.toFixed(5)} py=${c.risk.toFixed(5)} top: ${top}`);
}
process.exit(failed ? 1 : 0);
