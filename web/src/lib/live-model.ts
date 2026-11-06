/**
 * Run the exported XGBoost model in the browser — no server, no network.
 *
 * `public/live/model.json` is written by `scripts/export_live_model.py` from
 * XGBoost's own JSON dump. Evaluation follows XGBoost exactly: go left when
 * `x < split`, follow `default_left` when the value is missing, sum the leaves,
 * add the base margin, squash with the logistic.
 *
 * Contributions use the path-based (Saabas) decomposition: walking from the
 * root, each split credits its feature with the change in the subtree's
 * expected value. It is exact in sum (contributions + bias = margin) and close
 * to TreeSHAP for shallow trees; the page labels it as an approximation.
 */

export interface LiveTree {
  left: number[];
  right: number[];
  feature: number[];
  split: number[];
  default_left: (number | boolean)[];
  weight: number[];
  cover: number[];
}

export interface LiveModel {
  about: string;
  source: string;
  features: string[];
  window_steps: number;
  baseline_steps: number;
  threshold: number;
  base_score: number;
  test_auprc: number;
  trees: LiveTree[];
  parity?: { kind: string; readings: Record<string, number>[]; features: (number | null)[]; risk: number }[];
}

const logit = (p: number) => Math.log(p / (1 - p));
const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

function isLeaf(tree: LiveTree, node: number): boolean {
  return tree.left[node] === -1;
}

function next(tree: LiveTree, node: number, x: number[]): number {
  const value = x[tree.feature[node]];
  if (!Number.isFinite(value)) return tree.default_left[node] ? tree.left[node] : tree.right[node];
  return value < tree.split[node] ? tree.left[node] : tree.right[node];
}

/** Cover-weighted expected leaf value below each node (computed once per tree). */
const expectations = new WeakMap<LiveTree, number[]>();
function expected(tree: LiveTree): number[] {
  const cached = expectations.get(tree);
  if (cached) return cached;
  const out = new Array<number>(tree.left.length).fill(0);
  const visit = (node: number): number => {
    if (isLeaf(tree, node)) {
      out[node] = tree.split[node]; // leaf value lives in split_conditions
      return out[node];
    }
    const l = tree.left[node];
    const r = tree.right[node];
    const el = visit(l);
    const er = visit(r);
    const cl = tree.cover[l];
    const cr = tree.cover[r];
    out[node] = cl + cr > 0 ? (cl * el + cr * er) / (cl + cr) : (el + er) / 2;
    return out[node];
  };
  visit(0);
  expectations.set(tree, out);
  return out;
}

export function predictRisk(model: LiveModel, x: number[]): number {
  let margin = logit(model.base_score);
  for (const tree of model.trees) {
    let node = 0;
    while (!isLeaf(tree, node)) node = next(tree, node, x);
    margin += tree.split[node];
  }
  return sigmoid(margin);
}

export interface Contribution {
  feature: string;
  value: number;
  contribution: number;
}

/** Per-feature contributions to the margin (log-odds), largest |effect| first. */
export function contributions(model: LiveModel, x: number[]): Contribution[] {
  const total = new Array<number>(model.features.length).fill(0);
  for (const tree of model.trees) {
    const e = expected(tree);
    let node = 0;
    while (!isLeaf(tree, node)) {
      const child = next(tree, node, x);
      total[tree.feature[node]] += e[child] - e[node];
      node = child;
    }
  }
  return model.features
    .map((feature, i) => ({ feature, value: x[i], contribution: total[i] }))
    .filter((c) => c.contribution !== 0)
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution));
}
