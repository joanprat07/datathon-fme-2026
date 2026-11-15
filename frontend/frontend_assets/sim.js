// ---------------------------------------------------------------------------
// Synthetic shopper stream that follows the pandabuy algorithm (conv/).
//
// Products are a real sample of shop.catalog (category, subcategory, brand, price,
// 15-day purchases); shoppers are synthetic, with the real mix of pages, regions and
// recent views. A hidden "true" purchase probability stands in for the simulator.
// Our side: 6 candidate sources with the real quotas, a noisy estimate of a, sorting,
// exploration on slots 8-10 and the bought-product prior. Baseline: best sellers.
// CATALOG and PARAMS are injected by frontend.py.
// ---------------------------------------------------------------------------
const CATALOG = __CATALOG__;   // [{id, name, cat, sub, brand, price, pop}]
const PARAMS = __PARAMS__;     // {seed, exam[10], quotas, explore_eps, regions, pages, rv_share, target_cvr}

function mulberry32(a) {
  return function () {
    a |= 0; a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function hex(rnd, n) { let s = ""; for (let i = 0; i < n; i++) s += "0123456789abcdef"[Math.floor(rnd() * 16)]; return s; }
function pick(rnd, arr) { return arr[Math.floor(rnd() * arr.length)]; }
function pickW(rnd, items, w) { let r = rnd() * w.reduce((a, b) => a + b, 0); for (let i = 0; i < items.length; i++) { r -= w[i]; if (r <= 0) return items[i]; } return items[items.length - 1]; }
function gauss(rnd) { return Math.sqrt(-2 * Math.log(rnd() + 1e-12)) * Math.cos(2 * Math.PI * rnd()); }

const EXAM = PARAMS.exam;
const CATS = [...new Set(CATALOG.map(p => p.cat))];
const BY_CAT = Object.fromEntries(CATS.map(c => [c, CATALOG.filter(p => p.cat === c)]));
const BY_SUB = {}; CATALOG.forEach(p => (BY_SUB[p.sub] = BY_SUB[p.sub] || []).push(p));
const BRANDS = [...new Set(CATALOG.map(p => p.brand))];
const DEVICES = ["web", "ios", "android"];
const BASELINE = [...CATALOG].sort((a, b) => b.pop - a.pop).slice(0, 10);   // best sellers, same for everyone
const PAGE_F = { home: 0.22, category: 1.0, product: 1.0 };

// Hidden purchase probability of an examined product (the simulator's stand-in). K is calibrated below.
let K = 1;
function trueQ(u, p) {
  let aff = 0.35;
  if (u.subs.has(p.sub)) aff = 4.0; else if (u.cats.has(p.cat)) aff = 1.7;
  if (u.brands.has(p.brand)) aff *= 2.2;
  if (u.histSubs.has(p.sub)) aff *= 1.6;
  if (u.viewed.has(p.id)) aff *= 1.6;
  if (u.bought.has(p.id)) aff *= 0.35;
  const fit = Math.exp(-Math.abs(Math.log(p.price / u.budget)) * 0.8);
  return Math.min(0.6, K * PAGE_F[u.page] * (0.25 + p.pop) * aff * (0.35 + fit));
}
function listProb(u, items) { let none = 1; items.forEach((p, k) => { none *= 1 - EXAM[k] * trueQ(u, p); }); return 1 - none; }

// ---- our recommender -------------------------------------------------------
const exposure = new Map();   // how often each product has been shown (for exploration)

function sourceLists(u, rnd) {
  const jit = () => Math.exp(0.25 * gauss(rnd));
  const inSession = p => !u.hasRv || u.cats.has(p.cat);
  const score = (arr, f) => arr.map(p => [p, f(p) * jit()]).sort((a, b) => b[1] - a[1]).map(x => x[0]);
  const lists = {};
  if (u.hasRv) {
    const last = u.recent[u.recent.length - 1];
    lists.sim = score(CATALOG.filter(inSession), p => (p.sub === last.sub ? 3 : 0) + (u.cats.has(p.cat) ? 1 : 0)
      + (p.brand === last.brand ? 1.5 : 0) + Math.exp(-Math.abs(Math.log(p.price / last.price))) + 0.03 * p.pop);
    lists.cov = score(CATALOG.filter(p => u.subs.has(p.sub) || u.cats.has(p.cat)), p => (u.subs.has(p.sub) ? 2 : 0.6) * (0.2 + p.pop));
    lists.covx = score(CATALOG.filter(p => !u.cats.has(p.cat)), p => 0.2 + p.pop);
  }
  if (u.returning) lists.user = score(CATALOG.filter(inSession), p => (u.histSubs.has(p.sub) ? 1 : 0)
    + (u.brands.has(p.brand) ? 0.5 : 0) + (u.cats.has(p.cat) ? 0.25 : 0) + 0.02 * p.pop);
  lists.trend = score(CATALOG.filter(inSession), p => p.pop * Math.exp(0.5 * gauss(rnd)));
  lists.pop = score(CATALOG.filter(inSession), p => p.pop + (u.cats.has(p.cat) ? 0.3 : 0));
  return lists;
}

function recommend(u, rnd) {
  const quota = PARAMS.quotas[`${u.hasRv ? "rv" : "no_rv"}|${u.returning ? "hist" : "no_hist"}`];
  const lists = sourceLists(u, rnd), seen = new Map();
  for (const [src, n] of Object.entries(quota)) {
    let k = 0;
    for (const p of lists[src] || []) { if (k >= n) break; if (!seen.has(p.id)) { seen.set(p.id, { p, src }); k++; } }
  }
  const noise = u.returning ? 0.35 : 0.6;   // a good model, not an oracle; less sure about new shoppers
  const cands = [...seen.values()].map(c => {
    const q = trueQ(u, c.p) / (u.bought.has(c.p.id) ? 0.35 : 1);         // our model does not know the true effect...
    const prior = u.bought.has(c.p.id) ? Math.exp(PARAMS.bought_prior) : 1; // ...it starts from the bought prior
    return { ...c, a: q * prior * Math.exp(noise * gauss(rnd)) };
  }).sort((x, y) => y.a - x.a || y.p.price - x.p.price);
  const top = cands.slice(0, 10).map(c => ({ ...c, explore: false }));
  // exploration: slots 8-10, 5 % each, least-exposed of the next 30
  const pool = cands.slice(10, 40);
  for (const k of [7, 8, 9]) {
    if (rnd() < PARAMS.explore_eps && pool.length) {
      pool.sort((x, y) => (exposure.get(x.p.id) || 0) - (exposure.get(y.p.id) || 0));
      top[k] = { ...pool.shift(), explore: true };
    }
  }
  top.forEach(c => exposure.set(c.p.id, (exposure.get(c.p.id) || 0) + 1));
  return { recs: top, nCand: cands.length };
}

// ---- shopper stream ---------------------------------------------------------
function makeShopper(rnd, n, t) {
  const returning = rnd() > PARAMS.new_share / 100;
  const page = pickW(rnd, PARAMS.pages.map(p => p.page), PARAMS.pages.map(p => p.share));
  const hasRv = page !== "home" ? rnd() < PARAMS.rv_share / 100 + 0.1 : rnd() < 0.35;
  const cat1 = pick(rnd, CATS);
  const subs1 = [...new Set(BY_CAT[cat1].map(p => p.sub))];
  const sub1 = pick(rnd, subs1);
  const recent = [];
  if (hasRv) {
    const nV = 1 + Math.floor(rnd() * 4);
    for (let i = 0; i < nV; i++) recent.push(pick(rnd, rnd() < 0.7 ? BY_SUB[sub1] : BY_CAT[cat1]));
  }
  const brands = new Set(returning ? [pick(rnd, BRANDS), pick(rnd, BRANDS), ...(rnd() < 0.5 ? [pick(rnd, BY_SUB[sub1]).brand] : [])] : []);
  const histSubs = new Set(returning ? [rnd() < 0.6 ? sub1 : pick(rnd, subs1), pick(rnd, CATALOG).sub] : []);
  const bought = new Set(returning && rnd() < 0.3 ? [pick(rnd, BY_SUB[sub1]).id] : []);
  const ref = recent.length ? recent.reduce((a, p) => a + Math.log(p.price), 0) / recent.length : Math.log(pick(rnd, BY_CAT[cat1]).price);
  const u = {
    n, ts: new Date(t).toISOString(), request_id: "r_" + hex(rnd, 16), user_id: "u_" + hex(rnd, 8),
    is_new_user: !returning, returning, page, device: pick(rnd, DEVICES),
    region: pickW(rnd, PARAMS.regions.map(r => r[0]), PARAMS.regions.map(r => r[1])),
    hasRv, recent, cats: new Set(hasRv ? recent.map(p => p.cat) : returning ? [cat1] : []),
    subs: new Set(recent.map(p => p.sub)), viewed: new Set(recent.map(p => p.id)),
    brands, histSubs, bought, budget: Math.exp(ref + 0.3 * gauss(rnd)),
  };
  if (!hasRv && !returning) u.cats = new Set();
  return u;
}

// K so that the baseline converts like the real old carousel (target_cvr purchases per request)
(function calibrate() {
  const rnd = mulberry32(7);
  const us = []; for (let i = 0; i < 1500; i++) us.push(makeShopper(rnd, i, 0));
  let lo = 0.0005, hi = 0.2;
  for (let it = 0; it < 30; it++) {
    K = Math.sqrt(lo * hi);
    const m = us.reduce((a, u) => a + listProb(u, BASELINE), 0) / us.length;
    if (m > PARAMS.target_cvr / 100) hi = K; else lo = K;
  }
})();

function makeStream(seed) {
  const rnd = mulberry32(seed);
  let n = 0, t = Date.parse("2026-11-15T09:00:00Z");
  return function next() {
    n += 1; t += Math.round(-Math.log(rnd() + 1e-9) * 1000 / PARAMS.users_per_sec);
    const u = makeShopper(rnd, n, t);
    const { recs, nCand } = recommend(u, rnd);
    u.recs = recs; u.nCand = nCand;
    const ours = recs.map(c => c.p);
    u.pOurs = listProb(u, ours);
    u.pBase = listProb(u, BASELINE);
    // latency: log-normal around the measured p50 / p95
    u.latency = Math.round(1000 * PARAMS.age_p50 * Math.exp(Math.log(PARAMS.age_p95 / PARAMS.age_p50) / 1.645 * gauss(rnd)));
    // realised outcome for both systems with a common random number (fair comparison)
    const coin = rnd(), r2 = rnd();
    const outcome = (items, prob) => {
      if (coin >= prob) return null;
      const w = items.map((p, k) => EXAM[k] * trueQ(u, p));
      return pickW(() => r2, items, w);
    };
    u.boughtOurs = outcome(ours, u.pOurs);
    u.boughtBase = outcome(BASELINE, u.pBase);
    return u;
  };
}

const fmtEur = v => v.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 }) + " €";
