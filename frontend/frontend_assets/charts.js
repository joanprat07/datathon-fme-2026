// Small SVG chart helpers in the page style (no external libraries).
// Every function returns an SVG string; widths are viewBox units, so charts scale with their box.
const COL = { green: "#27f795", red: "#ff4f5e", violet: "#a594ff", amber: "#f5c84c", grid: "#262626",
  axis: "#636679", text: "#bdbdbd", dim: "#8d8d8d", white: "#ffffff" };
const MONO = 'font-family="Roboto Mono, monospace"';

function esc(s) { return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;"); }
function niceStep(range, n) {
  const raw = range / n, p = Math.pow(10, Math.floor(Math.log10(raw)));
  for (const m of [1, 2, 2.5, 5, 10]) if (m * p >= raw) return m * p;
  return 10 * p;
}
function ticks(max, n = 4) { const s = niceStep(max, n), out = []; for (let v = 0; v <= max + 1e-9; v += s) out.push(+v.toFixed(6)); return out; }
function text(x, y, s, o = {}) {
  return `<text x="${x}" y="${y}" ${MONO} font-size="${o.size || 11}" fill="${o.fill || COL.axis}" text-anchor="${o.anchor || "start"}"${o.weight ? ` font-weight="${o.weight}"` : ""}>${esc(s)}</text>`;
}
function svg(w, h, body) { return `<svg viewBox="0 0 ${w} ${h}" width="100%" preserveAspectRatio="xMidYMid meet" role="img">${body}</svg>`; }

// Vertical grouped bars. series: [{name, color, values}], labels: x labels.
function bars({ labels, series, w = 560, h = 260, fmt = v => v, yMax, highlight = -1, valueLabels = true }) {
  const L = 44, R = 10, T = 16, B = 30, iw = w - L - R, ih = h - T - B;
  const max = yMax || Math.max(...series.flatMap(s => s.values)) * 1.15;
  const tk = ticks(max), top = tk[tk.length - 1], Y = v => T + ih - (v / top) * ih;
  let g = "";
  tk.forEach(v => { g += `<line x1="${L}" x2="${w - R}" y1="${Y(v)}" y2="${Y(v)}" stroke="${COL.grid}" stroke-dasharray="2 4"/>` + text(L - 8, Y(v) + 4, fmt(v), { anchor: "end" }); });
  const bw = iw / labels.length, gap = bw * 0.22, sw = (bw - gap) / series.length;
  labels.forEach((lab, i) => {
    series.forEach((s, j) => {
      const v = s.values[i], x = L + i * bw + gap / 2 + j * sw, y = Y(v);
      const op = highlight < 0 || highlight === i ? 1 : 0.45;
      g += `<rect x="${x}" y="${y}" width="${sw - 2}" height="${T + ih - y}" fill="${s.color}" opacity="${op}"/>`;
      if (valueLabels && series.length === 1) g += text(x + (sw - 2) / 2, y - 5, fmt(v), { anchor: "middle", fill: COL.text, size: 10 });
    });
    g += text(L + i * bw + bw / 2, h - 10, lab, { anchor: "middle" });
  });
  return svg(w, h, g);
}

// Lines on a shared x. series: [{name, color, values, dash}]
function lines({ xs, series, w = 560, h = 260, fmt = v => v, xfmt = v => v, yMax, yMin = 0, area = false, dots = true, endLabels = true }) {
  const L = 44, R = 70, T = 16, B = 30, iw = w - L - R, ih = h - T - B;
  const max = yMax || Math.max(...series.flatMap(s => s.values)) * 1.1;
  const tk = ticks(max - yMin).map(v => v + yMin), top = tk[tk.length - 1];
  const X = i => L + (i / (xs.length - 1)) * iw, Y = v => T + ih - ((v - yMin) / (top - yMin)) * ih;
  let g = "";
  tk.forEach(v => { g += `<line x1="${L}" x2="${L + iw}" y1="${Y(v)}" y2="${Y(v)}" stroke="${COL.grid}" stroke-dasharray="2 4"/>` + text(L - 8, Y(v) + 4, fmt(v), { anchor: "end" }); });
  const every = Math.ceil(xs.length / 12);
  xs.forEach((x, i) => { if (i % every === 0 || (i === xs.length - 1 && i % every > every / 2)) g += text(X(i), h - 10, xfmt(x), { anchor: "middle" }); });
  series.forEach(s => {
    const pts = s.values.map((v, i) => `${X(i).toFixed(1)},${Y(v).toFixed(1)}`);
    if (area) g += `<polygon points="${X(0)},${Y(yMin)} ${pts.join(" ")} ${X(xs.length - 1)},${Y(yMin)}" fill="${s.color}" opacity="0.12"/>`;
    g += `<polyline points="${pts.join(" ")}" fill="none" stroke="${s.color}" stroke-width="2"${s.dash ? ` stroke-dasharray="${s.dash}"` : ""}/>`;
    if (dots) s.values.forEach((v, i) => { g += `<circle cx="${X(i)}" cy="${Y(v)}" r="2.6" fill="${s.color}"/>`; });
    const last = s.values.length - 1;
    if (endLabels) g += text(X(last) + 8, Y(s.values[last]) + 4, s.name, { fill: s.color, size: 10 });
  });
  return svg(w, h, g);
}

// Horizontal box plots. rows: [{label, box: [p5, q1, med, q3, p95], note}]; log x scale.
function boxes({ rows, w = 560, h, ticksAt = [2, 5, 10, 20, 50, 100, 200], fmt = v => v, color = COL.green, lo, hi }) {
  const L = 150, R = 56, T = 10, rowH = 30, B = 28; h = h || T + rows.length * rowH + B;
  lo = lo || Math.min(...rows.map(r => r.box[0])) * 0.8; hi = hi || Math.max(...rows.map(r => r.box[4])) * 1.1;
  const iw = w - L - R, X = v => L + (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) * iw;
  let g = "";
  ticksAt.filter(t => t >= lo && t <= hi).forEach(t => {
    g += `<line x1="${X(t)}" x2="${X(t)}" y1="${T}" y2="${h - B + 4}" stroke="${COL.grid}" stroke-dasharray="2 4"/>` + text(X(t), h - 8, fmt(t), { anchor: "middle" });
  });
  rows.forEach((r, i) => {
    const [a, q1, m, q3, b] = r.box, cy = T + i * rowH + rowH / 2;
    g += text(L - 12, cy + 4, r.label, { anchor: "end", fill: COL.text });
    g += `<line x1="${X(a)}" x2="${X(b)}" y1="${cy}" y2="${cy}" stroke="${COL.dim}"/>`;
    g += `<line x1="${X(a)}" x2="${X(a)}" y1="${cy - 5}" y2="${cy + 5}" stroke="${COL.dim}"/><line x1="${X(b)}" x2="${X(b)}" y1="${cy - 5}" y2="${cy + 5}" stroke="${COL.dim}"/>`;
    g += `<rect x="${X(q1)}" y="${cy - 9}" width="${X(q3) - X(q1)}" height="18" fill="${color}" fill-opacity="0.16" stroke="${color}"/>`;
    g += `<line x1="${X(m)}" x2="${X(m)}" y1="${cy - 9}" y2="${cy + 9}" stroke="${COL.white}" stroke-width="2"/>`;
    if (r.note) g += text(w - R + 8, cy + 4, r.note, { size: 10 });
  });
  return svg(w, h, g);
}

// Horizontal bars, optionally two-sided (diverging around 0). rows: [{label, values: [..]}]
function hbars({ rows, series, w = 560, fmt = v => v, rowH = 24, labelW = 160, diverging = false, max }) {
  const L = labelW, R = 56, T = 8, B = diverging ? 26 : 8, h = T + rows.length * rowH + B, iw = w - L - R;
  max = max || Math.max(...rows.flatMap(r => r.values.map(Math.abs))) * 1.05;
  const X0 = diverging ? L + iw / 2 : L, S = diverging ? iw / 2 / max : iw / max;
  let g = "";
  if (diverging) {
    g += `<line x1="${X0}" x2="${X0}" y1="${T}" y2="${h - B}" stroke="${COL.axis}"/>`;
    [-max, -max / 2, max / 2, max].forEach(v => { g += text(X0 + v * S, h - 8, (v > 0 ? "+" : "") + fmt(v), { anchor: "middle" }); });
  }
  const sh = (rowH - 6) / series.length;
  rows.forEach((r, i) => {
    const y = T + i * rowH;
    g += text(L - 10, y + rowH / 2 + 4, r.label, { anchor: "end", fill: COL.text });
    r.values.forEach((v, j) => {
      const x = v >= 0 ? X0 : X0 + v * S;
      g += `<rect x="${x}" y="${y + 3 + j * sh}" width="${Math.max(1, Math.abs(v) * S)}" height="${sh - 1}" fill="${series[j].color}"/>`;
      if (!diverging) g += text(X0 + v * S + 6, y + 3 + j * sh + sh / 2 + 4, fmt(v), { fill: COL.text, size: 10 });
    });
  });
  return svg(w, h, g);
}

// Concentration curve: share of purchases (y) held by the top x% of products.
function lorenz({ values, w = 560, h = 260, mark = 10 }) {
  const L = 44, R = 16, T = 14, B = 30, iw = w - L - R, ih = h - T - B;
  const X = f => L + f * iw, Y = f => T + ih - f * ih;
  let g = "";
  [0, .25, .5, .75, 1].forEach(v => {
    g += `<line x1="${L}" x2="${L + iw}" y1="${Y(v)}" y2="${Y(v)}" stroke="${COL.grid}" stroke-dasharray="2 4"/>` + text(L - 8, Y(v) + 4, (v * 100) + "%", { anchor: "end" });
    g += text(X(v), h - 10, (v * 100) + "%", { anchor: "middle" });
  });
  g += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(1)}" y2="${Y(1)}" stroke="${COL.axis}" stroke-dasharray="4 4"/>`;
  const pts = [[0, 0], ...values.map((v, i) => [(i + 1) / values.length, v])];
  g += `<polygon points="${pts.map(p => `${X(p[0])},${Y(p[1])}`).join(" ")} ${X(1)},${Y(0)}" fill="${COL.green}" opacity="0.12"/>`;
  g += `<polyline points="${pts.map(p => `${X(p[0])},${Y(p[1])}`).join(" ")}" fill="none" stroke="${COL.green}" stroke-width="2"/>`;
  const v = values[mark - 1];
  g += `<line x1="${X(mark / 100)}" x2="${X(mark / 100)}" y1="${Y(0)}" y2="${Y(v)}" stroke="${COL.white}" stroke-dasharray="2 3"/>`;
  g += `<circle cx="${X(mark / 100)}" cy="${Y(v)}" r="4" fill="${COL.white}"/>`;
  g += text(X(mark / 100) + 10, Y(v) + 14, `top ${mark}% of products = ${(v * 100).toFixed(0)}% of purchases`, { fill: COL.white, size: 11 });
  g += text(X(0.5), Y(0.5) + 18, "all products equal", { fill: COL.axis, size: 10 });
  return svg(w, h, g);
}

function legend(items) {
  return `<div class="legend">${items.map(([c, n, dash]) => `<span><i style="background:${dash ? "transparent" : c};${dash ? `border-top:2px dashed ${c};` : ""}"></i>${n}</span>`).join("")}</div>`;
}
