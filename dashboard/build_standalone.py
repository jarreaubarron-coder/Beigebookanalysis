"""Build a self-contained dashboard page for GitHub Pages: docs/index.html.

dashboard/index.html is written for Claude's dashboard runtime (a global `dash` object with data, calc, draw,
params). This script wraps that same page in a normal HTML document with a small stand-in for the runtime,
the theme tokens it expects, and the data embedded inline, so the charts and controls work in any browser.
d3 is loaded from cdnjs.

Run from the repo root, after dashboard/build_data.py:
    python3 dashboard/build_standalone.py
"""
import json
import os

DATASETS = ["releases", "predictions", "period_scores", "gains", "scorers"]
D3_URL = "https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"
REPO_URL = "https://github.com/jarreaubarron-coder/Beigebookanalysis"

TOKENS_CSS = """
:root {
  color-scheme: light;
  --color-bg: #fcfcfb; --color-fg: #1a1a19; --color-fg-muted: #5f5e5a; --color-panel: #f4f3ef;
  --color-border-line: #dcdbd5; --color-ok: #1a7f37; --color-warn: #b45309; --color-bad: #c2321f;
  --font-anthropic-sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  --font-anthropic-serif: Georgia, "Times New Roman", serif;
  --cds-font-size-caption: 12px; --cds-font-size-body: 14px; --cds-font-size-heading: 18px; --cds-font-size-title: 24px;
  --cds-font-weight-medium: 600;
  --cds-gap-xs: 4px; --cds-gap-sm: 8px; --cds-gap-md: 12px; --cds-gap-lg: 16px; --cds-gap-xl: 28px;
  --cds-pad-xs: 4px; --cds-pad-sm: 8px; --cds-pad-md: 12px; --cds-pad-lg: 16px; --cds-pad-xl: 28px;
  --cds-radius: 8px;
  --cds-chart-muted: #8b8a84; --cds-chart-grid: #e8e7e2; --cds-chart-axis: #bdbcb5; --cds-chart-reference: #9a998f;
  --cds-text-secondary: #5f5e5a; --cds-surface-popover: #ffffff; --cds-border: #d4d3cc;
  --cds-shadow-popover: 0 4px 14px rgba(0, 0, 0, 0.12);
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a; --series-4: #eda100; --series-5: #e87ba4;
}
@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;
    --color-bg: #1a1a19; --color-fg: #f0efec; --color-fg-muted: #b5b4ab; --color-panel: #242422;
    --color-border-line: #3a3a37; --color-ok: #4cc26a; --color-warn: #e0a030; --color-bad: #f07060;
    --cds-chart-muted: #8f8e86; --cds-chart-grid: #2e2e2b; --cds-chart-axis: #55554f; --cds-chart-reference: #75746c;
    --cds-text-secondary: #c3c2b7; --cds-surface-popover: #262624; --cds-border: #44443f;
    --cds-shadow-popover: 0 4px 14px rgba(0, 0, 0, 0.5);
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70; --series-4: #c98500; --series-5: #d55181;
  }
}
[hidden] { display: none !important; }
html { background: var(--color-bg); }
body { margin: 0; background: var(--color-bg); color: var(--color-fg); font: 14px/1.5 var(--font-anthropic-sans);
  padding: 28px max(16px, env(safe-area-inset-left, 0px)) 40px max(16px, env(safe-area-inset-right, 0px)); }
.dash-select { font: inherit; font-size: 12px; color: var(--color-fg); background-color: var(--color-bg);
  border: 1px solid var(--color-border-line); border-radius: var(--cds-radius); padding: 6px 10px; }
.bb-foot a { color: inherit; }
"""

# Minimal stand-in for the dashboard runtime used by dashboard/index.html.
SHIM_JS = """
(function () {
  const DATA = window.__DATA;
  const params = {period: 'all', scorer: 'dir_act_z', model: 'ridge_g_lean_act'};
  const calcs = {}, draws = [], listeners = [], cache = new Map();
  function rows(id) {
    if (DATA[id]) return DATA[id];
    const c = calcs[id];
    if (!c) throw new Error('unknown source ' + id);
    const key = id + '|' + JSON.stringify(params);
    if (!cache.has(key)) cache.set(key, c.fn.apply(null, c.inputs.map(rows)));
    return cache.get(key);
  }
  function redraw() {
    listeners.forEach((f) => f());
    draws.forEach((d) => {
      document.querySelectorAll('[data-draw="' + d[0] + '"]').forEach((el) => {
        try { d[1](el); } catch (e) { console.error('drawing ' + d[0], e); }
      });
    });
  }
  window.dash = {
    colors: ['var(--series-1)', 'var(--series-2)', 'var(--series-3)', 'var(--series-4)', 'var(--series-5)'],
    data(id) {
      try { return {status: 'ok', data: rows(id), columns: []}; }
      catch (e) { return {status: 'error', data: [], message: String(e)}; }
    },
    calc(id, o) { calcs[id] = o; },
    draw(name, fn) { draws.push([name, fn]); },
    onData(fn) { listeners.push(fn); },
    params() { return Object.assign({}, params); },
    setParams(p) { Object.assign(params, p); redraw(); },
    start: redraw
  };
  let t = null;
  window.addEventListener('resize', () => { clearTimeout(t); t = setTimeout(redraw, 120); });
})();
"""


def main():
    template = open("dashboard/index.html").read()
    # drop the two runtime-only meta lines; keep the page's own style, markup and script
    page = "\n".join(l for l in template.splitlines() if not l.startswith('<meta name="dash-'))
    data = {}
    for name in DATASETS:
        with open(f"dashboard/data/{name}.json") as f:
            data[name] = json.load(f)["rows"]
    foot = (f'<p class="bb-foot" style="max-width:1040px;margin:28px auto 0">Source code and data: '
            f'<a href="{REPO_URL}">{REPO_URL.replace("https://", "")}</a>.</p>')
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Beige Book vs the fed funds rate</title>
<style>{TOKENS_CSS}</style>
<script src="{D3_URL}"></script>
</head>
<body>
<script>window.__DATA = {json.dumps(data, separators=(",", ":"))};</script>
<script>{SHIM_JS}</script>
<div id="dash-root">
{page}
</div>
{foot}
<script>dash.start();</script>
</body>
</html>
"""
    os.makedirs("docs", exist_ok=True)
    with open("docs/index.html", "w") as f:
        f.write(html)
    print("docs/index.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    main()
