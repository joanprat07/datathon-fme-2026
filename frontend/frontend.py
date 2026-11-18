"""ShopperTwin · presentation frontend: cover, algorithm, RawTree and a live simulation.

    cd frontend && streamlit run frontend.py     # from this folder, so .streamlit/config.toml (theme) is picked up

Figures come from frontend_assets/real_stats.json, produced from our real RawTree snapshots by
`python frontend/build_stats.py`. The live tab simulates shoppers in the browser (real catalogue sample,
synthetic shoppers), so the page runs without RawTree.
Optional: drop the official logo as frontend_assets/tinybird_logo.svg (or .png) and it replaces the wordmark.
"""
from __future__ import annotations

import base64
import json
from pathlib import Path

import streamlit as st

ASSETS = Path(__file__).parent / "frontend_assets"
SEED = 2026

st.set_page_config(page_title="ShopperTwin · real-time recommendations", page_icon="🛒", layout="wide",
                   initial_sidebar_state="collapsed")


def asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


@st.cache_data
def stats() -> dict:
    return json.loads(asset("real_stats.json"))


def partner_logo() -> str:
    for name, mime in (("tinybird_logo.svg", "image/svg+xml"), ("tinybird_logo.png", "image/png")):
        f = ASSETS / name
        if f.exists():
            return f'<img alt="Tinybird" src="data:{mime};base64,{base64.b64encode(f.read_bytes()).decode()}">'
    return '<span class="wordmark">tinybird</span>'


def page(html_file: str, **extra: str) -> None:
    s = stats()
    html = (asset(html_file)
            .replace("/*__BASE_CSS__*/", asset("widget.css"))
            .replace("/*__PAGES_CSS__*/", asset("pages.css"))
            .replace("/*__CHARTS__*/", asset("charts.js"))
            .replace("__STATS__", json.dumps(s)))
    for k, v in extra.items():
        html = html.replace(k, v)
    st.iframe(html, height="content")


def live() -> None:
    s = stats()
    params = {
        "seed": SEED, "users_per_sec": s["stream"]["requests_per_s"], "new_share": s["stream"]["new_user_share"],
        "exam": s["exam"], "quotas": s["config"]["quotas"],
        "explore_eps": s["config"]["explore_eps"], "bought_prior": s["config"]["bought_prior"],
        "pages": s["sessions"]["page"], "rv_share": s["sessions"]["rv_share"], "target_cvr": s["stream"]["baseline_live_cvr"],
        "regions": s["regions"], "age_p50": s["dry"][-1]["age_p50"], "age_p95": s["dry"][-1]["age_p95"],
    }
    sim = asset("sim.js").replace("__CATALOG__", asset("catalog_sample.json")).replace("__PARAMS__", json.dumps(params))
    html = asset("live.html").replace("/*__BASE_CSS__*/", asset("widget.css")).replace("/*__SIM__*/", sim)
    st.iframe(html, height="content")


st.html(f"<style>{asset('style.css')}</style>")
st.html("""
<nav class="tb-nav">
  <div class="tb-logo">Shopper<span class="tb-dim">Twin</span></div>
  <div class="tb-links"><span>Algorithm</span><span>Model <i>[FTRL]</i></span><span>RawTree<sup>[LIVE]</sup></span></div>
  <div class="tb-actions"><span class="tb-link-green">Datathon FME</span><span class="tb-btn">team_28</span></div>
</nav>
""")

t0, t1, t2, t3 = st.tabs(["[00] ShopperTwin", "[01] The algorithm", "[02] RawTree", "[03] Live: ShopperTwin vs baseline"])
with t0:
    page("cover.html", __TB_LOGO__=partner_logo())
with t1:
    page("algo.html")
with t2:
    page("rawtree.html")
with t3:
    live()

st.html("""
<footer class="tb-footer">
  <span><i class="dot"></i>All systems operational</span>
  <span>Charts: real RawTree snapshots · Live tab: simulated shoppers on a real catalogue sample</span>
  <span>Datathon FME 2026 &nbsp;|&nbsp; RawTree challenge</span>
</footer>
""")
