"""Export the whole presentation to one self-contained HTML file (no Streamlit needed to open it).

    python frontend/export_html.py          # writes frontend/pandabuy.html

It runs frontend.py with a tiny stand-in for streamlit that collects the page parts,
so the HTML is always the same as the Streamlit app.
"""
from __future__ import annotations

import html
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE / "pandabuy.html"


def collect() -> tuple[list[str], list[str], list[str]]:
    parts: list[str] = []      # st.html blocks: <style>, nav, footer
    pages: list[str] = []      # st.iframe documents, one per tab
    names: list[str] = []

    class _Tab:
        def __enter__(self) -> "_Tab":
            return self

        def __exit__(self, *a: object) -> None:
            pass

    st = types.ModuleType("streamlit")
    st.set_page_config = lambda **k: None
    st.cache_data = lambda f: f
    st.html = lambda s, **k: parts.append(str(s))
    st.iframe = lambda s, **k: pages.append(str(s))

    def tabs(n: list[str]) -> list[_Tab]:
        names.extend(n)
        return [_Tab() for _ in n]

    st.tabs = tabs
    sys.modules["streamlit"] = st
    src = (HERE / "frontend.py").read_text(encoding="utf-8")
    exec(compile(src, str(HERE / "frontend.py"), "exec"), {"__file__": str(HERE / "frontend.py"), "__name__": "export"})
    return parts, pages, names


def main() -> None:
    parts, pages, names = collect()
    style, nav, footer = parts[0], parts[1], parts[-1]
    buttons = "".join(
        f'<button role="tab" data-i="{i}" aria-selected="{str(i == 0).lower()}"><p>{html.escape(n)}</p></button>'
        for i, n in enumerate(names))
    frames = "".join(
        f'<iframe data-i="{i}" srcdoc="{html.escape(p, quote=True)}" style="{"" if i == 0 else "display:none"}"></iframe>'
        for i, p in enumerate(pages))
    doc = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>pandabuy · real-time recommendations</title>
{style}
<style>
body {{ margin: 0; }}
.wrap {{ max-width: 1240px; margin: 0 auto; padding: 0 32px 48px; }}
[data-testid="stTabs"] [role="tablist"] {{ display: flex; }}
[data-testid="stTabs"] [role="tab"] {{ cursor: pointer; }}
[data-testid="stTabs"] [role="tab"] p {{ margin: 0; }}
iframe {{ width: 100%; border: 0; display: block; margin-top: 16px; }}
</style></head>
<body class="stApp"><div class="wrap">
{nav}
<div data-testid="stTabs"><div role="tablist">{buttons}</div></div>
{frames}
{footer}
</div>
<script>
const frames = [...document.querySelectorAll("iframe")];
function fit(f) {{ try {{ f.style.height = f.contentDocument.documentElement.scrollHeight + "px"; }} catch (e) {{ f.style.height = "1400px"; }} }}
frames.forEach(f => {{ f.addEventListener("load", () => fit(f)); setInterval(() => fit(f), 1000); }});
document.querySelectorAll('[role="tab"]').forEach(b => b.onclick = () => {{
  document.querySelectorAll('[role="tab"]').forEach(x => x.setAttribute("aria-selected", x === b));
  frames.forEach(f => {{ f.style.display = f.dataset.i === b.dataset.i ? "" : "none"; if (f.dataset.i === b.dataset.i) fit(f); }});
}});
</script>
</body></html>
"""
    OUT.write_text(doc, encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
