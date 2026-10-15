"""
Build The Nonlinearity Index.

    python3 build.py

Reads catalog.py and use_cases.py, writes public/index.html and public/use-cases.html,
then runs check.py and fails if the pages are inconsistent.
Every curve is pre-rendered as inline SVG, so the pages are plain HTML and CSS; public/app.js
only adds search, filters and the theme toggle. No third-party Python packages are needed.
"""
import html
import math
import re
from pathlib import Path

from catalog import FUNCS, FAMILIES, TIERS, NOT_PLOTTED
from use_cases import USE_CASES, PROPS, AREAS

OUT = Path(__file__).parent / "public"
GOAL = 400
SITE = "The Nonlinearity Index"
REPO = "https://github.com/NJROzzy/ActivationFWorks"
BASE_URL = "https://nonlinearity-index.vercel.app"
OG_IMAGE = f"{BASE_URL}/og-image.png"
GH_ICON = '<svg viewBox="0 0 16 16" aria-hidden="true" fill="currentColor"><path d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0016 8c0-4.42-3.58-8-8-8z"/></svg>'
esc = lambda s: html.escape(str(s), quote=True)


# ------------------------------------------------------------------ validation
def validate():
    slugs = [f["slug"] for f in FUNCS]
    dupes = {s for s in slugs if slugs.count(s) > 1}
    assert not dupes, f"duplicate slugs: {dupes}"
    fams = {k for k, *_ in FAMILIES}
    for f in FUNCS:
        assert f["family"] in fams, f"{f['slug']}: unknown family {f['family']}"
    props = {k for k, *_ in PROPS}
    for u in USE_CASES:
        assert u["area"] in AREAS, f"{u['slug']}: unknown area {u['area']}"
        for s, _ in u["picks"]:
            assert s in slugs, f"{u['slug']}: unknown function {s}"
        for p in u["props"]:
            assert p in props, f"{u['slug']}: unknown property {p}"


# ------------------------------------------------------------------ numerics
def ev(f, x):
    try:
        y = f(x)
    except (ValueError, OverflowError, ZeroDivisionError):
        return None
    if y is None or isinstance(y, complex) or not math.isfinite(y):
        return None
    return float(y)


def curves_of(fdef):
    return fdef["f"] if isinstance(fdef["f"], list) else [(None, fdef["f"])]


def is_monotonic(f):
    ys = [ev(f, -8 + i * 0.01) for i in range(1601)]
    ys = [y for y in ys if y is not None]
    d = [b - a for a, b in zip(ys, ys[1:])]
    return all(v >= -1e-9 for v in d) or all(v <= 1e-9 for v in d)


FAR = (1e2, 3e2, 1e3, 1e4, 1e6)


def far_samples(f, side, centre):
    """Finite values on a few points near side·centre (several points so oscillating functions are caught)."""
    vals = [ev(f, side * (centre + k * 0.731)) for k in range(25)]
    return [v for v in vals if v is not None]


def grows(f, side):
    """True if |f| keeps growing beyond x = 100 on this side. Overflowing points are skipped, not counted as growth."""
    peaks = [max(map(abs, v)) for v in (far_samples(f, side, c) for c in FAR) if v]
    if peaks and max(peaks) > 1e12:
        return True                  # exponential growth overflows soon after; already huge
    return len(peaks) >= 2 and max(peaks[1:]) > 1.05 * peaks[0] + 1e-6


def boundedness(f):
    return not grows(f, 1) and not grows(f, -1)


def is_smooth(f, lo=-8.0, hi=8.0):
    """Differentiable everywhere on [lo, hi]: no jump in the numerical derivative."""
    h, step = 1e-6, 1e-3
    prev = None
    x = lo
    while x <= hi:
        a, b = ev(f, x - h), ev(f, x + h)
        if a is None or b is None:
            prev = None
        else:
            d = (b - a) / (2 * h)
            if prev is not None and abs(d - prev) > 0.02 + 0.002 * abs(prev):
                return False
            prev = d
        x += step
    return True


def fmt_num(v):
    if abs(v) >= 100: return f"{v:.0f}"
    s = f"{v:.2g}" if abs(v) < 1 else f"{v:.3g}"
    return s.replace("-", "−")


def auto_range(f, bounded=None):
    """Output range estimated by sampling (shown with ≈). Each side is checked for growth far out."""
    def vals(a, b, n=400):
        return [v for v in (ev(f, a + (b - a) * k / n) for k in range(n + 1)) if v is not None]
    def peak(a, b):
        v = vals(a, b)
        return (max(v), min(v)) if v else (None, None)
    near = vals(-60, 60, 12000) + vals(-1000, -60) + vals(60, 1000)
    if not near:
        return "–"
    lo, hi = min(near), max(near)
    for side in (1, -1):
        if grows(f, side):
            far = [v for c in FAR for v in far_samples(f, side, c)]
            if max(far) > 0:
                hi = float("inf")
            if min(far) < 0:
                lo = float("-inf")
    clean = lambda v: 0.0 if abs(v) < 5e-4 else v
    lo_s = "−∞" if lo == float("-inf") else fmt_num(clean(lo))
    hi_s = "∞" if hi == float("inf") else fmt_num(clean(hi))
    left = "(" if lo_s == "−∞" else "["
    right = ")" if hi_s == "∞" else "]"
    if lo_s == "−∞" and hi_s == "∞":
        return "(−∞, ∞)"
    return f"≈ {left}{lo_s}, {hi_s}{right}"


# ------------------------------------------------------------------ SVG plots
W, H, PAD = 200, 120, 6


def path_for(pts, sx, sy, span):
    """Polyline with breaks at gaps (None) and at jump discontinuities."""
    out, prev, prev_d, last = [], None, 0.0, (0.0, 0.0)
    for x, y in pts:
        if y is None:
            prev = None
            continue
        if prev is not None:
            d = abs(y - prev[1])
            jump = d > 0.15 * span and d > 4 * prev_d + 1e-6
            prev_d = d
            if jump:
                prev = None
        px, py = round(sx(x), 1), round(sy(y), 1)
        if prev is None:
            prev_d = 0.0
            out.append(f"M{num(px)} {num(py)}")
        else:
            dx, dy = round(px - last[0], 1), round(py - last[1], 1)
            out.append(f"l{num(dx)}{'' if dy < 0 else ' '}{num(dy)}")
            px, py = round(last[0] + dx, 1), round(last[1] + dy, 1)
        last = (px, py)
        prev = (x, y)
    return "".join(out)


def num(v):
    """Shortest decimal form: 12.0 → 12, 0.5 → .5, -0.5 → -.5."""
    t = f"{v:.1f}".rstrip("0").rstrip(".")
    if t in ("-0", ""):
        t = "0"
    return t.replace("0.", ".", 1) if t.startswith("0.") else (t.replace("-0.", "-.", 1) if t.startswith("-0.") else t)


def plot(fdef, small=False):
    x0, x1 = fdef.get("xr", (-4, 4))
    n = 48 if small else 121
    xs = [x0 + (x1 - x0) * i / (n - 1) for i in range(n)]
    curves = curves_of(fdef)
    data = [(label, [(x, ev(f, x)) for x in xs]) for label, f in curves]
    ys = [y for _, pts in data for _, y in pts if y is not None]
    lo, hi = max(min(ys), -6), min(max(ys), 6)
    lo, hi = min(lo, 0), max(hi, 0)
    if hi - lo < 1:
        mid = (hi + lo) / 2
        lo, hi = mid - 0.5, mid + 0.5
    pad = (hi - lo) * 0.1
    lo, hi = lo - pad, hi + pad
    span = hi - lo
    w, h, p = (64, 26, 2) if small else (W, H, PAD)
    sx = lambda x: p + (x - x0) / (x1 - x0) * (w - 2 * p)
    sy = lambda y: h - p - (max(lo - span, min(hi + span, y)) - lo) / span * (h - 2 * p)

    if small:
        d = path_for(data[min(1, len(data) - 1)][1] if len(data) == 3 else data[0][1], sx, sy, span)
        return (f'<svg class="spark" viewBox="0 0 {w} {h}" aria-hidden="true">'
                f'<line class="ax" x1="{p}" x2="{w - p}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>'
                f'<path class="c" d="{d}"/></svg>')

    parts = [f'<svg class="plot" viewBox="0 0 {W} {H}" role="img" aria-label="Graph of {esc(fdef["name"])}">']
    for gx in range(math.ceil(x0), math.floor(x1) + 1):
        if gx and (x1 - x0) <= 12:
            parts.append(f'<line class="g" x1="{sx(gx):.1f}" x2="{sx(gx):.1f}" y1="{p}" y2="{h - p}"/>')
    step = 1 if span <= 8 else 2
    for gy in range(math.ceil(lo), math.floor(hi) + 1):
        if gy and gy % step == 0:
            parts.append(f'<line class="g" x1="{p}" x2="{w - p}" y1="{sy(gy):.1f}" y2="{sy(gy):.1f}"/>')
    parts.append(f'<line class="ax" x1="{p}" x2="{w - p}" y1="{sy(0):.1f}" y2="{sy(0):.1f}"/>')
    if x0 < 0 < x1:
        parts.append(f'<line class="ax" x1="{sx(0):.1f}" x2="{sx(0):.1f}" y1="{p}" y2="{h - p}"/>')

    if len(data) == 1:
        f = curves[0][1]
        hstep = 1e-4
        dpts = []
        for x in xs:
            a, b = ev(f, x - hstep), ev(f, x + hstep)
            dpts.append((x, None if a is None or b is None else (b - a) / (2 * hstep)))
        parts.append(f'<path class="d" d="{path_for(dpts, sx, sy, span)}"/>')
        parts.append(f'<path class="c" d="{path_for(data[0][1], sx, sy, span)}"/>')
    else:
        for i, (label, pts) in enumerate(data):
            parts.append(f'<path class="c c{i}" d="{path_for(pts, sx, sy, span)}"/>')
    parts.append("</svg>")

    legend = ""
    if len(data) > 1:
        legend = '<p class="legend">' + "".join(f'<span class="k{i}">{esc(lb)}</span>' for i, (lb, _) in enumerate(data)) + "</p>"
    return "".join(parts) + legend


# ------------------------------------------------------------------ formulas
VAR = re.compile(r"(?<![A-Za-z&#;])([xabzptfkjiy]|α|β|λ|θ)(?![A-Za-z;])")


def mathify(s):
    """Italicise single-letter variables in text, leaving tags and entities alone."""
    return "".join(part if part.startswith("<") else VAR.sub(r"<i>\1</i>", part)
                   for part in re.split(r"(<[^>]+>)", s))


def formula_html(fm):
    if isinstance(fm, str):
        return f'<div class="formula">{mathify(fm)}</div>'
    rows = "".join(f"<span>{mathify(e)}</span><span class=\"if\">if {mathify(c)}</span>"
                   if not c.startswith("otherwise") else f"<span>{mathify(e)}</span><span class=\"if\">otherwise</span>"
                   for e, c in fm)
    return f'<div class="formula pw"><span class="lhs"><i>f</i>(<i>x</i>) =</span><span class="cases">{rows}</span></div>'


# ------------------------------------------------------------------ shared chrome
FONT = ("https://fonts.googleapis.com/css2?family=Geist:wght@400;500;600;700"
        "&family=Geist+Mono:wght@400;500&family=STIX+Two+Text:ital@0;1&display=swap")
LOGO = ('<svg viewBox="0 0 32 32" aria-hidden="true"><path d="M3 24 C 11 24, 13 8, 29 8" fill="none" '
        'stroke="currentColor" stroke-width="2.4" stroke-linecap="round"/><circle cx="16" cy="16" r="3" fill="var(--acc)"/></svg>')


def page(title, desc, active, body, path):
    nav = "".join(f'<a href="{href}"{" aria-current=\"page\"" if key == active else ""}>{label}</a>'
                  for key, href, label in [("catalog", "index.html", "Catalog"), ("uses", "use-cases.html", "Use cases"),
                                           ("about", "index.html#about", "About")])
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{esc(title)}</title>
<meta name="description" content="{esc(desc)}">
<meta property="og:title" content="{esc(title)}">
<meta property="og:description" content="{esc(desc)}">
<meta property="og:type" content="website">
<meta property="og:url" content="{BASE_URL}{path}">
<meta property="og:site_name" content="{SITE}">
<meta property="og:image" content="{OG_IMAGE}">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{SITE}: a catalogue of activation functions">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:title" content="{esc(title)}">
<meta name="twitter:description" content="{esc(desc)}">
<meta name="twitter:image" content="{OG_IMAGE}">
<link rel="icon" href="favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="{FONT}">
<link rel="stylesheet" href="styles.css">
<script>try{{var t=localStorage.getItem("nli-theme");if(t)document.documentElement.dataset.theme=t}}catch(e){{}}document.documentElement.classList.add("js")</script>
</head>
<body>
<header class="site">
  <a class="brand" href="index.html">{LOGO}<span>{SITE}</span></a>
  <nav>{nav}<a class="gh" href="{REPO}" target="_blank" rel="noopener" title="Star the repo on GitHub">{GH_ICON}<span>Star on GitHub</span></a><button class="theme" type="button" aria-label="Toggle light or dark theme">◐</button></nav>
</header>
{body}
<footer class="site">
  <p><strong>{SITE}</strong>: a growing catalogue of activation functions for deep learning.</p>
  <p>{len(FUNCS)} of a planned ≈{GOAL} functions catalogued. Curves are computed from the formulas shown; dashed lines are derivatives.</p>
  <p class="star">Find this useful? <a href="{REPO}" target="_blank" rel="noopener">★ Star it on GitHub</a> so you can find it again and see new batches as they land.</p>
</footer>
<script src="app.js" defer></script>
</body>
</html>
"""


# ------------------------------------------------------------------ catalog page
def caption(f):
    x0, x1 = f.get("xr", (-4, 4))
    bits = [f"x ∈ [{x0:g}, {x1:g}]".replace("-", "−")]
    if not isinstance(f["f"], list):
        bits.append("dashed: f′(x)")
    if f["family"] == "glu":
        bits.append("a = b = x")
    if f.get("note"):
        bits.append(esc(f["note"]))
    return " · ".join(bits)


def build_catalog():
    seen = {}
    for u in USE_CASES:
        for s, _ in u["picks"]:
            seen.setdefault(s, []).append(u)
    fam_of = {k: (tier, name, d) for k, tier, name, d in FAMILIES}
    tier_name = {k: n for k, n, _ in TIERS}

    cards = {}
    rows = []
    for f in FUNCS:
        main_f = curves_of(f)[1 if len(curves_of(f)) == 3 else 0][1]
        mono = f["mono"] if "mono" in f else is_monotonic(main_f)
        all_f = [c for _, c in curves_of(f)]
        bounded = f["bounded"] if "bounded" in f else all(boundedness(c) for c in all_f)
        if f.get("smooth") is None:
            f["smooth"] = all(is_smooth(c) for c in all_f)
        if f.get("range") is None:
            f["range"] = auto_range(main_f, bounded)
        f["_mono"], f["_bounded"] = mono, bounded
        tier = fam_of[f["family"]][0]
        props = ["smooth" if f["smooth"] else "nonsmooth", "mono" if mono else "nonmono"]
        if bounded: props.append("bounded")
        if f.get("torch"): props.append("torch")
        tags = [f'<li title="output range">{f["range"]}</li>',
                f'<li>{"smooth" if f["smooth"] else "non-smooth"}</li>',
                f'<li>{"monotonic" if mono else "non-monotonic"}</li>']
        if bounded: tags.append("<li>bounded</li>")
        if f.get("torch"): tags.append(f'<li class="torch" title="Available in PyTorch">torch.{esc(f["torch"])}</li>')
        search = " ".join([f["name"], f.get("aka", ""), f["slug"], f["desc"], f.get("src", ""), f.get("torch", ""),
                           fam_of[f["family"]][1], tier_name[tier]]).lower()
        seen_html = ""
        if f["slug"] in seen:
            links = ", ".join(f'<a href="use-cases.html#{u["slug"]}">{esc(u["title"])}</a>' for u in seen[f["slug"]])
            seen_html = f'<p class="seen"><span>Chosen for</span> {links}</p>'
        cards.setdefault(f["family"], []).append(f"""
<article class="fn" id="{f['slug']}" data-tier="{tier}" data-props="{' '.join(props)}" data-search="{esc(search)}">
  <header><h4><a href="#{f['slug']}">{esc(f['name'])}</a></h4>{f'<p class="aka">{esc(f["aka"])}</p>' if f.get('aka') else ''}</header>
  <figure>{plot(f)}<figcaption>{caption(f)}</figcaption></figure>
  {formula_html(f['formula'])}
  <p class="desc">{esc(f['desc'])}</p>
  <ul class="tags">{''.join(tags)}</ul>
  {f'<p class="src">{esc(f["src"])}</p>' if f.get('src') else ''}
  {seen_html}
</article>""")
        rows.append((f["name"].lower(), f"""<tr data-tier="{tier}" data-props="{' '.join(props)}" data-search="{esc(search)}">
<td><a href="#{f['slug']}">{esc(f['name'])}</a></td><td>{plot(f, small=True)}</td><td>{esc(fam_of[f['family']][1])}</td>
<td>{f['range']}</td><td>{'✓' if f['smooth'] else '–'}</td><td>{'✓' if mono else '–'}</td><td>{esc(f.get('torch', '')) or '–'}</td></tr>"""))

    tiers_html = []
    for ti, (tk, tname, tdesc) in enumerate(TIERS, 1):
        fams = [x for x in FAMILIES if x[1] == tk]
        n_t = sum(len(cards.get(k, [])) for k, *_ in fams)
        fam_html = "".join(f"""
<section class="family" id="fam-{k}" data-family="{k}">
  <div class="fam-head"><h3>{esc(name)}</h3><span class="count">{len(cards.get(k, []))}</span></div>
  <p class="lead">{esc(d)}</p>
  <div class="grid">{''.join(cards.get(k, []))}</div>
</section>""" for k, _, name, d in fams)
        tiers_html.append(f"""
<section class="tier" id="tier-{tk}" data-tier="{tk}">
  <div class="tier-head"><span class="num">{'I' * ti if ti < 4 else ti}</span><div><h2>{esc(tname)}</h2><p>{esc(tdesc)}</p></div><span class="count">{n_t} functions</span></div>
  {fam_html}
</section>""")

    fam_nav = "".join(f'<a href="#fam-{k}">{esc(n)} <span>{len(cards.get(k, []))}</span></a>' for k, _, n, _ in FAMILIES)
    pct = len(FUNCS) / GOAL * 100
    body = f"""
<main>
<section class="hero">
  <p class="eyebrow">A catalogue of activation functions</p>
  <h1 class="eq"><i>y</i> = <span class="phi">φ</span>(<i>W</i><i>x</i> + <i>b</i>)</h1>
  <div class="hero-copy">
    <p class="big">Everything a neural network can learn beyond a straight line comes from <span class="phi">φ</span>, the activation function.</p>
    <p>This index collects them: the formula, how the curve behaves, and a plot of it and its derivative. It's organised by family, from the classics every framework ships to learned and experimental functions from the research literature. A <a href="use-cases.html">use-case guide</a> explains which ones get picked for which jobs, and why.</p>
  </div>
  <div class="progress" aria-label="{len(FUNCS)} of about {GOAL} functions catalogued">
    <div class="bar"><span style="width:{pct:.1f}%"></span></div>
    <p><strong>{len(FUNCS)}</strong> of ≈{GOAL} functions catalogued · {len(FAMILIES)} families · {len(TIERS)} tiers</p>
  </div>
</section>

<section class="controls" id="catalog" aria-label="Search and filter">
  <div class="row">
    <input id="q" type="search" placeholder="Search by name, alias, paper or property…  ( / )" aria-label="Search functions">
    <div class="seg" role="group" aria-label="Tier">
      <button data-tier="all" aria-pressed="true">All</button>{''.join(f'<button data-tier="{k}" aria-pressed="false">{esc(n)}</button>' for k, n, _ in TIERS)}
    </div>
  </div>
  <div class="row">
    <div class="toggles" role="group" aria-label="Properties">
      <button data-prop="smooth" aria-pressed="false">Smooth</button>
      <button data-prop="nonsmooth" aria-pressed="false">Non-smooth</button>
      <button data-prop="mono" aria-pressed="false">Monotonic</button>
      <button data-prop="nonmono" aria-pressed="false">Non-monotonic</button>
      <button data-prop="bounded" aria-pressed="false">Bounded</button>
      <button data-prop="torch" aria-pressed="false">In PyTorch</button>
    </div>
    <p class="shown" id="shown">{len(FUNCS)} functions</p>
  </div>
</section>

<nav class="families" aria-label="Families">{fam_nav}</nav>

<div id="results">{''.join(tiers_html)}</div>
<p class="empty" id="empty" hidden>No function matches those filters.</p>

<section class="block" id="index">
  <h2>A–Z index</h2>
  <p class="lead">Every catalogued function on one page. Click a name to jump to its card.</p>
  <div class="tbl"><table>
    <thead><tr><th>Function</th><th>Curve</th><th>Family</th><th>Range</th><th>Smooth</th><th>Monotonic</th><th>PyTorch</th></tr></thead>
    <tbody>{''.join(r for _, r in sorted(rows))}</tbody>
  </table></div>
</section>

<section class="block" id="not-drawn">
  <h2>In the survey, not drawn here</h2>
  <p class="lead">Most functions come from the survey by Kunc &amp; Kléma (2024), which collects about 400 activation functions. These {len(NOT_PLOTTED)} entries from it have no card because they can't be drawn as one fixed curve of one input:</p>
  <div class="tbl"><table>
    <thead><tr><th>Function</th><th>Why it isn't plotted</th></tr></thead>
    <tbody>{''.join(f'<tr><td>{esc(n)}</td><td>{esc(r)}</td></tr>' for n, r in NOT_PLOTTED)}</tbody>
  </table></div>
</section>

<section class="block about" id="reading">
  <h2>How to read a card</h2>
  <div class="cols">
    <div><h3>The plot</h3><p>The solid line is the function and the dashed line is its derivative, computed numerically from the same formula. The x-range is printed in the corner; the y-axis is scaled to fit and clipped at ±6. Where a function has parameters, the card plots the stated defaults or several values (see the legend). Vector functions such as softmax are plotted for two classes, and gated units with both inputs tied to x.</p></div>
    <div><h3>The tags</h3><p><b>Smooth</b> means differentiable everywhere (no kinks). <b>Monotonic</b> means never decreasing (or never increasing). <b>Bounded</b> means the output stays in a finite interval in both directions. A <b>torch.nn</b> tag means PyTorch ships it as a module. <b>Chosen for</b> links to the use cases where it's a typical pick.</p></div>
    <div><h3>Sources</h3><p>Each card names the paper usually credited with the function. Cards marked "via Kunc &amp; Kléma, 2024" were collected from the survey <a href="https://arxiv.org/abs/2402.09092" target="_blank" rel="noopener">Three Decades of Activations</a>: every one of their implementations was checked numerically against the survey's own formula. Where the survey's formula has an evident typo, the card says how it was corrected. Descriptions are our own.</p></div>
  </div>
</section>

<section class="block" id="about">
  <h2>About</h2>
  <p class="lead">{SITE} is a growing catalogue of activation functions for deep learning: the formula, how each curve behaves, and a plot of it and its derivative, organised by family and paired with a guide to which functions get chosen for which tasks. It is built as part of the open <a href="{REPO}" target="_blank" rel="noopener">ActivationFWorks</a> project and grows in batches toward ≈{GOAL} functions.</p>
  <p class="lead">Corrections to formulas, citations or use-case claims are welcome: open an issue or pull request on <a href="{REPO}" target="_blank" rel="noopener">GitHub</a>.</p>
</section>
</main>"""
    return page(f"{SITE}: a catalogue of activation functions",
                f"{len(FUNCS)} activation functions with formulas, behaviour notes and plots, organised by family.",
                "catalog", body, "/")


# ------------------------------------------------------------------ use-case page
def build_uses():
    F = {f["slug"]: f for f in FUNCS}
    P = {k: (label, d) for k, label, d in PROPS}

    def chip(slug, role=""):
        f = F[slug]
        r = f'<span class="role">{esc(role)}</span>' if role else ""
        return f'<li><a class="fchip" href="index.html#{slug}">{plot(f, small=True)}<span class="nm">{esc(f["name"])}</span></a>{r}</li>'

    cards = []
    for u in USE_CASES:
        props = "".join(f'<li title="{esc(P[p][1])}">{esc(P[p][0])}</li>' for p in u["props"])
        cards.append(f"""
<article class="uc" id="{u['slug']}" data-area="{esc(u['area'])}" data-props="{' '.join(u['props'])}">
  <p class="area">{esc(u['area'])}</p>
  <h3><a href="#{u['slug']}">{esc(u['title'])}</a></h3>
  <p class="arch">{esc(u['arch'])}</p>
  <h4>Typical picks</h4>
  <ul class="picks">{''.join(chip(s, r) for s, r in u['picks'])}</ul>
  <h4>Properties that matter</h4>
  <ul class="tags">{props}</ul>
  <p class="why">{esc(u['why'])}</p>
  <p class="caveat"><strong>Watch out:</strong> {esc(u['caveat'])}</p>
  {f'<p class="ex">{esc(u["examples"])}</p>' if u.get('examples') else ''}
</article>""")

    head = "".join(f'<th title="{esc(d)}"><span>{esc(lbl)}</span></th>' for k, lbl, d in PROPS)
    mrows = "".join(
        f'<tr data-area="{esc(u["area"])}" data-props="{" ".join(u["props"])}"><td><a href="#{u["slug"]}">{esc(u["title"])}</a></td>'
        + "".join(f'<td>{"●" if k in u["props"] else ""}</td>' for k, *_ in PROPS) + "</tr>"
        for u in USE_CASES)

    by_fn = {}
    for u in USE_CASES:
        for s, _ in u["picks"]:
            by_fn.setdefault(s, []).append(u)
    frows = "".join(
        f'<tr><td><a href="index.html#{s}">{esc(F[s]["name"])}</a></td><td>{plot(F[s], small=True)}</td><td>{len(us)}</td>'
        f'<td>{", ".join(f"<a href=\"#{u["slug"]}\">{esc(u["title"])}</a>" for u in us)}</td></tr>'
        for s, us in sorted(by_fn.items(), key=lambda kv: (-len(kv[1]), F[kv[0]]["name"].lower())))

    areas_present = [a for a in AREAS if any(u["area"] == a for u in USE_CASES)]
    body = f"""
<main>
<section class="hero">
  <p class="eyebrow">Use cases</p>
  <h1>Which activation, for which job?</h1>
  <div class="hero-copy">
    <p class="big">Tendencies, not territories.</p>
    <p>No activation function belongs to a domain. What carries over between tasks are <em>properties</em>: smooth gradients, sparsity, bounded outputs, gating, usable higher derivatives. This guide shows which functions practitioners tend to reach for in {len(USE_CASES)} kinds of tasks and architectures, and explains which property makes each one a good fit.</p>
  </div>
</section>

<section class="controls" aria-label="Filter use cases">
  <div class="row">
    <div class="seg wrap" role="group" aria-label="Area">
      <button data-area="all" aria-pressed="true">All areas</button>{''.join(f'<button data-area="{esc(a)}" aria-pressed="false">{esc(a)}</button>' for a in areas_present)}
    </div>
  </div>
  <div class="row">
    <div class="toggles" role="group" aria-label="Properties">{''.join(f'<button data-prop="{k}" aria-pressed="false" title="{esc(d)}">{esc(lbl)}</button>' for k, lbl, d in PROPS)}</div>
    <p class="shown" id="shown">{len(USE_CASES)} use cases</p>
  </div>
</section>

<div class="ucgrid" id="results">{''.join(cards)}</div>
<p class="empty" id="empty" hidden>No use case matches those filters.</p>

<section class="block" id="matrix">
  <h2>Property matrix</h2>
  <p class="lead">Which properties each use case relies on. Read down a column to see where a property matters; hover a heading for its definition.</p>
  <div class="tbl"><table class="matrix"><thead><tr><th>Use case</th>{head}</tr></thead><tbody>{mrows}</tbody></table></div>
</section>

<section class="block" id="by-function">
  <h2>By function</h2>
  <p class="lead">The same information turned around: every function that appears above, with the use cases it's picked for. Most show up in several unrelated areas, which is the point.</p>
  <div class="tbl"><table><thead><tr><th>Function</th><th>Curve</th><th>Use cases</th><th>Where</th></tr></thead><tbody>{frows}</tbody></table></div>
</section>
</main>"""
    return page(f"Use cases · {SITE}",
                "Which activation functions are chosen for which tasks and architectures, and the properties that explain why.",
                "uses", body, "/use-cases")


if __name__ == "__main__":
    validate()
    OUT.mkdir(exist_ok=True)
    (OUT / "index.html").write_text(build_catalog())
    (OUT / "use-cases.html").write_text(build_uses())
    print(f"built {len(FUNCS)} functions, {len(USE_CASES)} use cases → {OUT}")
    # Build-time check: exits non-zero if a section count differs from its rendered cards
    # or any #anchor on either page has no matching id (see check.py).
    import runpy
    runpy.run_path(str(Path(__file__).parent / "check.py"), run_name="__main__")
