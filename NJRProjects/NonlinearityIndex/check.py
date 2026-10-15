"""
Consistency checks for the built site.

    python3 check.py

Fails (exit code 1) if:
  - a family, tier or family-nav count differs from the number of cards actually rendered,
  - the "N of ≈400" figure, the A–Z index or the meta/OG descriptions disagree with the card total,
  - the use-case counts disagree with the number of use-case cards,
  - any #anchor link on either page points to an id that doesn't exist (same-page or cross-page),
  - a required Open Graph / Twitter tag is missing, or og:image isn't an absolute URL to a file in public/.

Uses only the standard library and reads the generated files in public/.
"""
import re
import sys
from html.parser import HTMLParser
from pathlib import Path

PUBLIC = Path(__file__).parent / "public"
PAGES = {"index.html": "/", "use-cases.html": "/use-cases"}
BASE_URL = "https://nonlinearity-index.vercel.app"
VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track", "wbr",
        "line", "path", "circle", "rect", "ellipse", "polyline", "polygon", "stop"}


class Node:
    def __init__(self, tag, attrs, parent):
        self.tag, self.attrs, self.parent = tag, dict(attrs), parent
        self.children, self.text = [], ""

    def classes(self):
        return (self.attrs.get("class") or "").split()

    def walk(self):
        yield self
        for c in self.children:
            yield from c.walk()

    def find_all(self, tag=None, cls=None):
        return [n for n in self.walk() if (tag is None or n.tag == tag) and (cls is None or cls in n.classes())]

    def find(self, tag=None, cls=None):
        found = self.find_all(tag, cls)
        return found[0] if found else None

    def all_text(self):
        return self.text + "".join(c.all_text() for c in self.children)


class TreeBuilder(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {}, None)
        self.cur = self.root

    def handle_starttag(self, tag, attrs):
        node = Node(tag, attrs, self.cur)
        self.cur.children.append(node)
        if tag not in VOID:
            self.cur = node

    def handle_startendtag(self, tag, attrs):
        self.cur.children.append(Node(tag, attrs, self.cur))

    def handle_endtag(self, tag):
        n = self.cur
        while n is not None and n.tag != tag:
            n = n.parent
        if n is not None and n.parent is not None:
            self.cur = n.parent

    def handle_data(self, data):
        self.cur.text += data


def parse(name):
    b = TreeBuilder()
    b.feed((PUBLIC / name).read_text())
    return b.root


def first_int(s):
    m = re.search(r"\d+", s or "")
    return int(m.group()) if m else None


failures = []


def check(ok, msg):
    if not ok:
        failures.append(msg)


def meta(doc, key):
    for m in doc.find_all("meta"):
        if m.attrs.get("property") == key or m.attrs.get("name") == key:
            return m.attrs.get("content")
    return None


# ------------------------------------------------------------------ catalog counts
idx = parse("index.html")
uses = parse("use-cases.html")
cards = idx.find_all("article", "fn")
total = len(cards)

print("Per-section counts (header / rendered):")
for fam in idx.find_all("section", "family"):
    header = first_int(fam.find("span", "count").all_text())
    rendered = len(fam.find_all("article", "fn"))
    name = fam.find("h3").all_text().strip()
    status = "ok" if header == rendered else "MISMATCH"
    print(f"  {name:28s} {header:>3} / {rendered:<3} {status}")
    check(header == rendered, f"family '{name}': header says {header}, {rendered} cards rendered")

for tier in idx.find_all("section", "tier"):
    head = tier.find("div", "tier-head")
    header = first_int(head.find("span", "count").all_text())
    rendered = len(tier.find_all("article", "fn"))
    name = head.find("h2").all_text().strip()
    print(f"  tier {name:23s} {header:>3} / {rendered:<3} {'ok' if header == rendered else 'MISMATCH'}")
    check(header == rendered, f"tier '{name}': header says {header}, {rendered} cards rendered")

fam_counts = {f.attrs["id"]: len(f.find_all("article", "fn")) for f in idx.find_all("section", "family")}
for a in idx.find("nav", "families").find_all("a"):
    target = a.attrs["href"].lstrip("#")
    shown = first_int(a.find("span").all_text())
    check(shown == fam_counts.get(target), f"family nav '{target}': shows {shown}, {fam_counts.get(target)} cards rendered")

progress = first_int(idx.find("div", "progress").find("strong").all_text())
check(progress == total, f"'N of ≈400' says {progress}, {total} cards rendered")
az_rows = [r for r in idx.find_all("tr") if "data-tier" in r.attrs]
check(len(az_rows) == total, f"A–Z index has {len(az_rows)} rows, {total} cards rendered")
for key in ("description", "og:description", "twitter:description"):
    check(first_int(meta(idx, key)) == total, f"index.html {key} says {first_int(meta(idx, key))}, {total} cards rendered")
shown_n = first_int(next(n for n in idx.walk() if n.attrs.get("id") == "shown").all_text())
check(shown_n == total, f"filter count says {shown_n}, {total} cards rendered")
print(f"  {'total cards':28s} {total}")

# ------------------------------------------------------------------ use-case counts
ucs = uses.find_all("article", "uc")
shown_uc = first_int(next(n for n in uses.walk() if n.attrs.get("id") == "shown").all_text())
check(shown_uc == len(ucs), f"use-cases filter count says {shown_uc}, {len(ucs)} use cases rendered")
matrix_rows = uses.find("table", "matrix").find("tbody").find_all("tr")
check(len(matrix_rows) == len(ucs), f"property matrix has {len(matrix_rows)} rows, {len(ucs)} use cases rendered")
print(f"  {'use cases':28s} {len(ucs)}")

# ------------------------------------------------------------------ anchors
ids = {name: {n.attrs["id"] for n in doc.walk() if "id" in n.attrs} for name, doc in [("index.html", idx), ("use-cases.html", uses)]}
docs = {"index.html": idx, "use-cases.html": uses}
aliases = {"": None, "index.html": "index.html", "/": "index.html", "use-cases.html": "use-cases.html", "/use-cases": "use-cases.html", "use-cases": "use-cases.html"}
bad_anchors = []
n_anchor_links = 0
for name, doc in docs.items():
    for a in doc.find_all("a"):
        href = a.attrs.get("href", "")
        if href.startswith("http") or "#" not in href:
            continue
        page_part, frag = href.split("#", 1)
        target = name if page_part == "" else aliases.get(page_part)
        n_anchor_links += 1
        if target is None:
            bad_anchors.append(f"{name}: {href} (unknown page)")
        elif frag and frag not in ids[target]:
            bad_anchors.append(f"{name}: {href} → no id '{frag}' in {target}")
for b in bad_anchors:
    failures.append("broken anchor " + b)
print(f"\nAnchor links checked: {n_anchor_links}, broken: {len(bad_anchors)}")
for b in bad_anchors:
    print("  ✗", b)

# ------------------------------------------------------------------ Open Graph / Twitter
required = ["og:title", "og:description", "og:type", "og:url", "og:image",
            "twitter:card", "twitter:title", "twitter:description", "twitter:image"]
for name, doc in docs.items():
    for key in required:
        check(meta(doc, key), f"{name}: missing <meta {key}>")
    check(meta(doc, "og:url") == BASE_URL + PAGES[name], f"{name}: og:url is {meta(doc, 'og:url')}, expected {BASE_URL + PAGES[name]}")
    check(meta(doc, "twitter:card") == "summary_large_image", f"{name}: twitter:card should be summary_large_image")
    for key in ("og:image", "twitter:image"):
        url = meta(doc, key) or ""
        check(url.startswith(BASE_URL + "/"), f"{name}: {key} should be an absolute URL on {BASE_URL}")
        check((PUBLIC / url.replace(BASE_URL + "/", "")).is_file(), f"{name}: {key} file {url} is not in public/")

# ------------------------------------------------------------------ result
if failures:
    print(f"\nFAILED: {len(failures)} problem(s)")
    for f in failures:
        print("  ✗", f)
    sys.exit(1)
print("\nAll checks passed.")
