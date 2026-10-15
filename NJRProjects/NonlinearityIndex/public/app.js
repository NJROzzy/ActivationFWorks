// The Nonlinearity Index: progressive enhancement only.
// The pages are complete without this file; it adds the theme toggle, search and filters.
(function () {
  var root = document.documentElement;

  // ---- theme toggle
  var themeBtn = document.querySelector(".theme");
  if (themeBtn) themeBtn.addEventListener("click", function () {
    var current = root.dataset.theme || (matchMedia("(prefers-color-scheme: light)").matches ? "light" : "dark");
    root.dataset.theme = current === "light" ? "dark" : "light";
    try { localStorage.setItem("nli-theme", root.dataset.theme); } catch (e) {}
  });

  var shown = document.getElementById("shown");
  var empty = document.getElementById("empty");
  var q = document.getElementById("q");

  function pressedValues(selector, attr) {
    return Array.prototype.filter.call(document.querySelectorAll(selector), function (b) {
      return b.getAttribute("aria-pressed") === "true";
    }).map(function (b) { return b.getAttribute(attr); });
  }
  function hasAll(el, props) {
    var have = (el.dataset.props || "").split(" ");
    return props.every(function (p) { return have.indexOf(p) !== -1; });
  }
  function wireSegment(selector, attr, onChange) {
    var buttons = document.querySelectorAll(selector);
    Array.prototype.forEach.call(buttons, function (b) {
      b.addEventListener("click", function () {
        Array.prototype.forEach.call(buttons, function (o) { o.setAttribute("aria-pressed", o === b ? "true" : "false"); });
        onChange();
      });
    });
  }
  function wireToggles(selector, onChange) {
    Array.prototype.forEach.call(document.querySelectorAll(selector), function (b) {
      b.addEventListener("click", function () {
        b.setAttribute("aria-pressed", b.getAttribute("aria-pressed") === "true" ? "false" : "true");
        onChange();
      });
    });
  }

  // ---- catalog page
  var cards = document.querySelectorAll(".fn");
  if (cards.length) {
    var rows = document.querySelectorAll("#index tbody tr");
    var filterCatalog = function () {
      var words = (q.value || "").toLowerCase().split(/\s+/).filter(Boolean);
      var tier = pressedValues(".seg [data-tier]", "data-tier")[0] || "all";
      var props = pressedValues(".toggles [data-prop]", "data-prop");
      var match = function (el) {
        return (tier === "all" || el.dataset.tier === tier) && hasAll(el, props) &&
          words.every(function (w) { return el.dataset.search.indexOf(w) !== -1; });
      };
      var n = 0;
      Array.prototype.forEach.call(cards, function (c) { var ok = match(c); c.hidden = !ok; if (ok) n++; });
      Array.prototype.forEach.call(rows, function (r) { r.hidden = !match(r); });
      Array.prototype.forEach.call(document.querySelectorAll(".family"), function (f) {
        f.hidden = !f.querySelector(".fn:not([hidden])");
      });
      Array.prototype.forEach.call(document.querySelectorAll(".tier"), function (t) {
        t.hidden = !t.querySelector(".fn:not([hidden])");
      });
      shown.textContent = n === cards.length ? n + " functions" : n + " of " + cards.length + " functions";
      empty.hidden = n > 0;
    };
    q.addEventListener("input", filterCatalog);
    wireSegment(".seg [data-tier]", "data-tier", filterCatalog);
    wireToggles(".toggles [data-prop]", filterCatalog);
    document.addEventListener("keydown", function (e) {
      if (e.key === "/" && !/^(INPUT|TEXTAREA|SELECT)$/.test(document.activeElement.tagName)) { e.preventDefault(); q.focus(); }
      if (e.key === "Escape" && document.activeElement === q) { q.value = ""; filterCatalog(); q.blur(); }
    });
  }

  // ---- use-case page
  var ucs = document.querySelectorAll(".uc");
  if (ucs.length) {
    var mrows = document.querySelectorAll("table.matrix tbody tr");
    var filterUses = function () {
      var area = pressedValues(".seg [data-area]", "data-area")[0] || "all";
      var props = pressedValues(".toggles [data-prop]", "data-prop");
      var match = function (el) { return (area === "all" || el.dataset.area === area) && hasAll(el, props); };
      var n = 0;
      Array.prototype.forEach.call(ucs, function (u) { var ok = match(u); u.hidden = !ok; if (ok) n++; });
      Array.prototype.forEach.call(mrows, function (r) { r.classList.toggle("dim", !match(r)); });
      shown.textContent = n === ucs.length ? n + " use cases" : n + " of " + ucs.length + " use cases";
      empty.hidden = n > 0;
    };
    wireSegment(".seg [data-area]", "data-area", filterUses);
    wireToggles(".toggles [data-prop]", filterUses);
  }
})();
