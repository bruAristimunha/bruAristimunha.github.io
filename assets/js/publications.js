(function () {
  "use strict";

  function init(root) {
    var filters = root.querySelectorAll("[data-filter]");
    var entries = root.querySelectorAll("[data-type]");
    var groups = root.querySelectorAll("[data-year-group]");
    var cells = root.querySelectorAll("[data-cell-type]");
    if (!filters.length || !entries.length) return;

    function apply(type) {
      entries.forEach(function (el) {
        var show = type === "all" || el.getAttribute("data-type") === type;
        el.hidden = !show;
      });
      // Hide a year group when none of its entries are visible under the filter.
      groups.forEach(function (g) {
        var visible = g.querySelectorAll("[data-type]:not([hidden])").length;
        g.hidden = visible === 0;
      });
      // Dim non-matching squares in the unit chart so it doubles as feedback.
      cells.forEach(function (c) {
        var match = type === "all" || c.getAttribute("data-cell-type") === type;
        c.classList.toggle("is-dim", !match);
      });
      filters.forEach(function (b) {
        var on = b.getAttribute("data-filter") === type;
        b.classList.toggle("is-active", on);
        b.setAttribute("aria-pressed", on ? "true" : "false");
      });
    }

    filters.forEach(function (b) {
      b.addEventListener("click", function () {
        apply(b.getAttribute("data-filter"));
      });
    });
  }

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    document.querySelectorAll("[data-pubtrack]").forEach(function (root) {
      init(root);
    });
  });
})();
