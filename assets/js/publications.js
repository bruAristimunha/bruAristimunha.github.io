(function () {
  "use strict";

  function init(root) {
    var filters = root.querySelectorAll("[data-filter]");
    var entries = root.querySelectorAll("[data-type]");
    var groups = root.querySelectorAll("[data-year-group]");
    var cells = root.querySelectorAll("[data-cell-type]");
    var status = root.querySelector("[data-pub-status]");
    if (!filters.length || !entries.length) return;

    var citations = root.querySelector(".citation-growth");
    if (citations) {
      var wide = window.matchMedia("(min-width: 36em)");
      citations.open = wide.matches;
      wide.addEventListener("change", function (event) { citations.open = event.matches; });
    }

    function apply(type) {
      var count = 0;
      entries.forEach(function (el) {
        var show = type === "all" || el.getAttribute("data-type") === type;
        el.hidden = !show;
        if (show) count++;
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
      status.textContent = count === 1 ? "Showing 1 publication." : "Showing " + count + " publications.";
    }

    filters.forEach(function (b) {
      b.addEventListener("click", function () {
        apply(b.getAttribute("data-filter"));
      });
    });

    function revealTarget(hash) {
      var target = document.getElementById(hash.slice(1));
      if (target && target.hidden && root.contains(target)) {
        apply("all");
        return target;
      }
    }

    // Reveal filtered years and research-tree references before native anchor navigation.
    document.addEventListener("click", function (event) {
      if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey) return;
      var link = event.target.closest('a[href^="#"]');
      if (link) revealTarget(link.getAttribute("href"));
    });
    window.addEventListener("hashchange", function () {
      var target = revealTarget(window.location.hash);
      if (target) target.scrollIntoView();
    });

    initCopy(root, status);
  }

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  function initCopy(root, status) {
    var manual = root.querySelector(".pubtrack__manual");
    var field = manual.querySelector("textarea");
    var latestRequest = 0;
    root.querySelectorAll(".pubtrack__copy").forEach(function (button) {
      var reset;
      button.addEventListener("click", async function () {
        var request = ++latestRequest;
        var reference = button.getAttribute("data-copy-reference");
        var text = reference || button.getAttribute("data-copy-bibtex");
        clearTimeout(reset);
        button.classList.remove("is-copied");
        manual.hidden = true;
        status.textContent = "";
        try {
          await navigator.clipboard.writeText(text);
        } catch {
          if (request !== latestRequest) return;
          button.closest(".pubtrack__links").after(manual);
          field.value = text;
          manual.hidden = false;
          field.focus();
          field.select();
          status.textContent = "Automatic copying is unavailable. The citation text is selected for manual copying.";
          return;
        }
        if (request !== latestRequest) return;
        button.classList.add("is-copied");
        status.textContent = reference ? "Reference copied." : "BibTeX copied.";
        reset = setTimeout(function () { button.classList.remove("is-copied"); }, 2000);
      });
    });
  }

  ready(function () {
    document.querySelectorAll("[data-pubtrack]").forEach(function (root) {
      init(root);
    });
  });
})();
