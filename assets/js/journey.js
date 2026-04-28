(function () {
  "use strict";

  function init(root) {
    var tabs = root.querySelectorAll("[data-journey-tab]");
    var panels = root.querySelectorAll("[data-journey-panel]");
    if (!tabs.length || !panels.length) return;

    function activate(key) {
      tabs.forEach(function (t) {
        var on = t.getAttribute("data-journey-tab") === key;
        t.classList.toggle("is-active", on);
        t.setAttribute("aria-selected", on ? "true" : "false");
      });
      panels.forEach(function (p) {
        var on = p.getAttribute("data-journey-panel") === key;
        p.classList.toggle("is-active", on);
        p.setAttribute("aria-hidden", on ? "false" : "true");
      });
    }

    tabs.forEach(function (t) {
      t.addEventListener("click", function () {
        activate(t.getAttribute("data-journey-tab"));
      });
      t.addEventListener("keydown", function (e) {
        if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
        e.preventDefault();
        var list = Array.prototype.slice.call(tabs);
        var idx = list.indexOf(t);
        var next = e.key === "ArrowRight" ? (idx + 1) % list.length : (idx - 1 + list.length) % list.length;
        list[next].focus();
        activate(list[next].getAttribute("data-journey-tab"));
      });
    });
  }

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    document.querySelectorAll("[data-journey]").forEach(init);
  });
})();
