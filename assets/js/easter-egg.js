(function () {
  "use strict";

  // Progressive enhancement for the .egg throwback popover. Desktop reveals on
  // :hover / keyboard :focus-visible via CSS; touch devices have neither, so
  // here we make a tap (or Enter/Space) toggle an .is-open class. Mirrors the
  // click + class-toggle pattern used in journey.js.

  function ready(fn) {
    if (document.readyState !== "loading") fn();
    else document.addEventListener("DOMContentLoaded", fn);
  }

  ready(function () {
    var eggs = Array.prototype.slice.call(document.querySelectorAll(".egg"));
    if (!eggs.length) return;

    function closeAll(except) {
      eggs.forEach(function (egg) {
        if (egg === except) return;
        egg.classList.remove("is-open");
        egg.setAttribute("aria-expanded", "false");
      });
    }

    eggs.forEach(function (egg, i) {
      egg.setAttribute("role", "button");
      egg.setAttribute("aria-expanded", "false");
      var pop = egg.querySelector(".egg__pop");
      if (pop) {
        if (!pop.id) pop.id = "egg-pop-" + i;
        egg.setAttribute("aria-controls", pop.id);
      }

      function toggle() {
        var open = !egg.classList.contains("is-open");
        closeAll(egg);
        egg.classList.toggle("is-open", open);
        egg.setAttribute("aria-expanded", open ? "true" : "false");
      }

      egg.addEventListener("click", function (e) {
        e.stopPropagation();
        toggle();
      });

      egg.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " " || e.key === "Spacebar") {
          e.preventDefault();
          toggle();
        } else if (e.key === "Escape" || e.key === "Esc") {
          egg.classList.remove("is-open");
          egg.setAttribute("aria-expanded", "false");
        }
      });
    });

    // A tap/click anywhere outside an open egg dismisses it.
    document.addEventListener("click", function () {
      closeAll(null);
    });
  });
})();
