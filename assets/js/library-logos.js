(function () {
  "use strict";

  var root = document.querySelector("[data-library-showcase]");
  if (!root) return;

  var toggle = root.querySelector("[data-library-motion-toggle]");
  var motion = window.matchMedia("(prefers-reduced-motion: reduce)");
  var userPaused = false;
  var scenes = Array.from(root.querySelectorAll("[data-library-scene]"), function (scene) {
    var image = scene.querySelector("img");
    var state = {
      scene: scene,
      visible: !window.IntersectionObserver,
      loaded: image ? !!(image.complete && image.naturalWidth) : !!scene.querySelector("svg"),
    };
    if (image) {
      image.addEventListener("load", function () {
        state.loaded = image.naturalWidth > 0;
        update();
      });
      image.addEventListener("error", function () {
        state.loaded = false;
        update();
      });
    }
    return state;
  });

  function update() {
    var allowed = !userPaused && !document.hidden && !motion.matches;
    scenes.forEach(function (state) {
      state.scene.classList.toggle("is-playing", allowed && state.visible && state.loaded);
    });
    if (toggle) {
      toggle.hidden = motion.matches;
      toggle.setAttribute("aria-pressed", String(userPaused));
      toggle.textContent = userPaused ? "Play animations" : "Pause animations";
    }
  }

  if (window.IntersectionObserver) {
    var observer = new window.IntersectionObserver(function (entries) {
      entries.forEach(function (entry) {
        var state = scenes.find(function (item) { return item.scene === entry.target; });
        if (state) state.visible = entry.isIntersecting;
      });
      update();
    });
    scenes.forEach(function (state) { observer.observe(state.scene); });
  }
  if (toggle) toggle.addEventListener("click", function () {
    userPaused = !userPaused;
    update();
  });
  document.addEventListener("visibilitychange", update);
  if (motion.addEventListener) motion.addEventListener("change", update);
  else motion.addListener(update);
  update();
})();
