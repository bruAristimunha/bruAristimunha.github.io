// Run: node scripts/check-library-logos.cjs
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { runInNewContext } = require('node:vm');
const source = readFileSync(resolve(__dirname, '../assets/js/library-logos.js'), 'utf8');

function events(properties = {}) {
  const listeners = {};
  return Object.assign(properties, {
    addEventListener(name, fn) { listeners[name] = fn; },
    emit(name) { listeners[name]?.(); },
  });
}

function setup({ observer = true, reduced = false, inline = false } = {}) {
  const images = [true, false].map(loaded => events({ complete: loaded, naturalWidth: loaded ? 1774 : 0 }));
  const scenes = images.map((image, i) => {
    const classes = new Set();
    return {
      querySelector: selector => inline && i === 0 ? (selector === 'svg' ? {} : null) : (selector === 'img' ? image : null),
      classList: { toggle(name, on) { on ? classes.add(name) : classes.delete(name); } },
      playing: () => classes.has('is-playing'),
    };
  });
  const button = events({ hidden: true, attributes: {}, setAttribute(name, value) { this.attributes[name] = value; } });
  const root = { querySelector: () => button, querySelectorAll: () => scenes };
  const document = events({ hidden: false, querySelector: () => root });
  const motion = events({ matches: reduced });
  const window = { matchMedia: () => motion };
  let report;
  if (observer) window.IntersectionObserver = class {
    constructor(callback) { report = callback; }
    observe() {}
  };
  runInNewContext(source, { document, window });
  return {
    images, scenes, button, document, motion,
    visible(index, on) { report([{ target: scenes[index], isIntersecting: on }]); },
    playing: () => scenes.map(scene => scene.playing()),
  };
}

// Other pages need neither a showcase nor browser media APIs.
runInNewContext(source, { document: { querySelector: () => null } });
const ui = setup();
assert.deepEqual(ui.playing(), [false, false]);
assert.equal(ui.button.hidden, false);
ui.visible(0, true);
ui.visible(1, true);
assert.deepEqual(ui.playing(), [true, false], 'Wait for the logo image to load.');
ui.images[1].naturalWidth = 1774;
ui.images[1].emit('load');
assert.deepEqual(ui.playing(), [true, true]);

ui.button.emit('click');
assert.deepEqual(ui.playing(), [false, false]);
assert.equal(ui.button.attributes['aria-pressed'], 'true');
assert.equal(ui.button.textContent, 'Play animations');
ui.button.emit('click');
assert.deepEqual(ui.playing(), [true, true]);
assert.equal(ui.button.attributes['aria-pressed'], 'false');
assert.equal(ui.button.textContent, 'Pause animations');

ui.visible(0, false);
assert.deepEqual(ui.playing(), [false, true], 'Offscreen scenes stop independently.');
ui.document.hidden = true;
ui.document.emit('visibilitychange');
assert.deepEqual(ui.playing(), [false, false]);
ui.document.hidden = false;
ui.document.emit('visibilitychange');
assert.deepEqual(ui.playing(), [false, true]);
ui.images[1].emit('error');
assert.deepEqual(ui.playing(), [false, false], 'A failed image never animates.');
ui.visible(0, true);
ui.motion.matches = true;
ui.motion.emit('change');
assert.deepEqual(ui.playing(), [false, false]);
assert.equal(ui.button.hidden, true);
ui.motion.matches = false;
ui.motion.emit('change');
assert.deepEqual(ui.playing(), [true, false]);
ui.button.emit('click');
ui.motion.matches = true;
ui.motion.emit('change');
ui.motion.matches = false;
ui.motion.emit('change');
assert.deepEqual(ui.playing(), [false, false], 'Keep the user pause choice after a media preference change.');
assert.equal(ui.button.attributes['aria-pressed'], 'true');

assert.deepEqual(setup({ observer: false }).playing(), [true, false], 'Without IntersectionObserver, loaded scenes can play.');
const reduced = setup({ observer: false, reduced: true });
assert.deepEqual(reduced.playing(), [false, false]);
assert.equal(reduced.button.hidden, true);
assert.deepEqual(setup({ observer: false, inline: true }).playing(), [true, false], 'Inline SVG logos are ready without an image load event.');
console.log('PASS: logo loading, failures, pause/resume, scene/page visibility, reduced motion, and observer fallback.');
