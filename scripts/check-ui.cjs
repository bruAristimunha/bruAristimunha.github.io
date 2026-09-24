// After a Jekyll build, run: node scripts/check-ui.cjs (uses Ruby's standard library).
const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { runInNewContext } = require('node:vm');
const { resolve } = require('node:path');
const source = (name) => readFileSync(resolve(__dirname, '../assets/js', name), 'utf8');

let focused;
function element(key, attribute) {
  const attributes = { [attribute]: key };
  return {
    attributes, listeners: {}, classList: { toggle() {} },
    getAttribute(name) { return attributes[name]; },
    setAttribute(name, value) { attributes[name] = value; },
    addEventListener(name, callback) { this.listeners[name] = callback; },
    focus() { focused = this; },
  };
}
const tabs = ['now', 'before', 'origin'].map(key => element(key, 'data-journey-tab'));
const panels = ['now', 'before', 'origin'].map(key => element(key, 'data-journey-panel'));
const root = { querySelectorAll: selector => selector === '[data-journey-tab]' ? tabs : panels };
runInNewContext(source('journey.js'), {
  document: { readyState: 'complete', querySelectorAll: () => [root] },
});
function selected(index) {
  assert.deepEqual(tabs.map(t => t.tabIndex), tabs.map((_, i) => i === index ? 0 : -1));
  assert.deepEqual(tabs.map(t => t.attributes['aria-selected']), tabs.map((_, i) => String(i === index)));
  assert.deepEqual(panels.map(p => p.attributes['aria-hidden']), panels.map((_, i) => String(i !== index)));
}
tabs[0].listeners.click();
selected(0);
tabs[0].listeners.keydown({ key: 'ArrowRight', preventDefault() {} });
selected(1);
assert.equal(focused, tabs[1]);
tabs[2].listeners.click();
selected(2);
tabs[2].listeners.keydown({ key: 'ArrowRight', preventDefault() {} });
selected(0);
tabs[0].listeners.keydown({ key: 'ArrowLeft', preventDefault() {} });
selected(2);

// Pages with native navigation must not enter the old overflow-menu loop.
const empty = { length: 0, resize() {}, on() {} };
runInNewContext(source('plugins/jquery.greedy-navigation.js'), { $: () => empty, window: {} });
console.log('PASS: journey selection, keyboard wrapping, focus, and absent legacy navigation.');

// Check the real publication data and both tree layouts, without a browser dependency.
const { execFileSync } = require('node:child_process');
const data = JSON.parse(execFileSync('ruby', ['-rjson', '-ryaml', '-e',
  'puts JSON.generate({areas: YAML.load_file("_data/research_areas.yml")["areas"], papers: YAML.load_file("_data/publications.yml")})',
], { cwd: resolve(__dirname, '..'), encoding: 'utf8' }));
const html = readFileSync(resolve(__dirname, '../_site/index.html'), 'utf8');
const mapped = data.papers.filter(p => p.map);
const order = [];
let previousYear = 0;
for (const match of html.matchAll(/<li class="rm-(area|topic|paper)"([^>]*)>/g)) {
  if (match[1] !== 'paper') { previousYear = 0; continue; }
  const id = /data-id="([^"]+)"/.exec(match[2])[1];
  const paper = mapped.find(p => p.id === id);
  assert.ok(paper.year >= previousYear, `${id} is out of chronological order`);
  previousYear = paper.year;
  order.push(id);
}
assert.equal(order.length, mapped.length);
assert.equal(new Set(order).size, mapped.length);
const areas = data.areas.map((area, index) => {
  const papers = mapped.filter(p => p.map.area === area.name).sort((a, b) => order.indexOf(a.id) - order.indexOf(b.id));
  return { name: area.name, order: index, n: papers.length, papers: area.topics ? [] : papers,
    topics: (area.topics || []).map(name => ({ name, papers: papers.filter(p => p.map.topic === name) })) };
});
const tree = source('research-tree.js');
// Evaluate the existing pure geometry section; production needs no test exports.
const layout = runInNewContext(tree.slice(tree.indexOf('  var RAD'), tree.indexOf('  /* --- render')) + '\nlayout;', { areas });
for (const mode of ['wide', 'tall']) {
  const geometry = layout(mode);
  assert.equal(geometry.leaves.length, mapped.length);
  const radius = mode === 'tall' ? 15 : 12;
  const centers = geometry.leaves.map(leaf => ({
    x: leaf.tip.x + 13 * Math.cos(leaf.ang * Math.PI / 180),
    y: leaf.tip.y + geometry.pad + 13 * Math.sin(leaf.ang * Math.PI / 180),
    id: leaf.p.id,
  }));
  centers.forEach((a, i) => {
    assert.ok(a.x >= radius && a.x <= geometry.W - radius && a.y >= radius && a.y <= geometry.H - radius,
      `${mode}: ${a.id} hit target extends outside the tree`);
    centers.slice(i + 1).forEach(b => assert.ok(Math.hypot(a.x - b.x, a.y - b.y) >= 2 * radius,
      `${mode}: ${a.id} and ${b.id} have overlapping hit targets`));
  });
}
console.log(`PASS: ${mapped.length} unique papers, chronological branches, and non-overlapping desktop/mobile leaf targets.`);
