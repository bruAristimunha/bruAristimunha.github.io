/* Research tree: grows an SVG silhouette of a young tree from the ordered
   list rendered by _includes/research_map.html. Trunk = root, limbs = areas
   (placed on the trunk by size rank), sub-branches = topics, leaves = papers.
   Geometry lives here; every visual quality (colour, growth, sway, hover)
   is CSS in assets/css/main.scss. No dependencies. */
(function () {
  'use strict';
  var root = document.querySelector('.research-map[data-research-tree]');
  if (!root) { return; }
  var svg = root.querySelector('.research-map__svg');
  var sky = root.querySelector('.research-map__sky');
  var list = root.querySelector('.research-map__list');
  var card = root.querySelector('.research-map__card');
  var toggle = root.querySelector('.research-map__toggle');
  if (!svg || !list) { return; }

  var NS = 'http://www.w3.org/2000/svg';
  var reduce = window.matchMedia('(prefers-reduced-motion: reduce)');
  var coarse = window.matchMedia('(hover: none)');
  var SPEED = 340; /* growth, viewBox units per second */

  /* --- data ------------------------------------------------------------ */
  function paperOf(li) {
    var link = li.querySelector('.rm-paper__link');
    return {
      id: li.getAttribute('data-id'), num: li.getAttribute('data-num'),
      type: li.getAttribute('data-type') || 'abstract',
      label: li.getAttribute('data-label') || '', venue: li.getAttribute('data-venue') || '',
      status: li.getAttribute('data-status') || '', figure: li.getAttribute('data-figure') || '',
      href: link && link.getAttribute('href')
    };
  }
  var areas = Array.prototype.map.call(list.querySelectorAll('.rm-area'), function (li, i) {
    var topics = Array.prototype.map.call(li.querySelectorAll('.rm-topic'), function (t) {
      return { name: t.getAttribute('data-name'), papers: Array.prototype.map.call(t.querySelectorAll('.rm-paper'), paperOf) };
    });
    var direct = Array.prototype.filter.call(li.querySelectorAll('.rm-paper'), function (p) { return !p.closest('.rm-topic'); }).map(paperOf);
    return { name: li.getAttribute('data-name'), order: i, topics: topics, papers: direct, n: li.querySelectorAll('.rm-paper').length };
  });

  /* --- geometry helpers ------------------------------------------------ */
  var RAD = Math.PI / 180;
  function pt(x, y) { return { x: x, y: y }; }
  function polar(p, ang, len) { return pt(p.x + Math.cos(ang * RAD) * len, p.y + Math.sin(ang * RAD) * len); }
  function rng(seed) { return function () { seed = (seed * 1664525 + 1013904223) >>> 0; return seed / 4294967296; }; }
  function f(v) { return Math.round(v * 10) / 10; }
  /* Cubic branch from a to b that sags under its chord, then lifts (bow>0),
     with a faint random wobble so no two branches are alike. */
  function branchCurve(a, b, bow, r) {
    var dx = b.x - a.x, dy = b.y - a.y, L = Math.hypot(dx, dy) || 1, nx = -dy / L, ny = dx / L;
    if (ny < 0) { nx = -nx; ny = -ny; }
    var w1 = bow * L * (0.8 + 0.5 * r()), w2 = bow * L * (0.2 + 0.5 * r()) * (r() < 0.35 ? -0.6 : 1);
    return { a: a, b: b, L: L,
      c1: pt(a.x + dx / 3 + nx * w1, a.y + dy / 3 + ny * w1),
      c2: pt(a.x + 2 * dx / 3 + nx * w2, a.y + 2 * dy / 3 + ny * w2) };
  }
  function cPoint(q, t) {
    var u = 1 - t, a = u * u * u, b = 3 * u * u * t, c = 3 * u * t * t, d = t * t * t;
    return pt(a * q.a.x + b * q.c1.x + c * q.c2.x + d * q.b.x, a * q.a.y + b * q.c1.y + c * q.c2.y + d * q.b.y);
  }
  function cAngle(q, t) { var p = cPoint(q, Math.max(0, t - 0.02)), n = cPoint(q, Math.min(1, t + 0.02)); return Math.atan2(n.y - p.y, n.x - p.x) / RAD; }
  function cLength(q) { var s = 0, p = q.a; for (var i = 1; i <= 12; i++) { var n = cPoint(q, i / 12); s += Math.hypot(n.x - p.x, n.y - p.y); p = n; } return s; }
  function cPath(q) { return 'M' + f(q.a.x) + ' ' + f(q.a.y) + 'C' + f(q.c1.x) + ' ' + f(q.c1.y) + ' ' + f(q.c2.x) + ' ' + f(q.c2.y) + ' ' + f(q.b.x) + ' ' + f(q.b.y); }
  function cSplit(q, s) { /* de Casteljau: the part of q from 0 to s */
    function lp(p, n) { return pt(p.x + (n.x - p.x) * s, p.y + (n.y - p.y) * s); }
    var q0 = lp(q.a, q.c1), q1 = lp(q.c1, q.c2), q2 = lp(q.c2, q.b), r0 = lp(q0, q1), r1 = lp(q1, q2);
    return { a: q.a, c1: q0, c2: r0, b: lp(r0, r1) };
  }
  function lerpAngle(a, b, k) { var d = ((b - a + 540) % 360) - 180; return a + d * k; }
  /* Long names break into two lines at the space nearest their middle. */
  function wrapText(text) {
    if (text.length <= 22) { return [text]; }
    var best = -1, mid = text.length / 2;
    for (var i = text.indexOf(' '); i >= 0; i = text.indexOf(' ', i + 1)) { if (best < 0 || Math.abs(i - mid) < Math.abs(best - mid)) { best = i; } }
    return best < 0 ? [text] : [text.slice(0, best), text.slice(best + 1)];
  }
  function labelHalf(text, em) { return Math.max.apply(null, wrapText(text).map(function (l) { return l.length; })) * em * 0.5; }
  /* Would a tip label of half-width hw, hung off a branch ending at `end` with direction `ang`, stay inside the canvas? */
  function labelFits(end, ang, hw, W) {
    var c = Math.cos(ang * RAD);
    if (c > 0.25) { return end.x + 10 + 2 * hw <= W - 6; }
    if (c < -0.25) { return end.x - 10 - 2 * hw >= 6; }
    return end.x - hw >= 6 && end.x + hw <= W - 6;
  }

  /* --- layout modes. Slots are ranked by area size: the three largest areas
     form the fork at the top of the trunk, the rest branch off lower. ------ */
  var MODES = {
    wide: { W: 960, H: 640, trunkX: 0.35, forkY: 0.62, lean: 0.03, limb: function (n) { return 92 + 60 * Math.sqrt(n); }, sub: function (n) { return 58 + 38 * Math.sqrt(n); }, spread: 46, twig: [28, 22],
      slots: [{ t: 1, ang: -33 }, { t: 1, ang: -127 }, { t: 1, ang: -86 }, { t: 0.55, ang: 4, len: 1.3 }, { t: 0.78, ang: -8 }, { t: 0.42, ang: -150 }, { t: 0.88, ang: -140, len: 0.9 }, { t: 0.3, ang: 10, len: 1.1 }, { t: 0.66, ang: -160 }, { t: 0.2, ang: -165 }] },
    tall: { W: 420, H: 880, trunkX: 0.47, forkY: 0.52, lean: 0.02, limb: function (n) { return 76 + 50 * Math.sqrt(n); }, sub: function (n) { return 48 + 30 * Math.sqrt(n); }, spread: 60, twig: [22, 18],
      slots: [{ t: 1, ang: -76 }, { t: 1, ang: -122 }, { t: 1, ang: -92 }, { t: 0.62, ang: -44 }, { t: 0.78, ang: -150 }, { t: 0.44, ang: -146 }, { t: 0.88, ang: -122, len: 0.9 }, { t: 0.54, ang: -14 }, { t: 0.3, ang: -160 }, { t: 0.7, ang: -8 }] }
  };

  function layout(mode, extraH) {
    var M = MODES[mode], W = M.W, H = M.H + (extraH || 0), r = rng(7), wood = [], labels = [], leaves = [];
    var base = pt(M.trunkX * W, H + 8), fork = pt((M.trunkX + M.lean) * W, M.forkY * H);
    var trunk = branchCurve(base, fork, 0.05, r), trunkLen = cLength(trunk);
    /* Five nested strokes of decreasing length taper the trunk. */
    [[1, 4.6], [0.8, 5.3], [0.6, 6.1], [0.4, 6.9], [0.2, 7.7]].forEach(function (s) { wood.push({ q: cSplit(trunk, s[0]), w: s[1], d0: 0, cls: 'rt-trunk' }); });

    function leavesAlong(q, papers, d0, depth) {
      var n = papers.length, L = cLength(q), flip = 1;
      papers.forEach(function (p, i) {
        var t = n === 1 ? 0.7 : 0.22 + 0.58 * i / (n - 1), side = (i % 2 ? 1 : -1) * flip;
        var at = cPoint(q, t), tang = cAngle(q, t), pr = rng(Number(p.num) * 131 + 17);
        /* Twigs on the upper side reach for the sky; on the lower side they hang. */
        var raw = tang + side * 58, up = Math.sin(raw * RAD) < Math.sin(tang * RAD) - 0.1;
        var ang = lerpAngle(raw, -90, up ? 0.3 : 0.08), len = M.twig[0] + M.twig[1] * pr();
        leaves.push({ p: p, at: at, tip: polar(at, ang, len), ang: ang, d0: d0 + t * L, seed: pr() * 1e6 });
      });
      if (n >= 3 && depth < 2) { /* one bare twig for air, like the photo */
        var bt = 0.15 + 0.3 * r(), bp = cPoint(q, bt), ba = lerpAngle(cAngle(q, bt) - flip * 40, -90, 0.3);
        wood.push({ q: branchCurve(bp, polar(bp, ba, 14 + 10 * r()), 0.1, r), w: 0.75, d0: d0 + bt * L, cls: 'rt-twig' });
      }
    }

    var ranked = areas.slice().sort(function (a, b) { return (b.n - a.n) || (a.order - b.order); });
    ranked.forEach(function (area, rank) {
      var slot = M.slots[rank % M.slots.length], start = cPoint(trunk, slot.t), d0 = slot.t * trunkLen;
      var q = branchCurve(start, polar(start, slot.ang, M.limb(area.n) * (slot.len || 1)), 0.12, r), L = cLength(q);
      wood.push({ q: q, w: rank === 0 ? 3 : 2.4, d0: d0, cls: 'rt-limb' });
      labels.push({ text: area.name, q: q, cls: 'rt-label--area', d0: d0 + L, tip: !area.topics.length });
      if (area.topics.length) {
        var m = area.topics.length, subs = area.topics.map(function (tp, j) {
          return { tp: tp, ang: slot.ang + (m === 1 ? 0 : -M.spread / 2 + M.spread * j / (m - 1)) + (r() - 0.5) * 8, len: M.sub(tp.papers.length) };
        });
        /* Bend the whole fan of topics skyward, a step at a time, until every
           topic's name fits inside the canvas beyond its tip. */
        var bend = [0.15, 0.3, 0.45, 0.6, 0.75, 0.9].filter(function (k) {
          return subs.every(function (sb) { var a2 = lerpAngle(sb.ang, -90, k); return labelFits(polar(q.b, a2, sb.len), a2, labelHalf(sb.tp.name, 5.9), W); });
        })[0];
        subs.forEach(function (sb) {
          var a2 = lerpAngle(sb.ang, -90, bend === undefined ? 0.15 : bend);
          var s2 = branchCurve(q.b, polar(q.b, a2, sb.len), 0.1, r);
          wood.push({ q: s2, w: 1.5, d0: d0 + L, cls: 'rt-sub' });
          labels.push({ text: sb.tp.name, q: s2, cls: 'rt-label--topic', d0: d0 + L + cLength(s2), tip: true });
          leavesAlong(s2, sb.tp.papers, d0 + L, 1);
        });
      } else { leavesAlong(q, area.papers, d0, 0); }
    });

    /* Labels: a terminal branch is named just beyond its tip, in the direction
       it grows; a branch that forks into topics is named under its middle. */
    labels.forEach(function (lb) {
      var area = lb.cls === 'rt-label--area', em = area ? 7 : 5.9;
      lb.lines = wrapText(lb.text); lb.hw = labelHalf(lb.text, em); lb.anchor = 'middle';
      var half = (lb.lines.length - 1) * 6.5; /* extra half-height of a second line */
      if (lb.tip) {
        var a = cAngle(lb.q, 1), c = Math.cos(a * RAD), p = polar(lb.q.b, a, 10);
        lb.anchor = c > 0.25 ? 'start' : c < -0.25 ? 'end' : 'middle';
        lb.ax = p.x; lb.y = p.y + (lb.anchor === 'middle' ? -6 - half : 4);
        lb.x = lb.anchor === 'start' ? p.x + lb.hw : lb.anchor === 'end' ? p.x - lb.hw : p.x;
        if (lb.anchor === 'start' && lb.x + lb.hw > W - 6) { lb.anchor = 'end'; lb.ax = lb.q.b.x - 2; lb.y = lb.q.b.y + 20 + half; lb.x = lb.ax - lb.hw; }
        else if (lb.anchor === 'end' && lb.x - lb.hw < 6) { lb.anchor = 'start'; lb.ax = lb.q.b.x + 2; lb.y = lb.q.b.y + 20 + half; lb.x = lb.ax + lb.hw; }
      } else {
        var m = cPoint(lb.q, 0.6), ma = cAngle(lb.q, 0.6) * RAD, nx = -Math.sin(ma), ny = Math.cos(ma), c = Math.cos(ma);
        if (ny < 0) { nx = -nx; ny = -ny; }
        lb.anchor = c > 0.25 ? 'start' : c < -0.25 ? 'end' : 'middle';
        lb.ax = m.x + nx * 11; lb.y = m.y + ny * 11 + 5;
        lb.x = lb.anchor === 'start' ? lb.ax + lb.hw : lb.anchor === 'end' ? lb.ax - lb.hw : lb.ax;
      }
      var shift = Math.min(Math.max(lb.x, lb.hw + 6), W - lb.hw - 6) - lb.x;
      lb.x += shift; lb.ax += shift;
    });

    /* Relax: labels step off each other; leaves keep their distance from
       each other, from labels and from the edges. */
    for (var pass = 0; pass < 3; pass++) {
      labels.forEach(function (a, i) {
        labels.forEach(function (b, j) {
          if (j <= i) { return; }
          var dy = b.y - a.y, need = 17 + 6.5 * (a.lines.length + b.lines.length - 2) + (a.cls === 'rt-label--area' && b.cls === 'rt-label--area' ? 6 : 2);
          if (Math.abs(b.x - a.x) < a.hw + b.hw + 10 && Math.abs(dy) < need) { b.y += dy < 0 ? -(need - Math.abs(dy)) : (need - Math.abs(dy)); }
        });
      });
    }
    /* A paper has one leaf, centred 13 units beyond its twig tip.
       Leave room around the hit targets and the branch labels. */
    var HIT = mode === 'tall' ? 15 : 12, MIN = 2 * HIT + 16;
    function hitC(lf) { var a = Math.atan2(lf.tip.y - lf.at.y, lf.tip.x - lf.at.x); return pt(lf.tip.x + 13 * Math.cos(a), lf.tip.y + 13 * Math.sin(a)); }
    for (var it = 0; it < 32; it++) {
      leaves.forEach(function (a, i) {
        leaves.forEach(function (b, j) {
          if (j <= i) { return; }
          var ca = hitC(a), cb = hitC(b), dx = cb.x - ca.x, dy = cb.y - ca.y, d = Math.hypot(dx, dy) || 0.01;
          if (d < MIN) { var k = (MIN - d) / d * 0.5; a.tip.x -= dx * k; a.tip.y -= dy * k; b.tip.x += dx * k; b.tip.y += dy * k; }
        });
        labels.forEach(function (lb) {
          /* Keep the leaf's centre clear, regardless of its twig's angle. */
          var center = hitC(a), dx = center.x - lb.x, dy = center.y - (lb.y - 4), clear = HIT + 12 + 6.5 * (lb.lines.length - 1);
          if (Math.abs(dx) < lb.hw + HIT + 6 && Math.abs(dy) < clear) { a.tip.y += dy < 0 ? -(clear - Math.abs(dy)) : (clear - Math.abs(dy)); }
        });
        /* Wide mode: the heading block sits bottom-left (CSS: left 3.6%, width 27%, bottom 6%); keep clusters above it. */
        if (mode === 'wide' && a.tip.x < 0.33 * W + 20 && a.tip.y > 0.72 * H - 26) { a.tip.y = 0.72 * H - 26; }
        a.tip.x = Math.min(Math.max(a.tip.x, 18), W - 18); a.tip.y = Math.min(Math.max(a.tip.y, 26), H - 26);
      });
    }
    leaves.forEach(function (lf) { lf.ang = Math.atan2(lf.tip.y - lf.at.y, lf.tip.x - lf.at.x) / RAD; });
    /* Whatever still reaches the top edge pushes the whole tree down a little (the trunk base is off-canvas anyway). */
    var top = Math.min.apply(null, leaves.map(function (lf) { return lf.tip.y - 30; }).concat(labels.map(function (lb) { return lb.y - 12 - 6.5 * (lb.lines.length - 1); })));
    var pad = Math.max(0, 12 - top);
    /* With many papers the pad can push clusters or labels below the canvas; grow the canvas once and lay out again. */
    var bottom = Math.max.apply(null, leaves.map(function (lf) { return lf.tip.y + 30; }).concat(labels.map(function (lb) { return lb.y + 12 + 6.5 * (lb.lines.length - 1); })));
    var over = bottom + pad - (H - 12);
    if (over > 0 && !extraH) { return layout(mode, Math.ceil(over)); }
    return { W: W, H: H, base: base, wood: wood, labels: labels, leaves: leaves, pad: pad };
  }

  /* --- render ----------------------------------------------------------- */
  function el(name, attrs, parent) {
    var e = document.createElementNS(NS, name);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    if (parent) { parent.appendChild(e); }
    return e;
  }
  function leafletPath(len, hw) { return 'M0 0C' + f(hw) + ' ' + f(-len * 0.3) + ' ' + f(hw * 0.9) + ' ' + f(-len * 0.72) + ' 0 ' + f(-len) + 'C' + f(-hw * 0.9) + ' ' + f(-len * 0.72) + ' ' + f(-hw) + ' ' + f(-len * 0.3) + ' 0 0Z'; }

  var pad = 0, leanNodes = [];
  function render(mode) {
    var G = layout(mode);
    Array.prototype.slice.call(svg.children).forEach(function (c) { if (c.tagName !== 'title' && c.tagName !== 'desc') { svg.removeChild(c); } });
    svg.setAttribute('viewBox', '0 0 ' + G.W + ' ' + G.H);
    pad = G.pad; leanNodes = [];
    var canvas = el('g', { 'class': 'rt-canvas', transform: 'translate(0 ' + f(G.pad) + ')' }, svg);
    var wood = el('g', { 'class': 'rt-wood', 'aria-hidden': 'true' }, canvas);
    var leaves = el('g', { 'class': 'rt-leaves' }, canvas);
    var labels = el('g', { 'class': 'rt-labels', 'aria-hidden': 'true' }, canvas);
    function stroke(item, parent) {
      var L = cLength(item.q);
      return el('path', { d: cPath(item.q), 'class': 'rt-wood__path ' + item.cls, 'stroke-width': item.w,
        style: '--len:' + f(L + item.w + 4) + ';--off:' + f(L + 1.5 * item.w + 8) + ';--d:' + f(item.d0 / SPEED) + 's;--dur:' + f(Math.max(0.25, L / SPEED)) + 's' }, parent);
    }
    G.wood.forEach(function (w) { stroke(w, wood); });
    G.labels.forEach(function (lb) {
      var t = el('text', { 'class': 'rt-label ' + lb.cls, 'text-anchor': lb.anchor, style: '--d:' + f(lb.d0 / SPEED + 0.1) + 's' }, labels);
      lb.lines.forEach(function (line, i) {
        el('tspan', { x: f(lb.ax), y: f(lb.y + (i - (lb.lines.length - 1) / 2) * 13) }, t).textContent = line;
      });
    });
    G.leaves.forEach(function (lf) {
      var p = lf.p, r = rng(lf.seed), isLink = !!p.href;
      var a = el(isLink ? 'a' : 'g', { 'class': 'rt-leaf rt-leaf--' + p.type, 'data-id': p.id || '' }, leaves);
      if (isLink) { a.setAttribute('href', p.href); } else { a.setAttribute('tabindex', '0'); a.setAttribute('role', 'img'); }
      a.setAttribute('aria-label', 'P' + p.num + ', ' + p.label + ', ' + p.venue + (p.status ? ', ' + p.status : '') + ', ' + p.type);
      stroke({ q: branchCurve(lf.at, lf.tip, 0.1, r), w: 1.1, d0: lf.d0, cls: 'rt-twig' }, a);
      var place = el('g', { transform: 'translate(' + f(lf.tip.x) + ' ' + f(lf.tip.y) + ') rotate(' + f(lf.ang + 90) + ')' }, a);
      var grow = el('g', { 'class': 'rt-leaf__grow', style: '--d:' + f(lf.d0 / SPEED + 0.15) + 's;--sw:' + f(-r() * 7) + 's;--sd:' + f(5 + r() * 3) + 's' }, place);
      var hover = el('g', { 'class': 'rt-leaf__hover' }, grow);
      var lean = el('g', { 'class': 'rt-leaf__lean' }, hover);
      var sway = el('g', { 'class': 'rt-leaf__sway' }, lean);
      leanNodes.push({ el: lean, leaf: a, x: lf.tip.x, y: lf.tip.y, cur: 0 });
      el('circle', { 'class': 'rt-leaf__hit', r: mode === 'tall' ? 15 : 12, cy: -13 }, sway);
      el('path', { d: leafletPath(26, 10), transform: 'translate(0 0) rotate(0)', 'class': 'rt-leaflet' }, sway);
    });
    root.classList.toggle('research-map--tall', mode === 'tall');
    root.classList.add('is-ready');
  }

  /* --- hover / focus card ----------------------------------------------- */
  var shownFor = null, armed = null;
  function showCard(leaf) {
    if (!card) { return; }
    var p = paperOf(list.querySelector('.rm-paper[data-id="' + leaf.getAttribute('data-id') + '"]') || document.createElement('li'));
    var html = '<span class="research-map__card-num">P' + p.num + '</span><span class="research-map__card-title">' + p.label + '</span>' +
      '<span class="research-map__card-meta">' + p.venue + (p.status ? ' (' + p.status + ')' : '') + ', ' + p.type + '</span>';
    if (p.figure && !coarse.matches) { html += '<img class="research-map__card-fig" src="' + p.figure + '" alt="" loading="lazy">'; }
    if (p.href && coarse.matches) { html += '<a class="research-map__card-go" href="' + p.href + '">Go to the reference</a>'; }
    card.innerHTML = html; card.hidden = false; shownFor = leaf;
    placeCard();
    var img = card.querySelector('img');
    if (img && !img.complete) { img.addEventListener('load', placeCard); } /* the lazy image changes the card's height after the first measure */
  }
  function placeCard() {
    if (!card || !shownFor || card.hidden) { return; }
    var hit = shownFor.querySelector('.rt-leaf__hit').getBoundingClientRect(), box = sky.getBoundingClientRect();
    var cx = hit.left + hit.width / 2 - box.left, top = hit.top - box.top;
    var w = card.offsetWidth, h = card.offsetHeight;
    card.style.left = Math.min(Math.max(cx - w / 2, 8), box.width - w - 8) + 'px';
    card.style.top = (top - h - 14 >= 4 ? top - h - 14 : top + hit.height + 12) + 'px';
    card.classList.toggle('is-below', top - h - 14 < 4);
  }
  if (card && 'ResizeObserver' in window) { new ResizeObserver(placeCard).observe(card); }
  function hideCard() { if (card) { card.hidden = true; } shownFor = null; armed = null; }
  svg.addEventListener('mouseover', function (e) { var lf = e.target.closest && e.target.closest('.rt-leaf'); if (lf && !coarse.matches) { showCard(lf); } });
  svg.addEventListener('mouseout', function (e) { var lf = e.target.closest && e.target.closest('.rt-leaf'); if (lf && !coarse.matches && !lf.contains(e.relatedTarget)) { hideCard(); } });
  svg.addEventListener('focusin', function (e) { var lf = e.target.closest && e.target.closest('.rt-leaf'); if (lf) { showCard(lf); } });
  svg.addEventListener('focusout', function (e) { if (!coarse.matches) { hideCard(); } });
  var lastPointer = null; /* pointer type of the press that leads to the next click; keyboard activation leaves it null */
  svg.addEventListener('pointerdown', function (e) { lastPointer = e.pointerType || 'touch'; });
  document.addEventListener('keydown', function () { lastPointer = null; });
  svg.addEventListener('click', function (e) {
    var lf = e.target.closest && e.target.closest('.rt-leaf'), byTouch = lastPointer === 'touch' || lastPointer === 'pen';
    lastPointer = null;
    if (lf && byTouch && armed !== lf) { e.preventDefault(); showCard(lf); armed = lf; }
  });
  document.addEventListener('click', function (e) { if (shownFor && !svg.contains(e.target) && !card.contains(e.target)) { hideCard(); } });
  document.addEventListener('keydown', function (e) { if (e.key === 'Escape') { hideCard(); } });

  /* Legend: hovering or focusing a type lets the other leaves fade back. */
  root.querySelectorAll('.research-map__legend-item').forEach(function (item) {
    function on() { root.setAttribute('data-highlight', item.getAttribute('data-type')); }
    function off() { root.removeAttribute('data-highlight'); }
    item.addEventListener('mouseenter', on); item.addEventListener('mouseleave', off);
    item.addEventListener('focus', on); item.addEventListener('blur', off);
  });
  if (toggle) {
    list.hidden = true; toggle.hidden = false;
    toggle.addEventListener('click', function () { list.hidden = !list.hidden; toggle.setAttribute('aria-expanded', String(!list.hidden)); });
  }

  /* --- Pointer: clusters near a fine pointer lean a few degrees away from it
     (nothing within the hit radius, so hover never fights it) and the whole
     tree parallaxes a touch toward it. Smoothed per frame, idle when settled. */
  var fine = window.matchMedia('(pointer: fine)');
  var pointer = null, raf = 0, par = { x: 0, y: 0 };
  function tick() {
    raf = 0;
    var box = svg.getBoundingClientRect(), M = MODES[mode], k = box.width / M.W, settled = true;
    var ux = pointer ? (pointer.x - box.left - par.x) / k : 0, uy = pointer ? (pointer.y - box.top - par.y) / k - pad : 0;
    leanNodes.forEach(function (n) {
      var target = 0;
      if (pointer) {
        var dx = n.x - ux, dy = n.y - uy, d = Math.hypot(dx, dy) || 1;
        var w = Math.min(1, Math.max(0, (d - 28) / 40)) * Math.min(1, Math.max(0, (180 - d) / 90));
        target = 5 * w * dx / d;
      }
      n.cur += (target - n.cur) * 0.1;
      if (Math.abs(target - n.cur) > 0.03) { settled = false; } else { n.cur = target; }
      n.el.style.transform = n.cur ? 'rotate(' + n.cur.toFixed(2) + 'deg)' : '';
    });
    var tx = 0, ty = 0;
    if (pointer) { tx = shownFor ? par.x : (ux / M.W - 0.5) * box.width * 0.012; ty = shownFor ? par.y : (uy / M.H - 0.5) * box.height * 0.012; }
    par.x += (tx - par.x) * 0.1; par.y += (ty - par.y) * 0.1;
    if (Math.abs(tx - par.x) > 0.05 || Math.abs(ty - par.y) > 0.05) { settled = false; }
    svg.style.transform = Math.abs(par.x) + Math.abs(par.y) > 0.05 ? 'translate(' + par.x.toFixed(2) + 'px,' + par.y.toFixed(2) + 'px)' : '';
    if (!settled) { raf = requestAnimationFrame(tick); }
  }
  function nudge() { if (!raf) { raf = requestAnimationFrame(tick); } }
  sky.addEventListener('pointermove', function (e) {
    if (!fine.matches || reduce.matches || (e.pointerType && e.pointerType !== 'mouse')) { return; }
    pointer = { x: e.clientX, y: e.clientY }; nudge();
  });
  sky.addEventListener('pointerleave', function () { pointer = null; nudge(); });

  /* --- The moment: every 25-45 s while on screen, one leaflet lets go,
     tumbles down through open sky with a slow drift and a turn, fades out
     near the ground, and its cluster quietly regrows it over ~2 s. --- */
  var momentTimer;
  function fallingLeaf() {
    var canvas = svg.querySelector('.rt-canvas'), M = MODES[mode];
    if (!canvas || !canvas.getCTM || !canvas.animate) { return; }
    var pool = leanNodes.filter(function (n) { /* clusters with nothing below them */
      return n.leaf !== shownFor && !leanNodes.some(function (o) { return o !== n && Math.abs(o.x - n.x) < 45 && o.y > n.y; });
    });
    if (!pool.length) { return; }
    var n = pool[Math.floor(Math.random() * pool.length)], leaflets = n.leaf.querySelectorAll('.rt-leaflet');
    var lf = leaflets[Math.floor(Math.random() * leaflets.length)], m = canvas.getCTM().inverse().multiply(lf.getCTM());
    var base = 'matrix(' + [m.a, m.b, m.c, m.d, m.e, m.f].map(function (v) { return v.toFixed(3); }).join(',') + ')';
    var type = (n.leaf.getAttribute('class').match(/rt-leaf--\w+/) || [''])[0];
    var g = el('g', { 'class': 'rt-fall ' + type, 'aria-hidden': 'true' }, canvas);
    el('path', { d: lf.getAttribute('d'), 'class': 'rt-leaflet' }, g);
    var drop = Math.max(60, M.H - pad - m.f - 14), dur = 2600 + drop * 3.2;
    var drift = (Math.random() < 0.5 ? -1 : 1) * (14 + Math.random() * 16), spin = (Math.random() < 0.5 ? -1 : 1) * (150 + Math.random() * 130);
    var frames = [];
    for (var i = 0; i <= 14; i++) {
      var p = i / 14, y = drop * Math.pow(p, 1.3), x = drift * Math.sin(p * Math.PI * 2.4) + drift * 0.9 * p;
      frames.push({ offset: p, opacity: p < 0.7 ? 1 : 1 - (p - 0.7) / 0.3, transform: 'translate(' + x.toFixed(1) + 'px,' + y.toFixed(1) + 'px) ' + base + ' rotate(' + (spin * p).toFixed(1) + 'deg)' });
    }
    g.animate(frames, { duration: dur, easing: 'linear', fill: 'forwards' }).onfinish = function () { g.remove(); };
    var t = /translate\(([-\d.]+) ([-\d.]+)\) rotate\(([-\d.]+)\)/.exec(lf.getAttribute('transform') || '');
    if (t) {
      var local = 'translate(' + t[1] + 'px,' + t[2] + 'px) rotate(' + t[3] + 'deg)';
      lf.animate([{ transform: local + ' scale(0)' }, { transform: local + ' scale(1)' }], { duration: 2000, delay: 900, easing: 'cubic-bezier(0.2, 0.8, 0.3, 1)', fill: 'backwards' });
    }
  }
  function scheduleMoment() {
    clearTimeout(momentTimer);
    if (reduce.matches) { return; }
    momentTimer = setTimeout(function () {
      if (root.classList.contains('is-swaying') && !shownFor && !document.hidden) { fallingLeaf(); }
      scheduleMoment();
    }, 25000 + Math.random() * 20000);
  }
  root.addEventListener('research-tree:moment', function () { if (!reduce.matches) { fallingLeaf(); } });
  scheduleMoment();

  /* --- grow once in view, sway only while visible ---------------------- */
  var mode = root.offsetWidth < 620 ? 'tall' : 'wide';
  render(mode);
  if (reduce.matches || !('IntersectionObserver' in window)) { root.classList.add('is-grown'); }
  else {
    new IntersectionObserver(function (entries) {
      entries.forEach(function (en) {
        if (en.isIntersecting) { root.classList.add('is-grown'); }
        root.classList.toggle('is-swaying', en.isIntersecting);
      });
    }, { threshold: 0 }).observe(sky);
  }
  var resizeTimer;
  window.addEventListener('resize', function () {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(function () {
      var next = root.offsetWidth < 620 ? 'tall' : 'wide';
      if (next !== mode) { mode = next; hideCard(); render(mode); root.classList.add('is-grown'); }
      else { placeCard(); }
    }, 150);
  });
})();
