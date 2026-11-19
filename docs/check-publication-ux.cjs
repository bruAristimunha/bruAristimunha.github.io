// Serve _site, then run with Playwright, axe-core, and Chrome installed:
// node docs/check-publication-ux.cjs http://127.0.0.1:4000
const assert = require('node:assert/strict');
const { chromium } = require('playwright');

async function checkAccessibility(page) {
  const violations = await page.evaluate(async () => {
    const results = await axe.run({ runOnly: ['wcag2a', 'wcag2aa', 'wcag21aa', 'wcag22aa', 'best-practice'] });
    return results.violations.map(issue => [issue.id, issue.nodes.map(node => node.target)]);
  });
  assert.deepEqual(violations, []);
}

async function checkPaperLayout(page, width, wide) {
  await page.setViewportSize({ width, height: 900 });
  const entry = page.locator('.pubtrack__entry:has(.pubpanel__body--split)').first();
  const citation = await entry.locator('.pubtrack__citation').boundingBox();
  const note = await entry.locator('.pubtrack__summary').boundingBox();
  const chart = page.locator('.citation-growth__chart');
  if (wide) {
    assert.ok(note.x >= citation.x + citation.width && Math.abs(note.y - citation.y) < 4, 'Takeaway beside its citation');
    const heading = await page.locator('.citation-growth summary').boundingBox();
    const years = await page.locator('.pubtrack__chart-wrap h3').boundingBox();
    assert.ok(Math.abs(heading.y - years.y) < 2 && Math.abs(heading.x - note.x) < 2, 'Shared chart and paper columns');
    await chart.waitFor();
  } else {
    assert.ok(note.y >= citation.y + citation.height && Math.abs(note.x - citation.x) < 2, 'Takeaway below its citation');
    await chart.waitFor({ state: 'hidden' });
    await page.locator('.citation-growth summary').focus();
    await page.keyboard.press('Enter');
    await chart.waitFor();
    await page.keyboard.press('Enter');
    await chart.waitFor({ state: 'hidden' });
  }
  const toggle = entry.locator('summary');
  await toggle.focus();
  const toggleOffset = () => toggle.evaluate(el =>
    el.getBoundingClientRect().top - el.closest('.pubtrack__links').getBoundingClientRect().top);
  const top = await toggleOffset();
  await page.keyboard.press('Enter');
  const body = await entry.locator('.pubpanel__body').boundingBox();
  const row = await entry.boundingBox();
  assert.ok(body && Math.abs(body.width - row.width) < 2, 'Expanded abstract and figure span the paper row');
  assert.ok(await entry.locator('.pubpanel__abstract').isVisible());
  await entry.locator('.pubpanel__figure img').evaluate(img => img.decode());
  assert.ok(Math.abs(await toggleOffset() - top) < 2, 'Disclosure stays on the same line as its sibling actions');
  await page.keyboard.press('Enter');
}

async function main() {
  const url = new URL(process.argv[2] || 'http://127.0.0.1:4000');
  const browser = await chromium.launch({ channel: 'chrome' });
  try {
    const context = await browser.newContext({ reducedMotion: 'reduce' });
    await context.grantPermissions(['clipboard-read', 'clipboard-write'], { origin: url.origin });
    const page = await context.newPage();
    await page.goto(url.href);
    for (const width of [320, 360, 390, 430, 600, 601, 667, 768, 924, 925, 1024, 1280, 1440, 1730, 1920]) {
      await page.setViewportSize({ width, height: 800 });
      assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth),
        `Page overflows at ${width}px`);
      assert.equal(await page.locator('.profile_box a[aria-label="Email"]:visible, .author__urls a[href^="mailto:"]:visible').count(),
        1, `Contact links at ${width}px`);
    }
    await checkPaperLayout(page, 1440, true);
    await page.addScriptTag({ path: require.resolve('axe-core/axe.min.js') });
    await checkAccessibility(page);
    await checkPaperLayout(page, 390, false);
    await page.setViewportSize({ width: 320, height: 700 });
    assert.ok(await page.evaluate(async () => {
      const root = document.documentElement;
      root.style.fontSize = '200%';
      await new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)));
      const name = document.querySelector('.author__name');
      const fits = root.scrollWidth <= root.clientWidth && name.scrollWidth <= name.clientWidth;
      root.style.removeProperty('font-size');
      return fits;
    }), 'Narrow layout with enlarged text');
    await checkAccessibility(page);

    await page.locator('[data-filter="software"]').click();
    const year = page.locator('[data-year-group][hidden]').first();
    const id = await year.getAttribute('id');
    await page.locator(`.pubtrack__chart a[href="#${id}"]`).click();
    assert.ok(await page.locator('#' + id).isVisible());
    assert.equal(await page.locator('[data-filter="all"]').getAttribute('aria-pressed'), 'true');

    const entry = page.locator('.pubtrack__entry').first();
    for (const format of ['reference', 'bibtex']) {
      const attribute = 'data-copy-' + format;
      const button = entry.locator(`[${attribute}]`);
      await button.focus();
      await page.keyboard.press('Enter');
      await entry.locator(`[${attribute}].is-copied`).waitFor();
      assert.equal(await page.evaluate(() => navigator.clipboard.readText()), await button.getAttribute(attribute));
    }
    // A second click during the confirmation must restore the original label.
    const bib = entry.locator('[data-copy-bibtex]');
    await bib.click();
    await entry.locator('[data-copy-bibtex].is-copied').waitFor();
    await entry.locator('[data-copy-bibtex].is-copied').waitFor({ state: 'hidden' });
    assert.equal(await bib.innerText(), 'Copy BibTeX');

    await page.evaluate(() => Object.defineProperty(navigator, 'clipboard', {
      value: { writeText: () => Promise.reject(new Error('Clipboard denied')) },
    }));
    await bib.click();
    const manual = entry.getByRole('textbox');
    await manual.waitFor();
    const text = await bib.getAttribute('data-copy-bibtex');
    assert.equal(await manual.inputValue(), text);
    assert.deepEqual(await manual.evaluate(el => [document.activeElement === el, el.selectionStart, el.selectionEnd]),
      [true, 0, text.length]);
    await checkAccessibility(page);
    await page.setViewportSize({ width: 1440, height: 900 });
    const fallback = await entry.locator('.pubtrack__manual').boundingBox();
    const actions = await entry.locator('.pubtrack__links').boundingBox();
    assert.ok(fallback.y >= actions.y + actions.height && Math.abs(fallback.width - actions.width) < 2,
      'Manual copying stays with the selected paper across both columns');
    console.log('PASS: responsive publication layout, keyboard disclosures, accessibility scan, filtered year links, clipboard formats and manual selection.');
  } finally {
    await browser.close();
  }
}
main().catch(error => { console.error(error); process.exitCode = 1; });
