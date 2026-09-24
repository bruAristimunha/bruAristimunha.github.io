# Library logo sources

- [Braindecode](https://braindecode.org/stable/_static/braindecode.svg)
- [MOABB](https://moabb.neurotechx.com/docs/_static/moabb_notext.svg)
- [EEG-DaSh](https://raw.githubusercontent.com/eegdash/EEGDash/main/docs/source/_static/eegdash_image_only.svg)
- [SPD Learn](https://raw.githubusercontent.com/spdlearn/spd_learn/main/docs/source/_static/spd_learn.png)

The site animates the original library marks. The SVG includes in
`_includes/library-logos/` preserve the original visible geometry and colours,
with prefixed IDs and classes for signal, circuit and light effects. SPD Learn
uses its unchanged official PNG with a subtle surface highlight.

`_sass/_library-logos.scss` defines motion; `assets/js/library-logos.js` handles
pause, offscreen suspension and reduced motion. Run
`node scripts/check-library-logos.cjs` to check the controls and SVG/image readiness.
