# Star Art

This project generates minimalist star-field artworks from real star positions.

## Code

- Loads locations from `stargazing-locations.json`.
- For each location and each registered style, computes astronomical dusk for the current UTC date and observes stars from that place and time.
- Uses the Hipparcos catalog (via Skyfield) and filters stars by a magnitude threshold.
- Projects the visible stars into a 2D stereographic view centered at azimuth 0 deg and altitude 90 deg with a 180 deg field of view.
- Renders each view with Matplotlib and saves a PNG to `images/<style>/...`.
- Adds a small footer line with location, time (localized when possible), and rendering parameters.

## Art

- The images are monochrome or near-monochrome; there are no labels, grids, or constellations.
- Star brightness is translated into point size (and sometimes opacity), creating a simple visual hierarchy.
- Each piece is framed by a thin circular boundary matching the projection radius.
- The palette stays restrained: light paper-like backgrounds with dark ink-like marks.
- Style variants are expressed through background tone, point size scaling, and opacity ranges:
  - `sumi`: ink-like strokes with wider opacity variation.