# Star Art

This project generates minimalist star-field artworks from real star positions.

## Code

- Loads locations from `stargazing-locations.json`.
- For each location and each registered style, computes astronomical dusk for the current UTC date and observes stars from that place and time.
- Uses the Hipparcos catalog (via Skyfield) and filters stars by a magnitude threshold.
- Projects the visible stars into a 2D stereographic view centered at azimuth 0 deg and altitude 90 deg with a 180 deg field of view.
- Renders each view with Matplotlib and saves a PNG to `images/<style>/...`.
- Adds a small footer line with location, time (localized when possible), and rendering parameters.

## Makefile

`make art` generates the base star art
`make stars` generates named star renders
`make galaxies` generates galaxy renders
`make planets` generates planet renders
`make nebulae` generates nebula renders
`make clusters` generates star cluster renders
`make exotic` generates exotic object renders
`make path` generates the wabi-sabi path renders
`make timelapse` generates timelapse frames for a single location