import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytz
from skyfield.api import load, wgs84

from star_art_utils import StarArtUtils

STYLES = {}
IMAGES_DIR = "images"

os.makedirs(IMAGES_DIR, exist_ok=True)


def style(name):
    """Decorator to register a style function"""

    def decorator(func):
        STYLES[name] = func
        return func

    return decorator


@style("sumi")
def sumi_style(stars, fov):
    return StarArtUtils.sumi_star_style(stars, fov)


def create_artwork(location, style_name, magnitude, fov, azimuth, altitude):
    """Create star map artwork for a single location and parameter set."""
    start_time = time.time()

    planets = load("de421.bsp")
    earth = planets["earth"]

    lat = float(location["lat"])
    lon = float(location["lon"])
    name = location.get("name", f"{lat},{lon}")

    observer = earth + wgs84.latlon(lat, lon)

    today = datetime.now(pytz.UTC).date()
    obs_time = StarArtUtils.get_astronomical_dusk(lat, lon, today)

    print(
        f"\nGenerating '{style_name}' for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    stars = StarArtUtils.get_visible_stars(
        observer, obs_time, magnitude, altitude, azimuth, fov
    )

    if stars is None or stars.get("count", 0) == 0:
        print("No stars visible in this FOV, skipping...")
        return

    print(f"Stars in FOV: {stars['count']}")

    if style_name not in STYLES:
        print(f"Error: Style '{style_name}' not found")
        return

    fig_bg = STYLES[style_name](stars, fov)
    if fig_bg is None:
        print("Failed to generate artwork (style returned None).")
        return

    fig, bg_color = fig_bg
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    details = f"Mag ≤{magnitude}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    os.makedirs(f"{IMAGES_DIR}/{style_name}", exist_ok=True)
    filename = (
        f"{IMAGES_DIR}/{style_name}/{safe_name}_"
        f"mag{magnitude}_fov{fov}_az{azimuth}_alt{altitude}_{date_stamp}.png"
    )

    try:
        fig.tight_layout(pad=0.5)
        plt.savefig(
            filename, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight"
        )
        duration = time.time() - start_time
        print(f"✓ Saved: {filename} ({duration:.2f}s)")
    except Exception as e:
        print(f"Error saving figure: {e}")
    finally:
        plt.close(fig)


def main(locations_file="stargazing-locations.json"):
    try:
        with open(locations_file, "r") as f:
            locations = json.load(f)
    except Exception as e:
        print(f"Could not load locations file '{locations_file}': {e}")
        return

    print(f"Available styles: {', '.join(sorted(STYLES.keys()))}")
    print(f"\nGenerating artworks for {len(locations)} locations...")

    fovs = [180]
    azimuths = [0]
    altitudes = [90]
    magnitudes = [12.4]

    total = (
        len(locations)
        * len(STYLES)
        * len(fovs)
        * len(azimuths)
        * len(altitudes)
        * len(magnitudes)
    )
    current = 0

    for location in locations:
        for style_name in sorted(STYLES.keys()):
            for fov in fovs:
                for azimuth in azimuths:
                    for altitude in altitudes:
                        for magnitude in magnitudes:
                            current += 1
                            print(f"\n[{current}/{total}]", end=" ")
                            try:
                                create_artwork(
                                    location,
                                    style_name,
                                    magnitude,
                                    fov,
                                    azimuth,
                                    altitude,
                                )
                            except Exception as e:
                                print(f"Error: {e}")

    print(f"\n✓ All artworks (attempted) saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
