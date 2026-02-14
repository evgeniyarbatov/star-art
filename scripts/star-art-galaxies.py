import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytz
from skyfield.api import load, wgs84

from star_art_utils import StarArtUtils

IMAGES_DIR = "images"

GALAXIES = [
    {"name": "Andromeda Galaxy", "ra": 0.712306, "dec": 41.269167, "mag": 3.4},
    {"name": "Triangulum Galaxy", "ra": 1.564139, "dec": 30.66, "mag": 5.7},
    {"name": "Whirlpool Galaxy", "ra": 13.497972, "dec": 47.195278, "mag": 8.4},
    {"name": "Sombrero Galaxy", "ra": 12.6665, "dec": -11.623056, "mag": 8.0},
    {"name": "Pinwheel Galaxy", "ra": 14.0535, "dec": 54.349167, "mag": 7.9},
    {"name": "Bode's Galaxy", "ra": 9.925889, "dec": 69.065278, "mag": 6.9},
    {"name": "Cigar Galaxy", "ra": 9.931167, "dec": 69.679722, "mag": 8.4},
    {"name": "Centaurus A", "ra": 13.424333, "dec": -43.019167, "mag": 6.8},
    {"name": "Sculptor Galaxy", "ra": 0.792528, "dec": -25.288333, "mag": 7.1},
    {"name": "Large Magellanic Cloud", "ra": 5.392944, "dec": -69.756111, "mag": 0.9},
    {"name": "Small Magellanic Cloud", "ra": 0.879111, "dec": -72.828611, "mag": 2.7},
    {"name": "Virgo A (M87)", "ra": 12.513722, "dec": 12.391111, "mag": 8.6},
]

os.makedirs(IMAGES_DIR, exist_ok=True)


def create_artwork(location, fov, azimuth, altitude):
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
        f"\nGenerating 'sumi' galaxies for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    objects = StarArtUtils.get_objects_by_ra_dec(
        observer, obs_time, GALAXIES, altitude, azimuth, fov
    )
    if objects is None or objects.get("count", 0) == 0:
        print("No galaxies visible in this FOV, skipping...")
        return

    print(f"Galaxies in FOV: {objects['count']}")

    fig, bg_color = StarArtUtils.sumi_object_style(objects)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    details = f"Galaxies {objects['count']}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    out_dir = f"{IMAGES_DIR}/sumi-galaxies"
    os.makedirs(out_dir, exist_ok=True)
    filename = (
        f"{out_dir}/{safe_name}_"
        f"galaxies_fov{fov}_az{azimuth}_alt{altitude}_{date_stamp}.png"
    )

    fig.tight_layout(pad=0.5)
    plt.savefig(
        filename, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight"
    )
    plt.close(fig)
    duration = time.time() - start_time
    print(f"✓ Saved: {filename} ({duration:.2f}s)")


def main(locations_file="stargazing-locations.json"):
    with open(locations_file, "r") as f:
        locations = json.load(f)

    fovs = [180]
    azimuths = [0]
    altitudes = [90]

    total = len(locations) * len(fovs) * len(azimuths) * len(altitudes)
    current = 0

    for location in locations:
        for fov in fovs:
            for azimuth in azimuths:
                for altitude in altitudes:
                    current += 1
                    print(f"\n[{current}/{total}]", end=" ")
                    create_artwork(location, fov, azimuth, altitude)

    print(f"\n✓ All artworks (attempted) saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
