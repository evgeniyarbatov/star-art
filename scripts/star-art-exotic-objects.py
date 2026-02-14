import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytz
from skyfield.api import load, wgs84

from star_art_utils import StarArtUtils

IMAGES_DIR = "images"

EXOTIC_OBJECTS = [
    {"name": "Sagittarius A*", "ra": 17.761, "dec": -29.01, "mag": 17.0},
    {"name": "Cygnus X-1", "ra": 19.97, "dec": 35.2, "mag": 14.8},
    {"name": "M87*", "ra": 12.514, "dec": 12.39, "mag": 10.8},
    {"name": "Crab Pulsar", "ra": 5.575, "dec": 22.01, "mag": 16.5},
    {"name": "Vela Pulsar", "ra": 8.58, "dec": -45.18, "mag": 23.0},
    {"name": "Geminga", "ra": 6.57, "dec": 17.77, "mag": 25.0},
    {"name": "SGR 1806-20", "ra": 18.14, "dec": -20.4, "mag": 20.0},
    {"name": "3C 273", "ra": 12.485, "dec": 2.05, "mag": 12.9},
    {"name": "3C 48", "ra": 1.63, "dec": 33.16, "mag": 16.0},
    {"name": "3C 279", "ra": 12.93, "dec": -5.79, "mag": 17.8},
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
        f"\nGenerating 'sumi' exotic objects for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    objects = StarArtUtils.get_objects_by_ra_dec(
        observer, obs_time, EXOTIC_OBJECTS, altitude, azimuth, fov
    )
    if objects is None or objects.get("count", 0) == 0:
        print("No exotic objects visible in this FOV, skipping...")
        return

    print(f"Exotic objects in FOV: {objects['count']}")

    fig, bg_color = StarArtUtils.sumi_object_style(objects)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    details = (
        f"Exotic Objects {objects['count']}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    )
    StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    out_dir = f"{IMAGES_DIR}/sumi-exotic-objects"
    os.makedirs(out_dir, exist_ok=True)
    filename = (
        f"{out_dir}/{safe_name}_"
        f"exotic_objects_fov{fov}_az{azimuth}_alt{altitude}_{date_stamp}.png"
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
