import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytz
from skyfield.api import load, wgs84

from star_art_utils import StarArtUtils

IMAGES_DIR = "images"

NEBULAE = [
    {"name": "Orion Nebula (M42)", "ra": 5.591, "dec": -5.45, "mag": 4.0},
    {"name": "Lagoon Nebula (M8)", "ra": 18.066, "dec": -24.39, "mag": 6.0},
    {"name": "Eagle Nebula (M16)", "ra": 18.313, "dec": -13.78, "mag": 6.0},
    {"name": "Omega Nebula (M17)", "ra": 18.346, "dec": -16.17, "mag": 6.0},
    {"name": "Trifid Nebula (M20)", "ra": 18.046, "dec": -23.03, "mag": 6.3},
    {"name": "North America Nebula", "ra": 20.97, "dec": 44.3, "mag": 4.0},
    {"name": "Rosette Nebula", "ra": 6.33, "dec": 4.9, "mag": 9.0},
    {"name": "Crab Nebula (M1)", "ra": 5.575, "dec": 22.01, "mag": 8.4},
    {"name": "Helix Nebula (NGC 7293)", "ra": 22.49, "dec": -20.84, "mag": 7.6},
    {"name": "Ring Nebula (M57)", "ra": 18.893, "dec": 33.03, "mag": 8.8},
    {"name": "Dumbbell Nebula (M27)", "ra": 19.99, "dec": 22.72, "mag": 7.5},
    {"name": "Carina Nebula (NGC 3372)", "ra": 10.75, "dec": -59.68, "mag": 1.0},
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
        f"\nGenerating 'sumi' nebulae for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    objects = StarArtUtils.get_objects_by_ra_dec(
        observer, obs_time, NEBULAE, altitude, azimuth, fov
    )
    if objects is None or objects.get("count", 0) == 0:
        print("No nebulae visible in this FOV, skipping...")
        return

    print(f"Nebulae in FOV: {objects['count']}")

    fig, bg_color = StarArtUtils.sumi_object_style(objects)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    details = f"Nebulae {objects['count']}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    out_dir = f"{IMAGES_DIR}/sumi-nebulae"
    os.makedirs(out_dir, exist_ok=True)
    filename = (
        f"{out_dir}/{safe_name}_"
        f"nebulae_fov{fov}_az{azimuth}_alt{altitude}_{date_stamp}.png"
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
