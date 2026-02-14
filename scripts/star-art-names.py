import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytz
from skyfield.api import load, wgs84

from star_art_utils import StarArtUtils

IMAGES_DIR = "images"
STAR_NAMES_FILE = "data/star_names.csv"
HIPPARCOS_FILE = "hip_main.dat"

os.makedirs(IMAGES_DIR, exist_ok=True)


def create_artwork(location, named_stars, fov, azimuth, altitude):
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
        f"\nGenerating 'sumi' for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    stars = StarArtUtils.get_named_stars(
        observer, obs_time, named_stars, altitude, azimuth, fov, HIPPARCOS_FILE
    )
    if stars is None or stars.get("count", 0) == 0:
        print("No named stars visible in this FOV, skipping...")
        return

    print(f"Named stars in FOV: {stars['count']}")

    fig, bg_color = StarArtUtils.sumi_object_style(stars)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    details = (
        f"Stars {stars['count']}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    )
    StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    os.makedirs(f"{IMAGES_DIR}/sumi-stars", exist_ok=True)
    filename = (
        f"{IMAGES_DIR}/sumi-stars/{safe_name}_"
        f"fov{fov}_az{azimuth}_alt{altitude}_{date_stamp}.png"
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

    named_stars = StarArtUtils.load_named_stars(STAR_NAMES_FILE)

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
                    create_artwork(location, named_stars, fov, azimuth, altitude)

    print(f"\n✓ All artworks (attempted) saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
