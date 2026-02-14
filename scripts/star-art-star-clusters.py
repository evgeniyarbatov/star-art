import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import pytz
from skyfield.api import load, wgs84

from star_art_utils import StarArtUtils

IMAGES_DIR = "images"

STAR_CLUSTERS = [
    {"name": "Pleiades (M45)", "ra": 3.79, "dec": 24.12, "mag": 1.6},
    {"name": "Hyades", "ra": 4.45, "dec": 15.87, "mag": 0.5},
    {"name": "Beehive Cluster (M44)", "ra": 8.67, "dec": 19.67, "mag": 3.7},
    {"name": "Double Cluster (NGC 869/884)", "ra": 2.33, "dec": 57.13, "mag": 4.3},
    {"name": "Omega Centauri", "ra": 13.44, "dec": -47.48, "mag": 3.7},
    {"name": "47 Tucanae", "ra": 0.4, "dec": -72.08, "mag": 4.1},
    {"name": "Hercules Cluster (M13)", "ra": 16.7, "dec": 36.46, "mag": 5.8},
    {"name": "M3", "ra": 13.7, "dec": 28.38, "mag": 6.2},
    {"name": "M5", "ra": 15.3, "dec": 2.08, "mag": 5.7},
    {"name": "M92", "ra": 17.28, "dec": 43.14, "mag": 6.4},
    {"name": "M22", "ra": 18.61, "dec": -23.9, "mag": 5.1},
    {"name": "Butterfly Cluster (M6)", "ra": 17.67, "dec": -32.22, "mag": 4.2},
    {"name": "Ptolemy Cluster (M7)", "ra": 17.89, "dec": -34.82, "mag": 3.3},
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
        f"\nGenerating 'sumi' star clusters for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    objects = StarArtUtils.get_objects_by_ra_dec(
        observer, obs_time, STAR_CLUSTERS, altitude, azimuth, fov
    )
    if objects is None or objects.get("count", 0) == 0:
        print("No star clusters visible in this FOV, skipping...")
        return

    print(f"Star clusters in FOV: {objects['count']}")

    fig, bg_color = StarArtUtils.sumi_object_style(objects)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    details = (
        f"Star Clusters {objects['count']}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    )
    StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    out_dir = f"{IMAGES_DIR}/sumi-star-clusters"
    os.makedirs(out_dir, exist_ok=True)
    filename = (
        f"{out_dir}/{safe_name}_"
        f"star_clusters_fov{fov}_az{azimuth}_alt{altitude}_{date_stamp}.png"
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
