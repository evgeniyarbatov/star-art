import json
import os
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pytz
from skyfield.api import load, wgs84

from star_art_utils import StarArtUtils

IMAGES_DIR = "images"

BODIES = [
    {"name": "Mercury", "key": "mercury", "mag": -0.9},
    {"name": "Venus", "key": "venus", "mag": -3.9},
    {"name": "Mars", "key": "mars", "mag": 1.0},
    {"name": "Jupiter", "key": "jupiter barycenter", "mag": -2.7},
    {"name": "Saturn", "key": "saturn barycenter", "mag": 1.0},
    {"name": "Uranus", "key": "uranus barycenter", "mag": 5.7},
    {"name": "Neptune", "key": "neptune barycenter", "mag": 7.8},
    {"name": "Pluto", "key": "pluto barycenter", "mag": 14.6},
]

os.makedirs(IMAGES_DIR, exist_ok=True)


def get_bodies(observer, planets, obs_time, center_alt, center_az, fov):
    alt_list = []
    az_list = []
    mags = []
    names = []

    ts = load.timescale(builtin=True)
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=pytz.UTC)
    t = ts.from_datetime(obs_time)

    for body in BODIES:
        target = planets[body["key"]]
        alt, az, _ = observer.at(t).observe(target).apparent().altaz()
        alt_list.append(alt.degrees)
        az_list.append(az.degrees)
        mags.append(body["mag"])
        names.append(body["name"])

    projected = StarArtUtils.stereographic_project(
        alt_list, az_list, center_alt, center_az, fov
    )
    if projected[0] is None:
        return None

    x, y, mask = projected
    mags = np.asarray(mags)
    names = np.asarray(names)

    return {
        "x": x[mask],
        "y": y[mask],
        "mag": mags[mask],
        "name": names[mask],
        "count": int(np.sum(mask)),
    }


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
        f"\nGenerating 'sumi' planets for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    objects = get_bodies(observer, planets, obs_time, altitude, azimuth, fov)
    if objects is None or objects.get("count", 0) == 0:
        print("No bodies visible in this FOV, skipping...")
        return

    print(f"Bodies in FOV: {objects['count']}")

    fig, bg_color = StarArtUtils.sumi_object_style(objects)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    details = f"Bodies {objects['count']}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    out_dir = f"{IMAGES_DIR}/sumi-planets"
    os.makedirs(out_dir, exist_ok=True)
    filename = (
        f"{out_dir}/{safe_name}_"
        f"planets_fov{fov}_az{azimuth}_alt{altitude}_{date_stamp}.png"
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
