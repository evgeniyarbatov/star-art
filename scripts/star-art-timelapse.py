import json
import os
import time
from datetime import datetime, timedelta

import matplotlib.pyplot as plt
import pytz
from skyfield.api import load, wgs84

from star_art_utils import StarArtUtils

IMAGES_DIR = "images"
LOCATIONS_FILE = "stargazing-locations.json"

FRAME_MINUTES = 10
MAGNITUDE = 12.4
FOV = 180
AZIMUTH = 0
ALTITUDE = 90
LOCATION_NAME = None
LOCATION_INDEX = 0

os.makedirs(IMAGES_DIR, exist_ok=True)


def pick_location(locations, name, index):
    if name:
        for location in locations:
            if location.get("name") == name:
                return location
        return None
    if index < len(locations):
        return locations[index]
    return None


def generate_timelapse(location):
    start_time = time.time()

    planets = load("de421.bsp")
    earth = planets["earth"]

    lat = float(location["lat"])
    lon = float(location["lon"])
    name = location.get("name", f"{lat},{lon}")

    observer = earth + wgs84.latlon(lat, lon)

    tz = StarArtUtils.get_timezone(lat, lon)
    now_utc = datetime.now(pytz.UTC)
    local_date = now_utc.astimezone(tz).date() if tz else now_utc.date()

    dusk = StarArtUtils.get_astronomical_dusk(
        lat, lon, local_date, tzinfo=tz or pytz.UTC
    )
    sunrise = StarArtUtils.get_sunrise(lat, lon, local_date, tzinfo=tz or pytz.UTC)
    if sunrise <= dusk:
        sunrise = StarArtUtils.get_sunrise(
            lat, lon, local_date + timedelta(days=1), tzinfo=tz or pytz.UTC
        )

    total_minutes = max(0, int((sunrise - dusk).total_seconds() / 60))
    total_frames = total_minutes // FRAME_MINUTES + 1
    safe_name = name.replace(" ", "_").replace(",", "_")
    out_dir = f"{IMAGES_DIR}/timelapse/{safe_name}"
    os.makedirs(out_dir, exist_ok=True)

    print(
        f"\nGenerating timelapse for {name} starting at astronomical dusk (UTC): {dusk.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    for idx in range(total_frames):
        obs_time = dusk + timedelta(minutes=FRAME_MINUTES * idx)
        stars = StarArtUtils.get_visible_stars(
            observer, obs_time, MAGNITUDE, ALTITUDE, AZIMUTH, FOV
        )

        if stars is None or stars.get("count", 0) == 0:
            print(f"Frame {idx + 1}/{total_frames}: no stars visible, skipping...")
            continue

        fig, bg_color = StarArtUtils.sumi_star_style(stars, FOV)
        if fig is None:
            print(f"Frame {idx + 1}/{total_frames}: failed to render, skipping...")
            continue

        details = f"Mag ≤{MAGNITUDE}  |  FOV {FOV}°  |  Az {AZIMUTH}°  Alt {ALTITUDE}°"
        StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

        filename = f"{out_dir}/frame_{idx + 1:04d}.png"
        fig.tight_layout(pad=0.5)
        plt.savefig(
            filename, dpi=300, facecolor=bg_color, edgecolor="none", bbox_inches="tight"
        )
        plt.close(fig)
        print(f"✓ Saved frame {idx + 1}/{total_frames}: {filename}")

    duration = time.time() - start_time
    print(f"\n✓ Timelapse frames saved to {out_dir} ({duration:.2f}s)")


def main(locations_file=LOCATIONS_FILE):
    with open(locations_file, "r") as f:
        locations = json.load(f)

    location = pick_location(locations, LOCATION_NAME, LOCATION_INDEX)
    if location is None:
        print("No matching location found.")
        return

    generate_timelapse(location)


if __name__ == "__main__":
    main()
