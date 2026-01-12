import csv
import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time as dtime
import pytz
from skyfield.api import load, wgs84, Star
from skyfield.data import hipparcos
from astral import Observer
from astral.sun import sun
from timezonefinder import TimezoneFinder

IMAGES_DIR = "images"
STAR_NAMES_FILE = "data/star_names.csv"
HIPPARCOS_FILE = "hip_main.dat"
TF = TimezoneFinder()

os.makedirs(IMAGES_DIR, exist_ok=True)

_HIPPARCOS_DF = None


def add_info_text(fig, location, obs_time, star_count, fov, azimuth, altitude, bg_color):
    def get_luminance(color):
        if color.startswith("#"):
            hex_color = color.lstrip("#")
            if len(hex_color) == 6:
                r, g, b = (
                    int(hex_color[0:2], 16),
                    int(hex_color[2:4], 16),
                    int(hex_color[4:6], 16),
                )
            elif len(hex_color) == 3:
                r, g, b = (
                    int(hex_color[0] * 2, 16),
                    int(hex_color[1] * 2, 16),
                    int(hex_color[2] * 2, 16),
                )
            else:
                return 255
        elif color.lower() == "white":
            r, g, b = 255, 255, 255
        elif color.lower() == "black":
            r, g, b = 0, 0, 0
        else:
            return 200
        return 0.299 * r + 0.587 * g + 0.114 * b

    luminance = get_luminance(bg_color)
    text_color = "black" if luminance > 128 else "white"

    name = (
        location.get("name", "Unknown") if isinstance(location, dict) else str(location)
    )
    lat = location.get("lat", 0.0)
    lon = location.get("lon", 0.0)

    tz_name = TF.timezone_at(lat=lat, lng=lon)
    if tz_name:
        tz = pytz.timezone(tz_name)
        local_time = obs_time.astimezone(tz)
        time_str = local_time.strftime("%Y-%m-%d %H:%M %Z")
    else:
        time_str = obs_time.strftime("%Y-%m-%d %H:%M UTC")

    info_text = (
        f"{name}  |  {lat:.2f}°, {lon:.2f}°  |  "
        f"{time_str}  |  "
        f"Stars {star_count}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    )

    fig.text(
        0.5,
        -0.02,
        info_text,
        ha="center",
        fontsize=9,
        family="monospace",
        weight="normal",
        color=text_color,
    )


def stereographic_project(alt, az, center_alt, center_az, fov):
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)

    alt_rad = np.radians(alt)
    az_rad = np.radians(az)
    center_alt_rad = np.radians(float(center_alt))
    center_az_rad = np.radians(float(center_az))

    x1 = np.cos(alt_rad) * np.sin(az_rad)
    y1 = np.cos(alt_rad) * np.cos(az_rad)
    z1 = np.sin(alt_rad)

    x_c = np.cos(center_alt_rad) * np.sin(center_az_rad)
    y_c = np.cos(center_alt_rad) * np.cos(center_az_rad)
    z_c = np.sin(center_alt_rad)

    cos_dist = x1 * x_c + y1 * y_c + z1 * z_c
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    angular_dist = np.degrees(np.arccos(cos_dist))

    mask = angular_dist <= (float(fov) / 2.0)

    if not np.any(mask):
        return None, None, mask

    denom = 1.0 + cos_dist
    denom[denom == 0] = 1e-12
    k = 2.0 / denom

    dx = x1 - x_c
    dy = y1 - y_c

    x = k * dx
    y = k * dy

    return x, y, mask


def get_astronomical_dusk(lat, lon, date):
    try:
        observer = Observer(latitude=float(lat), longitude=float(lon), elevation=0)
        s = sun(observer, date=date, tzinfo=pytz.UTC)
        if "dusk" in s and s["dusk"] is not None:
            return s["dusk"]
        if "sunset" in s and s["sunset"] is not None:
            return s["sunset"] + timedelta(hours=2)
    except Exception as e:
        print(f"Could not calculate dusk using astral: {e}")

    try:
        return datetime.combine(date, dtime(hour=20, minute=0), tzinfo=pytz.UTC)
    except Exception:
        return datetime.now(pytz.UTC)


def load_named_stars(csv_path):
    stars = []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stars.append({"name": row["name"].strip(), "hip_id": int(row["hip_id"])})
    return stars


def _load_hipparcos():
    global _HIPPARCOS_DF
    if _HIPPARCOS_DF is not None:
        return _HIPPARCOS_DF

    try:
        with open(HIPPARCOS_FILE, "rb") as f:
            _HIPPARCOS_DF = hipparcos.load_dataframe(f)
    except Exception as e:
        print(f"Failed to load Hipparcos catalog from '{HIPPARCOS_FILE}': {e}")
        _HIPPARCOS_DF = None

    return _HIPPARCOS_DF


def get_named_stars(observer, obs_time, named_stars, center_alt, center_az, fov):
    df = _load_hipparcos()
    if df is None or len(df) == 0:
        print("Hipparcos catalog not available.")
        return None

    hip_ids = [row["hip_id"] for row in named_stars]
    available = df.loc[df.index.intersection(hip_ids)]
    if len(available) == 0:
        return None

    ordered = [row for row in named_stars if row["hip_id"] in available.index]
    df_ordered = available.loc[[row["hip_id"] for row in ordered]]

    stars = Star.from_dataframe(df_ordered)
    ts = load.timescale(builtin=True)
    if obs_time.tzinfo is None:
        obs_time = obs_time.replace(tzinfo=pytz.UTC)
    t = ts.from_datetime(obs_time)

    alt, az, _ = observer.at(t).observe(stars).apparent().altaz()
    x, y, mask = stereographic_project(
        alt.degrees, az.degrees, center_alt, center_az, fov
    )
    if x is None:
        return None

    mags = np.asarray(df_ordered["magnitude"].values)
    names = np.asarray([row["name"] for row in ordered])

    return {
        "x": x[mask],
        "y": y[mask],
        "mag": mags[mask],
        "name": names[mask],
        "count": int(np.sum(mask)),
    }


def sumi_style(stars):
    if stars is None or stars.get("count", 0) == 0:
        return None, "white"

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#fdfdf9", dpi=300)
    ax.set_facecolor("#fdfdf9")
    ax.set_aspect("equal")

    radii = np.sqrt(stars["x"] ** 2 + stars["y"] ** 2)
    r_max = np.max(radii) * 1.0

    sizes = 40 * np.exp(-stars["mag"] / 2.2)
    alphas = np.clip(0.9 - (stars["mag"] - np.min(stars["mag"])) / 12, 0.3, 0.9)

    ax.scatter(stars["x"], stars["y"], s=sizes, c="#1a1a1a", alpha=alphas, linewidths=0)

    label_offset = 0.02 * r_max
    for x, y, name in zip(stars["x"], stars["y"], stars["name"]):
        ax.text(
            x + label_offset,
            y + label_offset,
            name,
            fontsize=7,
            family="monospace",
            color="#1a1a1a",
            ha="left",
            va="bottom",
        )

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.axis("off")

    circle = plt.Circle((0, 0), r_max, color="#1a1a1a", fill=False, linewidth=0.25)
    ax.add_patch(circle)

    return fig, "#fdfdf9"


def create_artwork(location, named_stars, fov, azimuth, altitude):
    start_time = time.time()

    planets = load("de421.bsp")
    earth = planets["earth"]

    lat = float(location["lat"])
    lon = float(location["lon"])
    name = location.get("name", f"{lat},{lon}")

    observer = earth + wgs84.latlon(lat, lon)

    today = datetime.now(pytz.UTC).date()
    obs_time = get_astronomical_dusk(lat, lon, today)

    print(
        f"\nGenerating 'sumi' for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    stars = get_named_stars(observer, obs_time, named_stars, altitude, azimuth, fov)
    if stars is None or stars.get("count", 0) == 0:
        print("No named stars visible in this FOV, skipping...")
        return

    print(f"Named stars in FOV: {stars['count']}")

    fig, bg_color = sumi_style(stars)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    add_info_text(fig, location, obs_time, stars["count"], fov, azimuth, altitude, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    os.makedirs(f"{IMAGES_DIR}/sumi", exist_ok=True)
    filename = (
        f"{IMAGES_DIR}/sumi/{safe_name}_"
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

    named_stars = load_named_stars(STAR_NAMES_FILE)

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
