import json
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time as dtime
import pytz
from matplotlib.transforms import Bbox
from skyfield.api import load, wgs84
from astral import Observer
from astral.sun import sun
from timezonefinder import TimezoneFinder

IMAGES_DIR = "images"
TF = TimezoneFinder()

BODIES = [
    {"name": "Mercury", "key": "mercury", "mag": 2.0},
    {"name": "Venus", "key": "venus", "mag": 0.5},
    {"name": "Mars", "key": "mars", "mag": 1.2},
    {"name": "Jupiter", "key": "jupiter barycenter", "mag": 0.6},
    {"name": "Saturn", "key": "saturn barycenter", "mag": 1.6},
    {"name": "Uranus", "key": "uranus barycenter", "mag": 3.5},
    {"name": "Neptune", "key": "neptune barycenter", "mag": 4.2},
    {"name": "Pluto", "key": "pluto barycenter", "mag": 6.0},
    {"name": "Moon", "key": "moon", "mag": 0.4},
]

os.makedirs(IMAGES_DIR, exist_ok=True)


def add_info_text(fig, location, obs_time, count, fov, azimuth, altitude, bg_color):
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
        f"Bodies {count}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
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


def _place_labels(ax, objects):
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    inv = ax.transData.inverted()

    obj_xy = np.column_stack([objects["x"], objects["y"]])
    obj_disp = ax.transData.transform(obj_xy)
    max_obj_x = np.max(obj_disp[:, 0])
    max_obj_y = np.max(obj_disp[:, 1])

    order = np.argsort(objects["mag"])
    placed_bboxes = []

    font = {
        "fontsize": 7,
        "family": "monospace",
        "color": "#1a1a1a",
        "ha": "left",
        "va": "bottom",
        "clip_on": False,
    }

    max_label_width = 0.0
    for idx in order:
        temp = ax.text(0, 0, objects["name"][idx], alpha=0, **font)
        bbox = temp.get_window_extent(renderer)
        temp.remove()
        max_label_width = max(max_label_width, bbox.width)

    fallback_x = max_obj_x + max_label_width * 3 + 20
    fallback_y = max_obj_y + 20
    fallback_gap = 6

    for idx in order:
        name = objects["name"][idx]
        temp = ax.text(0, 0, name, alpha=0, **font)
        bbox = temp.get_window_extent(renderer)
        temp.remove()
        w, h = bbox.width, bbox.height

        sx, sy = obj_disp[idx]
        min_offset = max(10, 0.6 * max(w, h))
        step = 6
        found = False

        for ring in range(80):
            r = min_offset + ring * step
            points = 12 + ring // 3
            for k in range(points):
                angle = 2 * np.pi * k / points
                cx = sx + np.cos(angle) * r
                cy = sy + np.sin(angle) * r
                cand = Bbox.from_bounds(cx, cy, w, h)

                if any(cand.overlaps(b) for b in placed_bboxes):
                    continue
                pad = 2.0
                if np.any(
                    (obj_disp[:, 0] >= cand.x0 - pad)
                    & (obj_disp[:, 0] <= cand.x1 + pad)
                    & (obj_disp[:, 1] >= cand.y0 - pad)
                    & (obj_disp[:, 1] <= cand.y1 + pad)
                ):
                    continue

                data_x, data_y = inv.transform((cx, cy))
                ax.text(data_x, data_y, name, **font)
                placed_bboxes.append(cand)
                found = True
                break
            if found:
                break

        if not found:
            while True:
                cx = fallback_x
                cy = fallback_y
                cand = Bbox.from_bounds(cx, cy, w, h)
                if not any(cand.overlaps(b) for b in placed_bboxes):
                    data_x, data_y = inv.transform((cx, cy))
                    ax.text(data_x, data_y, name, **font)
                    placed_bboxes.append(cand)
                    fallback_y -= h + fallback_gap
                    break
                fallback_y -= h + fallback_gap


def sumi_style(objects):
    if objects is None or objects.get("count", 0) == 0:
        return None, "white"

    fig, ax = plt.subplots(figsize=(12, 12), facecolor="#fdfdf9", dpi=300)
    ax.set_facecolor("#fdfdf9")
    ax.set_aspect("equal")

    radii = np.sqrt(objects["x"] ** 2 + objects["y"] ** 2)
    r_max = np.max(radii) * 1.0

    sizes = 40 * np.exp(-objects["mag"] / 2.2)
    sizes = np.maximum(sizes, 6.0)
    alphas = np.clip(0.9 - (objects["mag"] - np.min(objects["mag"])) / 12, 0.3, 0.9)
    alphas = np.maximum(alphas, 0.55)

    ax.scatter(objects["x"], objects["y"], s=sizes, c="#1a1a1a", alpha=alphas, linewidths=0)

    _place_labels(ax, objects)

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.axis("off")

    circle = plt.Circle((0, 0), r_max, color="#1a1a1a", fill=False, linewidth=0.25)
    ax.add_patch(circle)

    return fig, "#fdfdf9"


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

    x, y, mask = stereographic_project(alt_list, az_list, center_alt, center_az, fov)
    if x is None:
        return None

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
    obs_time = get_astronomical_dusk(lat, lon, today)

    print(
        f"\nGenerating 'sumi' planets for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    objects = get_bodies(observer, planets, obs_time, altitude, azimuth, fov)
    if objects is None or objects.get("count", 0) == 0:
        print("No bodies visible in this FOV, skipping...")
        return

    print(f"Bodies in FOV: {objects['count']}")

    fig, bg_color = sumi_style(objects)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    add_info_text(fig, location, obs_time, objects["count"], fov, azimuth, altitude, bg_color)

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
