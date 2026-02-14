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
STAR_NAMES_FILE = "data/star_names.csv"
HIPPARCOS_FILE = "hip_main.dat"

os.makedirs(IMAGES_DIR, exist_ok=True)


def _path_length(points, order):
    p = points[order]
    d = p[1:] - p[:-1]
    return float(np.sum(np.hypot(d[:, 0], d[:, 1])))


def _two_opt(points, order, max_passes=6):
    n = len(order)
    if n < 4:
        return order

    best = order.copy()
    best_len = _path_length(points, best)

    for _ in range(max_passes):
        improved = False
        for i in range(1, n - 2):
            for j in range(i + 1, n - 1):
                cand = best.copy()
                cand[i : j + 1] = cand[i : j + 1][::-1]
                cand_len = _path_length(points, cand)
                if cand_len + 1e-12 < best_len:
                    best = cand
                    best_len = cand_len
                    improved = True
        if not improved:
            break
    return best


def compute_shortest_visit_order(x, y, mag=None):
    points = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    n = len(points)
    if n <= 1:
        return np.arange(n, dtype=int)

    if mag is not None and len(mag) == n:
        start = int(np.argmin(mag))
    else:
        start = 0

    remaining = set(range(n))
    remaining.remove(start)
    order = [start]
    cur = start

    while remaining:
        rem = np.array(sorted(remaining), dtype=int)
        dx = points[rem, 0] - points[cur, 0]
        dy = points[rem, 1] - points[cur, 1]
        j = int(rem[np.argmin(dx * dx + dy * dy)])
        order.append(j)
        remaining.remove(j)
        cur = j

    order = np.array(order, dtype=int)
    order = _two_opt(points, order, max_passes=7)
    return order


def wabi_sabi_minimal_style(stars):
    if stars is None or stars.get("count", 0) == 0:
        return None, "white"

    bg = "#f2efe6"
    ink = "#141414"
    path_ink = "#2a2a2a"
    label_ink = "#1f1f1f"

    fig, ax = plt.subplots(figsize=(12, 12), facecolor=bg, dpi=300)
    ax.set_facecolor(bg)
    ax.set_aspect("equal")

    radii = np.sqrt(stars["x"] ** 2 + stars["y"] ** 2)
    r_max = np.max(radii) * 1.02

    sizes = 32 * np.exp(-stars["mag"] / 2.4)
    sizes = np.clip(sizes, 5.0, 42.0)

    alphas = np.clip(0.85 - (stars["mag"] - np.min(stars["mag"])) / 13, 0.25, 0.85)
    alphas = np.maximum(alphas, 0.45)

    if stars["count"] >= 2:
        order = compute_shortest_visit_order(stars["x"], stars["y"], stars["mag"])
        ax.plot(
            stars["x"][order],
            stars["y"][order],
            color=path_ink,
            linewidth=0.55,
            alpha=0.55,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=1,
        )

    ax.scatter(
        stars["x"],
        stars["y"],
        s=sizes,
        c=ink,
        alpha=alphas,
        linewidths=0,
        zorder=2,
    )

    StarArtUtils.place_labels(ax, stars, color=label_ink)

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.axis("off")

    circle = plt.Circle((0, 0), r_max, color=ink, fill=False, linewidth=0.18, alpha=0.35)
    ax.add_patch(circle)

    return fig, bg


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
        f"\nGenerating 'wabi-sabi minimal' for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}"
    )

    stars = StarArtUtils.get_named_stars(
        observer, obs_time, named_stars, altitude, azimuth, fov, HIPPARCOS_FILE
    )
    if stars is None or stars.get("count", 0) == 0:
        print("No named stars visible in this FOV, skipping...")
        return

    print(f"Named stars in FOV: {stars['count']}")

    fig, bg_color = wabi_sabi_minimal_style(stars)
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    details = f"Stars {stars['count']}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°"
    StarArtUtils.add_info_text(fig, location, obs_time, details, bg_color)

    date_stamp = obs_time.strftime("%Y%m%d")
    safe_name = name.replace(" ", "_").replace(",", "_")
    os.makedirs(f"{IMAGES_DIR}/wabi-sabi-stars", exist_ok=True)
    filename = (
        f"{IMAGES_DIR}/wabi-sabi-stars/{safe_name}_"
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
