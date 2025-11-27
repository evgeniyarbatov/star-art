import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta, time as dtime
import pytz
from skyfield.api import load, wgs84, Star
from skyfield.data import hipparcos
from astral import Observer
from astral.sun import sun
from timezonefinder import TimezoneFinder

# Global registry for styles
STYLES = {}
IMAGES_DIR = "images"
TF = TimezoneFinder()

# Create output directory
os.makedirs(IMAGES_DIR, exist_ok=True)

# Load heavy catalogs once
_HIPPARCOS_DF = None


def style(name):
    """Decorator to register a style function"""
    def decorator(func):
        STYLES[name] = func
        return func
    return decorator


def add_info_text(fig, ax, location, obs_time, magnitude, fov, azimuth, altitude, bg_color):
    """Add information text with automatic light/dark contrast."""

    # Determine readable text color
    # simple luminance check: white = 255, black = 0
    luminance = 255 if bg_color.lower() == "white" else 0
    text_color = "black" if luminance > 128 else "white"

    # Process location/time
    name = location.get('name', 'Unknown') if isinstance(location, dict) else str(location)
    lat = location.get('lat', 0.0)
    lon = location.get('lon', 0.0)

    tz_name = TF.timezone_at(lat=lat, lng=lon)
    if tz_name:
        try:
            tz = pytz.timezone(tz_name)
            local_time = obs_time.astimezone(tz)
            time_str = local_time.strftime('%Y-%m-%d %H:%M %Z')
        except Exception:
            time_str = obs_time.strftime('%Y-%m-%d %H:%M UTC')
    else:
        time_str = obs_time.strftime('%Y-%m-%d %H:%M UTC')

    info_text = (f"{name}  |  {lat:.2f}°, {lon:.2f}°  |  "
                 f"{time_str}  |  "
                 f"Mag ≤{magnitude}  |  FOV {fov}°  |  Az {azimuth}°  Alt {altitude}°")

    fig.text(
        0.5, 0.02, info_text,
        ha='center', fontsize=9,
        family='monospace', weight='normal',
        color=text_color
    )


def stereographic_project(alt, az, center_alt, center_az, fov):
    """Project arrays of alt (deg) and az (deg) to 2D using stereographic projection

    Returns (x, y, mask) where x,y are numpy arrays and mask is a boolean array
    indicating which points lie within the FOV (angular distance <= fov/2).
    If no points in FOV returns (None, None, mask).
    """
    # Convert inputs to numpy arrays
    alt = np.asarray(alt, dtype=float)
    az = np.asarray(az, dtype=float)

    # Convert degrees to radians
    alt_rad = np.radians(alt)
    az_rad = np.radians(az)
    center_alt_rad = np.radians(float(center_alt))
    center_az_rad = np.radians(float(center_az))

    # Convert to Cartesian coordinates on unit sphere
    x1 = np.cos(alt_rad) * np.sin(az_rad)
    y1 = np.cos(alt_rad) * np.cos(az_rad)
    z1 = np.sin(alt_rad)

    x_c = np.cos(center_alt_rad) * np.sin(center_az_rad)
    y_c = np.cos(center_alt_rad) * np.cos(center_az_rad)
    z_c = np.sin(center_alt_rad)

    # Dot product = cos(angular distance)
    cos_dist = x1 * x_c + y1 * y_c + z1 * z_c
    cos_dist = np.clip(cos_dist, -1.0, 1.0)
    angular_dist = np.degrees(np.arccos(cos_dist))

    # Mask by FOV (angular distance from center)
    mask = angular_dist <= (float(fov) / 2.0)

    if not np.any(mask):
        return None, None, mask

    # Stereographic projection factor (vectorized)
    # avoid division by zero when cos_dist == -1
    denom = 1.0 + cos_dist
    denom[denom == 0] = 1e-12
    k = 2.0 / denom

    # Projected coordinates relative to center direction
    dx = x1 - x_c
    dy = y1 - y_c

    x = k * dx
    y = k * dy

    return x, y, mask


def get_astronomical_dusk(lat, lon, date):
    """Get astronomical dusk time (UTC) for a given date and lat/lon.

    Uses astral's Observer/sun. Returns an aware datetime in UTC.
    If calculation fails, returns a sensible fallback (20:00 UTC on that date).
    """
    try:
        observer = Observer(latitude=float(lat), longitude=float(lon), elevation=0)
        s = sun(observer, date=date, tzinfo=pytz.UTC)
        # Astronomical dusk key can be 'dusk' depending on astral version; fall back to 'sunset' + 2h
        if 'dusk' in s and s['dusk'] is not None:
            return s['dusk']
        # if dusk not provided, use sunset + 2 hours as a reasonable proxy
        if 'sunset' in s and s['sunset'] is not None:
            return s['sunset'] + timedelta(hours=2)
    except Exception as e:
        print(f"Could not calculate dusk using astral: {e}")

    # Fallback to 20:00 UTC on that date
    try:
        fallback = datetime.combine(date, dtime(hour=20, minute=0), tzinfo=pytz.UTC)
        return fallback
    except Exception:
        # Last resort: now
        return datetime.now(pytz.UTC)


def _load_hipparcos():
    """Load Hipparcos catalog into global variable (cached)."""
    global _HIPPARCOS_DF
    if _HIPPARCOS_DF is not None:
        return _HIPPARCOS_DF

    ts_local = load.timescale()
    try:
        print("Loading Hipparcos star catalog (this may take a moment)...")
        with load.open(hipparcos.URL) as f:
            _HIPPARCOS_DF = hipparcos.load_dataframe(f)
    except Exception as e:
        print(f"Failed to load Hipparcos catalog: {e}")
        _HIPPARCOS_DF = None

    return _HIPPARCOS_DF


def get_visible_stars(observer, obs_time, magnitude_limit, center_alt, center_az, fov):
    """Get stars visible from observer location at given time within FOV.

    Returns dict with 'x','y','mag','count' or None if none visible.
    """
    # Ensure catalog is loaded
    df = _load_hipparcos()
    if df is None or len(df) == 0:
        print("Hipparcos catalog not available.")
        return None

    # Filter by magnitude
    visible_df = df[df['magnitude'] <= float(magnitude_limit)].copy()
    if len(visible_df) == 0:
        return None

    # Create Skyfield Star objects
    try:
        stars = Star.from_dataframe(visible_df)
    except Exception as e:
        print(f"Error creating Star objects: {e}")
        return None

    # Observe from location/time
    ts = load.timescale()
    try:
        t = ts.from_datetime(obs_time)
    except Exception:
        # Ensure obs_time is timezone-aware in UTC
        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=pytz.UTC)
        t = ts.from_datetime(obs_time)

    astrometric = observer.at(t).observe(stars)
    app = astrometric.apparent()
    alt, az, distance = app.altaz()

    # Convert to numpy arrays (degrees)
    alt_deg = np.asarray(alt.degrees)
    az_deg = np.asarray(az.degrees)

    # Project to 2D with FOV filtering
    x, y, mask = stereographic_project(alt_deg, az_deg, center_alt, center_az, fov)

    if x is None:
        return None

    # Apply mask to magnitudes
    mags = np.asarray(visible_df['magnitude'].values)

    # Ensure mask length equals mags length
    if mask.shape[0] != mags.shape[0]:
        # In case shapes mismatch for any reason, try to broadcast sensibly
        minlen = min(mask.shape[0], mags.shape[0])
        mask = mask[:minlen]
        mags = mags[:minlen]
        x = x[:minlen]
        y = y[:minlen]

    return {
        'x': x[mask],
        'y': y[mask],
        'mag': mags[mask],
        'count': int(np.sum(mask))
    }


@style('minimal')
def minimal_style(stars, fov):

    if stars is None or stars.get('count', 0) == 0:
        return None, 'white'

    sizes = 50 * np.exp(-stars['mag'] / 2.5)

    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white', dpi=300)
    ax.set_facecolor('white')
    ax.set_aspect('equal')

    # plot stars
    ax.scatter(stars['x'], stars['y'], s=sizes, c='black', alpha=0.9, linewidths=0)

    # use actual projection radius
    radii = np.sqrt(stars['x']**2 + stars['y']**2)
    r_max = np.max(radii) * 1

    # expand visible area
    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)

    ax.axis('off')

    # draw circle boundary
    circle = plt.Circle((0, 0), r_max, color="black", fill=False, linewidth=0.2)
    ax.add_patch(circle)

    return fig, 'white'

def create_artwork(location, style_name, magnitude, fov, azimuth, altitude):
    """Create star map artwork for a single location and parameter set."""
    start_time = time.time()

    # Load ephemeris and observer
    planets = load('de421.bsp')
    earth = planets['earth']

    lat = float(location['lat'])
    lon = float(location['lon'])
    name = location.get('name', f"{lat},{lon}")

    observer = earth + wgs84.latlon(lat, lon)

    # Determine observation time (astronomical dusk for today)
    today = datetime.now(pytz.UTC).date()
    obs_time = get_astronomical_dusk(lat, lon, today)

    print(f"\nGenerating '{style_name}' for {name} at astronomical dusk (UTC): {obs_time.strftime('%Y-%m-%d %H:%M UTC')}")

    # Get visible stars: center_alt = altitude, center_az = azimuth
    stars = get_visible_stars(observer, obs_time, magnitude, altitude, azimuth, fov)

    if stars is None or stars.get('count', 0) == 0:
        print("No stars visible in this FOV, skipping...")
        return

    print(f"Stars in FOV: {stars['count']}")

    # Generate artwork using selected style
    if style_name not in STYLES:
        print(f"Error: Style '{style_name}' not found")
        return

    fig_bg = STYLES[style_name](stars, fov)
    if fig_bg is None:
        print("Failed to generate artwork (style returned None).")
        return

    fig, bg_color = fig_bg
    if fig is None:
        print("Failed to generate artwork, skipping...")
        return

    # Add information text (use first axis if available)
    ax = fig.axes[0] if len(fig.axes) > 0 else None
    if ax is not None:
        add_info_text(fig, ax, location, obs_time, magnitude, fov, azimuth, altitude, bg_color)

    # Save artwork
    date_stamp = obs_time.strftime('%Y%m%d')
    safe_name = name.replace(' ', '_').replace(',', '_')
    filename = (f"{IMAGES_DIR}/{safe_name}_{style_name}_"
                f"mag{magnitude}_fov{fov}_az{azimuth}_alt{altitude}_{date_stamp}.png")

    try:
        fig.tight_layout(pad=0.5)
        plt.savefig(filename, dpi=300, facecolor=bg_color, edgecolor='none', bbox_inches='tight')
        plt.close(fig)
        duration = time.time() - start_time
        print(f"✓ Saved: {filename} ({duration:.2f}s)")
    except Exception as e:
        print(f"Error saving figure: {e}")


def main(locations_file="stargazing-locations.json"):
    # Load locations from file
    try:
        with open(locations_file, "r") as f:
            locations = json.load(f)
    except Exception as e:
        print(f"Could not load locations file '{locations_file}': {e}")
        return

    print(f"Available styles: {', '.join(sorted(STYLES.keys()))}")
    print(f"\nGenerating artworks for {len(locations)} locations...")

    # Parameters to vary
    fovs = [180]
    azimuths = [0]
    altitudes = [90]
    magnitudes = [3.5, 6.0, 10.0]

    total = len(locations) * len(STYLES) * len(fovs) * len(azimuths) * len(altitudes) * len(magnitudes)
    current = 0

    for location in locations:
        for style_name in sorted(STYLES.keys()):
            for fov in fovs:
                for azimuth in azimuths:
                    for altitude in altitudes:
                        for magnitude in magnitudes:
                            current += 1
                            print(f"\n[{current}/{total}]", end=" ")
                            try:
                                create_artwork(location, style_name, magnitude, fov, azimuth, altitude)
                            except Exception as e:
                                print(f"Error: {e}")

    print(f"\n✓ All artworks (attempted) saved to {IMAGES_DIR}/")


if __name__ == "__main__":
    main()
