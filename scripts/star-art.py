import time
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pytz

from skyfield.api import load, wgs84, Star
from skyfield.data import hipparcos
from scipy.spatial import Delaunay

# Global registry for styles
STYLES = {}
IMAGES_DIR = "images"

# Create output directory
os.makedirs(IMAGES_DIR, exist_ok=True)

def style(name):
    """Decorator to register a style function"""
    def decorator(func):
        STYLES[name] = func
        return func
    return decorator

def stereographic_project(alt, az):
    """Project alt/az to 2D using stereographic projection"""
    alt_rad = np.radians(alt.degrees)
    az_rad = np.radians(az.degrees)
    
    # Stereographic projection from south pole
    r = 2 * np.tan((np.pi/2 - alt_rad) / 2)
    x = r * np.sin(az_rad)
    y = r * np.cos(az_rad)
    
    return x, y

def get_visible_stars(observer, obs_time, magnitude_limit, min_altitude=0):
    """Get stars visible from observer location at given time"""
    ts = load.timescale()
    
    # Load Hipparcos catalog
    print(f"Loading Hipparcos star catalog...")
    with load.open(hipparcos.URL) as f:
        df = hipparcos.load_dataframe(f)
    
    # Filter by magnitude
    visible_df = df[df['magnitude'] <= magnitude_limit].copy()
    print(f"Found {len(visible_df)} stars with magnitude ≤ {magnitude_limit}")
    
    # Create star objects
    stars = Star.from_dataframe(visible_df)
    
    # Observe from location
    t = ts.from_datetime(obs_time)
    astrometric = observer.at(t).observe(stars)
    alt, az, distance = astrometric.apparent().altaz()
    
    # Filter by altitude
    mask = alt.degrees > min_altitude
    
    return {
        'alt': alt.degrees[mask],
        'az': az.degrees[mask],
        'mag': visible_df['magnitude'].values[mask],
        'count': np.sum(mask)
    }

@style('minimal')
def minimal_style(observer, obs_time, magnitude, min_alt):
    """Minimalistic black stars on white background"""
    stars = get_visible_stars(observer, obs_time, magnitude, min_alt)
    
    if stars['count'] == 0:
        print("No visible stars found")
        return None, 'white'
    
    # Project to 2D
    x, y = stereographic_project(stars['alt'], stars['az'])
    
    # Star sizes based on magnitude (brighter = bigger)
    sizes = 50 * np.exp(-stars['mag'] / 2.5)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='white', dpi=300)
    ax.set_facecolor('white')
    ax.set_aspect('equal')
    
    # Plot stars
    ax.scatter(x, y, s=sizes, c='black', alpha=0.9, linewidths=0)
    
    # Add horizon circle
    circle = plt.Circle((0, 0), 2, fill=False, color='black', linewidth=1, alpha=0.3)
    ax.add_patch(circle)
    
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.axis('off')
    
    return fig, 'white'

@style('colorful')
def colorful_style(observer, obs_time, magnitude, min_alt):
    """Colorful stars with gradient colors"""
    stars = get_visible_stars(observer, obs_time, magnitude, min_alt)
    
    if stars['count'] == 0:
        return None, '#0a0e27'
    
    x, y = stereographic_project(stars['alt'], stars['az'])
    sizes = 60 * np.exp(-stars['mag'] / 2.5)
    
    # Create figure with dark blue background
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#0a0e27', dpi=300)
    ax.set_facecolor('#0a0e27')
    ax.set_aspect('equal')
    
    # Color based on magnitude (bright stars = warm colors)
    colors = plt.cm.plasma(1 - (stars['mag'] - stars['mag'].min()) / 
                           (stars['mag'].max() - stars['mag'].min() + 0.1))
    
    # Plot stars with glow effect
    for i in range(len(x)):
        ax.scatter(x[i], y[i], s=sizes[i]*2, c=[colors[i]], alpha=0.2, linewidths=0)
        ax.scatter(x[i], y[i], s=sizes[i], c=[colors[i]], alpha=0.8, linewidths=0)
    
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.axis('off')
    
    return fig, '#0a0e27'

@style('neon')
def neon_style(observer, obs_time, magnitude, min_alt):
    """Neon/cyberpunk style with bright colors on black"""
    stars = get_visible_stars(observer, obs_time, magnitude, min_alt)
    
    if stars['count'] == 0:
        return None, 'black'
    
    x, y = stereographic_project(stars['alt'], stars['az'])
    sizes = 40 * np.exp(-stars['mag'] / 2.5)
    
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='black', dpi=300)
    ax.set_facecolor('black')
    ax.set_aspect('equal')
    
    # Neon colors - cyan, magenta, yellow
    neon_colors = ['#00ffff', '#ff00ff', '#ffff00', '#00ff88', '#ff0088']
    colors = [neon_colors[i % len(neon_colors)] for i in range(len(x))]
    
    # Plot with strong glow
    for i in range(len(x)):
        ax.scatter(x[i], y[i], s=sizes[i]*4, c=colors[i], alpha=0.1, linewidths=0)
        ax.scatter(x[i], y[i], s=sizes[i]*2, c=colors[i], alpha=0.3, linewidths=0)
        ax.scatter(x[i], y[i], s=sizes[i], c=colors[i], alpha=0.9, linewidths=0)
    
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.axis('off')
    
    return fig, 'black'

@style('watercolor')
def watercolor_style(observer, obs_time, magnitude, min_alt):
    """Soft watercolor style with pastel colors"""
    stars = get_visible_stars(observer, obs_time, magnitude, min_alt)
    
    if stars['count'] == 0:
        return None, '#f5f5dc'
    
    x, y = stereographic_project(stars['alt'], stars['az'])
    sizes = 80 * np.exp(-stars['mag'] / 2.5)
    
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#f5f5dc', dpi=300)
    ax.set_facecolor('#f5f5dc')
    ax.set_aspect('equal')
    
    # Soft pastel colors
    colors = plt.cm.Pastel1(np.random.rand(len(x)))
    
    # Plot with watercolor effect (multiple layers)
    for i in range(len(x)):
        ax.scatter(x[i], y[i], s=sizes[i]*3, c=[colors[i]], alpha=0.1, linewidths=0)
        ax.scatter(x[i], y[i], s=sizes[i]*1.5, c=[colors[i]], alpha=0.2, linewidths=0)
        ax.scatter(x[i], y[i], s=sizes[i]*0.5, c=[colors[i]], alpha=0.4, linewidths=0)
    
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.axis('off')
    
    return fig, '#f5f5dc'

@style('constellation')
def constellation_style(observer, obs_time, magnitude, min_alt):
    """Connect nearby stars like constellations"""
    stars = get_visible_stars(observer, obs_time, magnitude, min_alt)
    
    if stars['count'] == 0:
        return None, '#001133'
    
    x, y = stereographic_project(stars['alt'], stars['az'])
    sizes = 30 * np.exp(-stars['mag'] / 2.5)
    
    fig, ax = plt.subplots(figsize=(12, 12), facecolor='#001133', dpi=300)
    ax.set_facecolor('#001133')
    ax.set_aspect('equal')
    
    # Connect nearby stars
    if len(x) > 3:
        points = np.column_stack([x, y])
        tri = Delaunay(points)
        
        # Only draw short connections
        for simplex in tri.simplices:
            for i in range(3):
                p1 = points[simplex[i]]
                p2 = points[simplex[(i+1)%3]]
                dist = np.linalg.norm(p1 - p2)
                if dist < 0.3:  # Only nearby stars
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'white', alpha=0.2, linewidth=0.5)
    
    # Plot stars
    ax.scatter(x, y, s=sizes, c='white', alpha=0.9, linewidths=0)
    
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.2, 2.2)
    ax.axis('off')
    
    return fig, '#001133'

def create_artwork(location, style_name, magnitude, date_str=None):
    """Create star map artwork for a location"""
    start_time = time.time()
    
    # Setup observer
    ts = load.timescale()
    planets = load('de421.bsp')
    earth = planets['earth']
    
    lat = location['lat']
    lon = location['lon']
    name = location.get('name', f"{lat},{lon}")
    
    observer = earth + wgs84.latlon(lat, lon)
    
    # Use provided date or current time
    if date_str:
        obs_time = datetime.fromisoformat(date_str)
    else:
        obs_time = datetime.now(pytz.UTC)
    
    # Generate artwork
    print(f"\nGenerating {style_name} artwork for {name}...")
    print(f"Date: {obs_time.strftime('%Y-%m-%d %H:%M UTC')}")
    
    if style_name not in STYLES:
        print(f"Error: Style '{style_name}' not found")
        return
    
    fig, bg_color = STYLES[style_name](observer, obs_time, magnitude, min_alt=10)
    
    if fig is None:
        print("No stars visible, skipping...")
        return
    
    # Save artwork
    date_stamp = obs_time.strftime('%Y%m%d')
    safe_name = name.replace(' ', '_').replace(',', '_')
    filename = f"{IMAGES_DIR}/{safe_name}_{style_name}_mag{magnitude}_{date_stamp}.png"
    
    fig.tight_layout(pad=0)
    plt.savefig(filename, dpi=300, facecolor=bg_color, 
                edgecolor='none', bbox_inches='tight')
    plt.close()
    
    duration = time.time() - start_time
    print(f"✓ Saved: {filename} ({duration:.2f}s)")

def main():
    with open("stargazing-locations.json", "r") as f:
        locations = json.load(f)
    
    print(f"Available styles: {', '.join(sorted(STYLES.keys()))}")
    print(f"\nGenerating artworks for {len(locations)} locations...")
    
    # Generate artworks
    for location in locations:
        for style_name in sorted(STYLES.keys()):
            # Create with different magnitude limits
            for magnitude in [3.5, 6.0]:
                try:
                    create_artwork(location, style_name, magnitude)
                except Exception as e:
                    print(f"Error creating {style_name} for {location.get('name')}: {e}")
    
    print(f"\n✓ All artworks saved to {IMAGES_DIR}/")

if __name__ == "__main__":
    main()