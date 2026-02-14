import csv
from datetime import datetime, timedelta, time as dtime

import matplotlib.pyplot as plt
import numpy as np
import pytz
from astral import Observer
from astral.sun import sun
from matplotlib.transforms import Bbox
from skyfield.api import load, Star
from skyfield.data import hipparcos
from timezonefinder import TimezoneFinder


class StarArtUtils:
    _TF = TimezoneFinder()
    _HIPPARCOS_CACHE = {}

    @staticmethod
    def _get_luminance(color):
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

    @classmethod
    def add_info_text(cls, fig, location, obs_time, details, bg_color):
        luminance = cls._get_luminance(bg_color)
        text_color = "black" if luminance > 128 else "white"

        if isinstance(location, dict):
            name = location.get("name", "Unknown")
            lat = location.get("lat", 0.0)
            lon = location.get("lon", 0.0)
        else:
            name = str(location)
            lat = 0.0
            lon = 0.0

        tz_name = cls._TF.timezone_at(lat=lat, lng=lon)
        if tz_name:
            try:
                tz = pytz.timezone(tz_name)
                local_time = obs_time.astimezone(tz)
                time_str = local_time.strftime("%Y-%m-%d %H:%M %Z")
            except Exception:
                time_str = obs_time.strftime("%Y-%m-%d %H:%M UTC")
        else:
            time_str = obs_time.strftime("%Y-%m-%d %H:%M UTC")

        info_text = (
            f"{name}  |  {lat:.2f}°, {lon:.2f}°  |  "
            f"{time_str}  |  {details}"
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

    @classmethod
    def get_timezone(cls, lat, lon):
        tz_name = cls._TF.timezone_at(lat=lat, lng=lon)
        if tz_name:
            try:
                return pytz.timezone(tz_name)
            except Exception:
                return None
        return None

    @staticmethod
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

    @staticmethod
    def get_astronomical_dusk(lat, lon, date, tzinfo=pytz.UTC):
        try:
            observer = Observer(latitude=float(lat), longitude=float(lon), elevation=0)
            s = sun(observer, date=date, tzinfo=tzinfo)
            if "dusk" in s and s["dusk"] is not None:
                return s["dusk"]
            if "sunset" in s and s["sunset"] is not None:
                return s["sunset"] + timedelta(hours=2)
        except Exception as e:
            print(f"Could not calculate dusk using astral: {e}")

        tz_fallback = tzinfo or pytz.UTC
        try:
            return datetime.combine(date, dtime(hour=20, minute=0), tzinfo=tz_fallback)
        except Exception:
            return datetime.now(tz_fallback)

    @staticmethod
    def get_sunrise(lat, lon, date, tzinfo=pytz.UTC):
        try:
            observer = Observer(latitude=float(lat), longitude=float(lon), elevation=0)
            s = sun(observer, date=date, tzinfo=tzinfo)
            if "sunrise" in s and s["sunrise"] is not None:
                return s["sunrise"]
        except Exception as e:
            print(f"Could not calculate sunrise using astral: {e}")

        tz_fallback = tzinfo or pytz.UTC
        try:
            return datetime.combine(date, dtime(hour=6, minute=0), tzinfo=tz_fallback)
        except Exception:
            return datetime.now(tz_fallback)

    @staticmethod
    def place_labels(ax, objects, color="#1a1a1a"):
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
            "color": color,
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

    @staticmethod
    def load_named_stars(csv_path):
        stars = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                stars.append({"name": row["name"].strip(), "hip_id": int(row["hip_id"])})
        return stars

    @classmethod
    def _load_hipparcos(cls, source="remote", hipparcos_file=None):
        cache_key = (source, hipparcos_file)
        if cache_key in cls._HIPPARCOS_CACHE:
            return cls._HIPPARCOS_CACHE[cache_key]

        try:
            if source == "remote":
                with load.open(hipparcos.URL) as f:
                    df = hipparcos.load_dataframe(f)
            else:
                with open(hipparcos_file, "rb") as f:
                    df = hipparcos.load_dataframe(f)
        except Exception as e:
            if source == "remote":
                print(f"Failed to load Hipparcos catalog: {e}")
            else:
                print(f"Failed to load Hipparcos catalog from '{hipparcos_file}': {e}")
            df = None

        cls._HIPPARCOS_CACHE[cache_key] = df
        return df

    @classmethod
    def get_visible_stars(
        cls, observer, obs_time, magnitude_limit, center_alt, center_az, fov
    ):
        df = cls._load_hipparcos(source="remote")
        if df is None or len(df) == 0:
            print("Hipparcos catalog not available.")
            return None

        visible_df = df[df["magnitude"] <= float(magnitude_limit)].copy()
        if len(visible_df) == 0:
            return None

        try:
            stars = Star.from_dataframe(visible_df)
        except Exception as e:
            print(f"Error creating Star objects: {e}")
            return None

        ts = load.timescale()
        try:
            t = ts.from_datetime(obs_time)
        except Exception:
            if obs_time.tzinfo is None:
                obs_time = obs_time.replace(tzinfo=pytz.UTC)
            t = ts.from_datetime(obs_time)

        astrometric = observer.at(t).observe(stars)
        app = astrometric.apparent()
        alt, az, _ = app.altaz()

        alt_deg = np.asarray(alt.degrees)
        az_deg = np.asarray(az.degrees)

        x, y, mask = cls.stereographic_project(alt_deg, az_deg, center_alt, center_az, fov)

        if x is None:
            return None

        mags = np.asarray(visible_df["magnitude"].values)

        if mask.shape[0] != mags.shape[0]:
            minlen = min(mask.shape[0], mags.shape[0])
            mask = mask[:minlen]
            mags = mags[:minlen]
            x = x[:minlen]
            y = y[:minlen]

        return {"x": x[mask], "y": y[mask], "mag": mags[mask], "count": int(np.sum(mask))}

    @classmethod
    def get_named_stars(
        cls,
        observer,
        obs_time,
        named_stars,
        center_alt,
        center_az,
        fov,
        hipparcos_file,
    ):
        df = cls._load_hipparcos(source="local", hipparcos_file=hipparcos_file)
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
        x, y, mask = cls.stereographic_project(
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

    @staticmethod
    def get_objects_by_ra_dec(observer, obs_time, objects, center_alt, center_az, fov):
        alt_list = []
        az_list = []
        mags = []
        names = []

        ts = load.timescale(builtin=True)
        if obs_time.tzinfo is None:
            obs_time = obs_time.replace(tzinfo=pytz.UTC)
        t = ts.from_datetime(obs_time)

        for obj in objects:
            star = Star(ra_hours=obj["ra"], dec_degrees=obj["dec"])
            alt, az, _ = observer.at(t).observe(star).apparent().altaz()
            alt_list.append(alt.degrees)
            az_list.append(az.degrees)
            mags.append(obj["mag"])
            names.append(obj["name"])

        x, y, mask = StarArtUtils.stereographic_project(
            alt_list, az_list, center_alt, center_az, fov
        )
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

    @staticmethod
    def sumi_star_style(stars, fov):
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

        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.axis("off")

        circle = plt.Circle((0, 0), r_max, color="#1a1a1a", fill=False, linewidth=0.25)
        ax.add_patch(circle)

        return fig, "#fdfdf9"

    @staticmethod
    def sumi_object_style(objects):
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

        StarArtUtils.place_labels(ax, objects)

        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.axis("off")

        circle = plt.Circle((0, 0), r_max, color="#1a1a1a", fill=False, linewidth=0.25)
        ax.add_patch(circle)

        return fig, "#fdfdf9"
