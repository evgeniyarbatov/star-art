import unittest
from datetime import datetime
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytz

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from star_art_utils import StarArtUtils


class TestStarArtUtils(unittest.TestCase):
    def test_stereographic_project_returns_mask(self):
        alt = [0, 10]
        az = [0, 90]
        x, y, mask = StarArtUtils.stereographic_project(alt, az, 0, 0, 180)
        self.assertEqual(mask.shape[0], 2)
        self.assertTrue(mask.all())
        self.assertEqual(len(x), 2)
        self.assertEqual(len(y), 2)

    def test_add_info_text_adds_text(self):
        fig = plt.figure()
        before = len(fig.texts)
        StarArtUtils.add_info_text(
            fig,
            {"name": "Test", "lat": 0.0, "lon": 0.0},
            datetime(2020, 1, 1, tzinfo=pytz.UTC),
            "Details",
            "#ffffff",
        )
        after = len(fig.texts)
        plt.close(fig)
        self.assertEqual(after, before + 1)

    def test_get_astronomical_dusk_has_tzinfo(self):
        dusk = StarArtUtils.get_astronomical_dusk(0.0, 0.0, datetime(2020, 1, 1).date())
        self.assertIsNotNone(dusk.tzinfo)


if __name__ == "__main__":
    unittest.main()
