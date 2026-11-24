import unittest
from importlib import import_module
from pathlib import Path

import numpy as np
import pandas as pd

import panel_distances as distance


block = import_module("07_block_bootstrap")
BASE = Path(__file__).resolve().parents[1]


class BlockContrastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.panel = distance.load_panel_data(BASE, "main")

    def test_exact_zero_follows_each_direction_rule(self):
        counts = np.full((9, 9), 12, dtype=np.int64)
        np.fill_diagonal(counts, 0)
        row = block.evaluate_contrasts(self.panel, counts, 31)
        for metric, flag, expected in (
            ("loo_statistic", "loo_direction_ok", False),
            ("ev13_difference", "ev13_direction_ok", False),
            ("homogenization_separation", "homogenization_direction_ok", True),
        ):
            with self.subTest(metric=metric):
                self.assertEqual(row[metric], 0.0)
                self.assertEqual(row[flag], expected)

    def test_count_and_fraction_share_the_total_replicate_denominator(self):
        observed = {"panel": "main", block.ALL_FOUR: True}
        passing = {"block_size_cm": 5, block.ALL_FOUR: True}
        undefined = {"block_size_cm": 5, block.ALL_FOUR: False}
        for metric, flag in block.METRICS:
            observed[metric] = passing[metric] = 1.0
            observed[flag] = passing[flag] = True
            undefined[metric] = np.nan
            undefined[flag] = False
        for row in block.summarize(pd.DataFrame([passing, undefined]), observed, 2, 31):
            self.assertEqual(row["n_finite_replicates"], 1)
            self.assertEqual(row["n_replicates"], 2)
            self.assertEqual(row["n_preserved"], 1)
            self.assertEqual(row["direction_preservation_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
