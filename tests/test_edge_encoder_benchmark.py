from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cross_device_gemma.edge_encoder_benchmark import (
    compare_bfloat16,
    format_report,
    percentile,
    summarize,
)


class EdgeEncoderBenchmarkTests(unittest.TestCase):
    def test_summary_interpolates_percentiles(self):
        summary = summarize([4.0, 1.0, 2.0, 3.0, 5.0])
        self.assertEqual(summary["p50_ms"], 3.0)
        self.assertEqual(summary["p90_ms"], 4.6)
        self.assertEqual(percentile([1.0, 2.0], 0.5), 1.5)

    def test_bfloat16_comparison_reads_raw_tensor_bits(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.bin"
            right = root / "right.bin"
            np.array([0x3F80, 0x4000], dtype="<u2").tofile(left)
            np.array([0x3F80, 0x4040], dtype="<u2").tofile(right)

            comparison = compare_bfloat16(left, right)

        self.assertEqual(comparison["elements"], 2)
        self.assertEqual(comparison["bit_identical_fraction"], 0.5)
        self.assertTrue(comparison["finite"])
        self.assertGreater(comparison["relative_l2_difference"], 0)

    def test_bfloat16_comparison_rejects_different_shapes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            left = root / "left.bin"
            right = root / "right.bin"
            np.array([1], dtype="<u2").tofile(left)
            np.array([1, 2], dtype="<u2").tofile(right)
            with self.assertRaisesRegex(RuntimeError, "different element counts"):
                compare_bfloat16(left, right)

    def test_report_highlights_product_metrics(self):
        arm = {
            "device": "mps",
            "tensor": {"shape": [264, 2560], "dtype": "bfloat16"},
            "timing": {
                "encode": {"p50_ms": 600.0, "p90_ms": 700.0},
                "total_local_work": {"p50_ms": 625.0},
            },
        }
        report = {
            "model": "google/gemma-4-E4B-it",
            "revision": "abcdef1234567890",
            "input": "/tmp/image.png",
            "baseline": arm,
            "optimized": {
                **arm,
                "device": "mlx:gpu",
                "timing": {
                    "encode": {"p50_ms": 300.0, "p90_ms": 350.0},
                    "total_local_work": {"p50_ms": 325.0},
                },
            },
            "comparison": {
                "encoder_p50_speedup": 2.0,
                "encoder_p50_latency_reduction_percent": 50.0,
                "total_local_work_p50_latency_reduction_percent": 48.0,
                "numerical": {"relative_l2_difference": 0.01, "finite": True},
            },
        }

        rendered = format_report(report)

        self.assertIn("Encoder speedup: 2.00x", rendered)
        self.assertIn("Encoder latency reduction: 50.0%", rendered)
        self.assertIn("[264 x 2560] bfloat16", rendered)


if __name__ == "__main__":
    unittest.main()
