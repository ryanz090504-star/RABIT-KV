"""Tests for kvquant.snapshots."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from kvquant.snapshots import (
    KVSnapshot,
    capture_snapshots,
    load_layer_snapshot,
    load_snapshots,
    save_snapshots,
    _validate_layer_indices,
)


class TestKVSnapshot(unittest.TestCase):
    def test_properties(self):
        k = np.zeros((1, 4, 16, 64), dtype=np.float32)
        v = np.ones((1, 4, 16, 64), dtype=np.float32)
        snap = KVSnapshot(keys=k, values=v, layer_idx=3, step=42)
        self.assertEqual(snap.layer_idx, 3)
        self.assertEqual(snap.step, 42)
        self.assertEqual(snap.batch_size, 1)
        self.assertEqual(snap.kv_heads, 4)
        self.assertEqual(snap.seq_len, 16)
        self.assertEqual(snap.head_dim, 64)
        self.assertEqual(snap.shape, (1, 4, 16, 64))


class TestSaveLoadRoundTrip(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_round_trip_single(self):
        k = np.random.randn(1, 4, 8, 16).astype(np.float32)
        v = np.random.randn(1, 4, 8, 16).astype(np.float32)
        snap = KVSnapshot(
            keys=k, values=v, layer_idx=0, step=100, metadata={"model": "test-model"}
        )
        saved = save_snapshots([snap], self.dir)
        self.assertTrue(any(p.suffix == ".npz" for p in saved))
        self.assertTrue((self.dir / "index.json").exists())

        loaded = load_snapshots(self.dir)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].layer_idx, 0)
        self.assertEqual(loaded[0].step, 100)
        np.testing.assert_array_equal(loaded[0].keys, k)
        np.testing.assert_array_equal(loaded[0].values, v)
        self.assertEqual(loaded[0].metadata.get("model"), "test-model")

    def test_round_trip_multiple(self):
        snaps = []
        for layer in range(3):
            k = np.random.randn(1, 2, 4, 8).astype(np.float32) + layer
            v = np.random.randn(1, 2, 4, 8).astype(np.float32) + layer
            snaps.append(KVSnapshot(keys=k, values=v, layer_idx=layer, step=255))
        save_snapshots(snaps, self.dir)
        loaded = load_snapshots(self.dir)
        self.assertEqual(len(loaded), 3)
        self.assertEqual({s.layer_idx for s in loaded}, {0, 1, 2})

    def test_load_layer_snapshot(self):
        snaps = []
        for layer in range(3):
            k = np.full((1, 2, 4, 8), float(layer), dtype=np.float32)
            v = np.full((1, 2, 4, 8), float(layer), dtype=np.float32)
            snaps.append(KVSnapshot(keys=k, values=v, layer_idx=layer, step=10))
        save_snapshots(snaps, self.dir)

        snap = load_layer_snapshot(self.dir, layer_idx=1)
        self.assertIsNotNone(snap)
        self.assertEqual(snap.layer_idx, 1)

    def test_load_layer_snapshot_missing_returns_none(self):
        k = np.zeros((1, 2, 4, 8), dtype=np.float32)
        v = np.zeros((1, 2, 4, 8), dtype=np.float32)
        save_snapshots([KVSnapshot(keys=k, values=v, layer_idx=0, step=1)], self.dir)
        self.assertIsNone(load_layer_snapshot(self.dir, layer_idx=99))

    def test_load_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            load_snapshots("/nonexistent/dir/12345")

    def test_load_missing_index(self):
        self.dir.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(FileNotFoundError):
            load_snapshots(self.dir)

    def test_corrupted_npz_skipped(self):
        """A corrupted or unreadable .npz should be skipped, not crash."""
        k = np.zeros((1, 2, 4, 8), dtype=np.float32)
        v = np.zeros((1, 2, 4, 8), dtype=np.float32)
        save_snapshots([KVSnapshot(keys=k, values=v, layer_idx=0, step=1)], self.dir)
        # write a bogus file that is not a valid npz
        bad = self.dir / "step_99_layer_0.npz"
        bad.write_bytes(b"not a numpy archive")
        # corrupt the index to reference the bad file
        index_path = self.dir / "index.json"
        index = json.loads(index_path.read_text())
        index.append({"file": "step_99_layer_0.npz", "layer_idx": 99, "step": 99})
        index_path.write_text(json.dumps(index))
        # loading should still succeed (bad file is skipped)
        loaded = load_snapshots(self.dir)
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0].layer_idx, 0)


class TestCaptureSnapshotsLegacyCache(unittest.TestCase):
    def _make_tensor(self, data):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not available")
        return torch.tensor(data)

    def test_capture_all_layers(self):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not available")
        cache = (
            (
                self._make_tensor(np.zeros((1, 4, 3, 8), dtype=np.float32)),
                self._make_tensor(np.ones((1, 4, 3, 8), dtype=np.float32)),
            ),
            (
                self._make_tensor(np.full((1, 4, 3, 8), 2.0, dtype=np.float32)),
                self._make_tensor(np.full((1, 4, 3, 8), 3.0, dtype=np.float32)),
            ),
        )
        snaps = capture_snapshots(cache, step=5)
        self.assertEqual(len(snaps), 2)
        self.assertEqual(snaps[0].layer_idx, 0)
        self.assertEqual(snaps[1].layer_idx, 1)
        self.assertEqual(snaps[0].step, 5)
        self.assertAlmostEqual(float(snaps[0].keys.mean()), 0.0, places=3)
        self.assertAlmostEqual(float(snaps[1].keys.mean()), 2.0, places=3)

    def test_capture_subset(self):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not available")
        cache = tuple(
            (
                self._make_tensor(np.zeros((1, 2, 2, 4), dtype=np.float32)),
                self._make_tensor(np.zeros((1, 2, 2, 4), dtype=np.float32)),
            )
            for _ in range(5)
        )
        snaps = capture_snapshots(cache, step=10, layer_indices=[0, 4])
        self.assertEqual(len(snaps), 2)
        self.assertEqual({s.layer_idx for s in snaps}, {0, 4})

    def test_layer_index_out_of_range(self):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not available")
        cache = (
            (
                self._make_tensor(np.zeros((1, 2, 2, 4), dtype=np.float32)),
                self._make_tensor(np.zeros((1, 2, 2, 4), dtype=np.float32)),
            ),
        )
        with self.assertRaises(ValueError):
            capture_snapshots(cache, step=0, layer_indices=[999])

    def test_empty_cache_raises(self):
        with self.assertRaises((RuntimeError, ValueError)):
            capture_snapshots(None, step=0)

    def test_cache_list_format(self):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("torch not available")
        cache = [
            (
                self._make_tensor(np.zeros((1, 2, 2, 4), dtype=np.float32)),
                self._make_tensor(np.ones((1, 2, 2, 4), dtype=np.float32)),
            )
        ]
        snaps = capture_snapshots(cache, step=0)
        self.assertEqual(len(snaps), 1)


class TestValidateLayerIndices(unittest.TestCase):
    def test_valid(self):
        _validate_layer_indices([0, 1, 2], 3)

    def test_negative(self):
        with self.assertRaises(ValueError):
            _validate_layer_indices([-1], 3)

    def test_out_of_range(self):
        with self.assertRaises(ValueError):
            _validate_layer_indices([3], 3)


if __name__ == "__main__":
    unittest.main()
