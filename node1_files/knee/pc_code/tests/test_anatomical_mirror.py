import unittest
import numpy as np
import torch
import torch.nn as nn

from anatomical_mirror import (
    get_slot_index,
    mirror_triplets,
    infer_with_anatomical_mirror,
    SLOT_CONFIGS
)


class DummyModel(nn.Module):
    """Simple linear projection model to verify dual-pass forward/mirror inference."""
    def __init__(self, in_features=10, num_classes=12):
        super().__init__()
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # x: (B, K, 3, H, W) -> flatten and take first 10 features
        B = x.shape[0]
        feat = x.reshape(B, -1)[:, :10]
        return self.fc(feat)


class TestAnatomicalMirror(unittest.TestCase):
    def setUp(self):
        self.k = 94
        self.total_slices = 96
        # Window centers spaced evenly across slices 1..94
        self.centers = np.linspace(1, 94, self.k).astype(int)

    def test_slot_index_generation(self):
        for total_s in [96, 80, 64]:
            slots, rel, sag_slots = get_slot_index(total_s)
            self.assertEqual(len(slots), total_s)
            self.assertEqual(len(rel), total_s)
            self.assertTrue(np.all(rel >= 0.0) and np.all(rel <= 1.0))
            self.assertEqual(sag_slots, (0, 1))

    def test_involution_property_torch(self):
        """Mirroring twice must yield the exact original tensor."""
        x = torch.randn(self.k, 3, 64, 64)
        m1 = mirror_triplets(x, self.centers, total_slices=self.total_slices)
        m2 = mirror_triplets(m1, self.centers, total_slices=self.total_slices)
        self.assertEqual(m1.shape, x.shape)
        self.assertTrue(torch.allclose(m2, x, atol=1e-6))

    def test_involution_property_numpy(self):
        """NumPy array inputs must preserve shape, dtype, and involution."""
        x = np.random.randint(0, 256, size=(self.k, 3, 64, 64), dtype=np.uint8)
        m1 = mirror_triplets(x, self.centers, total_slices=self.total_slices)
        m2 = mirror_triplets(m1, self.centers, total_slices=self.total_slices)
        self.assertEqual(m1.shape, x.shape)
        self.assertEqual(m1.dtype, np.uint8)
        np.testing.assert_array_equal(m2, x)

    def test_sagittal_vs_coronal_flip_axes(self):
        """Sagittal slices must flip channels (C=3); Coronal/Axial must flip width (W)."""
        # Sagittal slice center: center 10 is in slot 0 (0..25)
        # Coronal slice center: center 60 is in slot 2 (48..65)
        centers = [10, 60]
        K = len(centers)
        
        # Construct an asymmetric triplet
        # Channel 0: all 1s, Channel 1: all 2s, Channel 2: all 3s
        # Left half of width (:16): +10.0, Right half of width (16:): +20.0
        x = torch.zeros(K, 3, 32, 32)
        x[:, 0, :, :] = 1.0
        x[:, 1, :, :] = 2.0
        x[:, 2, :, :] = 3.0
        x[:, :, :, :16] += 10.0
        x[:, :, :, 16:] += 20.0

        mirrored = mirror_triplets(x, centers, total_slices=self.total_slices)

        # Sagittal window (idx 0):
        # Channels must be reversed: Channel 0 was 1, now channel 0 should be 3
        self.assertTrue(torch.allclose(mirrored[0, 0, :, :], x[0, 2, :, :]))
        self.assertTrue(torch.allclose(mirrored[0, 2, :, :], x[0, 0, :, :]))
        # Width should NOT be flipped for sagittal: left stays left (11.0)
        self.assertTrue(torch.allclose(mirrored[0, 0, :, 0], torch.tensor([13.0])))  # 3.0 + 10.0

        # Coronal window (idx 1):
        # Channels must remain identical: Channel 0 was 1
        # Width MUST be flipped: column 0 now gets column 31 (+20.0), so 1.0 + 20.0 = 21.0
        self.assertTrue(torch.allclose(mirrored[1, 0, :, 0], torch.tensor([21.0])))
        self.assertTrue(torch.allclose(mirrored[1, 1, :, :16], x[1, 1, :, 16:]))
        self.assertTrue(torch.allclose(mirrored[1, 1, :, 16:], x[1, 1, :, :16]))

    def test_infer_with_anatomical_mirror(self):
        """Tests dual-pass TTA inference probability range and shape."""
        model = DummyModel(in_features=10, num_classes=12)
        x = torch.randn(2, self.k, 3, 32, 32)
        probs = infer_with_anatomical_mirror(model, x, self.centers, total_slices=self.total_slices)
        self.assertEqual(probs.shape, (2, 12))
        self.assertTrue(torch.all(probs >= 0.0) and torch.all(probs <= 1.0))


if __name__ == "__main__":
    unittest.main()
