import unittest

import torch
import torch.nn.functional as F

from pc_specific_psd import basis
from pc_specific_psd.compat_generation import derived_seed


def _make_synthetic_manifest(num_images: int, source: str = "synthetic_test_fixture") -> basis.DatasetManifest:
    entries = tuple(
        basis.ImageManifestEntry(image_id=f"img{i:04d}", path=f"synthetic://img{i:04d}", sha256=f"sha{i:04d}")
        for i in range(num_images)
    )
    return basis.DatasetManifest(source=source, entries=entries)


def _blur(image: torch.Tensor) -> torch.Tensor:
    """3x3 circular-padded smoothing, applied per channel, to inject spatial
    correlation into synthetic test images (so their patch covariance has a
    non-degenerate, well-separated eigenvalue spectrum instead of pure iid noise).
    """
    channels = image.shape[0]
    kernel = torch.tensor([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])
    kernel = (kernel / kernel.sum()).view(1, 1, 3, 3).expand(channels, 1, 3, 3)
    padded = F.pad(image.unsqueeze(0), (1, 1, 1, 1), mode="circular")
    return F.conv2d(padded, kernel, groups=channels).squeeze(0)


def _make_smooth_encoder(channels: int, height: int, width: int, base_seed: int = 0):
    def encoder(image_paths):
        latents = []
        for path in image_paths:
            seed = derived_seed(base_seed, "test_smooth_encoder_v1", path)
            generator = torch.Generator().manual_seed(seed)
            noise = torch.randn(channels, height, width, generator=generator)
            smoothed = _blur(_blur(noise))
            latents.append(smoothed)
        return torch.stack(latents, dim=0)

    return encoder


class DatasetManifestTests(unittest.TestCase):
    def test_split_disjoint_halves_covers_all_and_is_disjoint(self):
        manifest = _make_synthetic_manifest(10)
        first, second = manifest.split_disjoint_halves(seed=0)
        self.assertEqual(len(first) + len(second), 10)
        first_ids = {e.image_id for e in first.entries}
        second_ids = {e.image_id for e in second.entries}
        self.assertEqual(first_ids & second_ids, set())
        self.assertEqual(first_ids | second_ids, {e.image_id for e in manifest.entries})

    def test_split_rejects_too_few_images(self):
        manifest = _make_synthetic_manifest(1)
        with self.assertRaises(ValueError):
            manifest.split_disjoint_halves(seed=0)

    def test_manifest_hash_is_deterministic_and_order_sensitive_content(self):
        manifest_a = _make_synthetic_manifest(5)
        manifest_b = _make_synthetic_manifest(5)
        self.assertEqual(basis.manifest_hash(manifest_a), basis.manifest_hash(manifest_b))
        manifest_c = _make_synthetic_manifest(6)
        self.assertNotEqual(basis.manifest_hash(manifest_a), basis.manifest_hash(manifest_c))

    def test_manifest_roundtrip_via_json(self, tmp_path=None):
        import tempfile
        from pathlib import Path

        manifest = _make_synthetic_manifest(3)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            basis.write_dataset_manifest(manifest, path)
            loaded = basis.load_dataset_manifest(path)
        self.assertEqual(loaded.source, manifest.source)
        self.assertEqual(loaded.entries, manifest.entries)


class SampleRandomPatchesTests(unittest.TestCase):
    def test_shape_and_bounds(self):
        latents = torch.randn(2, 3, 10, 10)
        patches = basis.sample_random_patches(latents, patch_size=4, patches_per_image=5, seed=0)
        self.assertEqual(patches.shape, (10, 3 * 4 * 4))

    def test_rejects_patch_larger_than_plane(self):
        latents = torch.randn(1, 3, 4, 4)
        with self.assertRaises(ValueError):
            basis.sample_random_patches(latents, patch_size=5, patches_per_image=1, seed=0)

    def test_deterministic_given_seed(self):
        latents = torch.randn(2, 2, 8, 8)
        a = basis.sample_random_patches(latents, patch_size=3, patches_per_image=4, seed=42)
        b = basis.sample_random_patches(latents, patch_size=3, patches_per_image=4, seed=42)
        torch.testing.assert_close(a, b)

    def test_flatten_order_matches_channel_major_row_col(self):
        # A single 2x2x2 patch with distinct values at every (c, row, col) position,
        # extracted at the only possible offset, must flatten as [c0r0c0, c0r0c1,
        # c0r1c0, c0r1c1, c1r0c0, ...] -- channel-major, then row, then col.
        latents = torch.arange(2 * 2 * 2, dtype=torch.float32).reshape(1, 2, 2, 2)
        patches = basis.sample_random_patches(latents, patch_size=2, patches_per_image=1, seed=0)
        torch.testing.assert_close(patches[0], latents[0].reshape(-1))


class StreamingMeanCovarianceTests(unittest.TestCase):
    def test_matches_direct_computation(self):
        torch.manual_seed(0)
        patches = torch.randn(37, 6)
        streaming = basis.StreamingMeanCovariance(dimension=6)
        streaming.update(patches[:10])
        streaming.update(patches[10:23])
        streaming.update(patches[23:])
        mean, covariance = streaming.finalize()
        direct_mean, direct_covariance = basis.direct_mean_covariance(patches)
        torch.testing.assert_close(mean, direct_mean)
        torch.testing.assert_close(covariance, direct_covariance)

    def test_rejects_wrong_dimension(self):
        streaming = basis.StreamingMeanCovariance(dimension=4)
        with self.assertRaises(ValueError):
            streaming.update(torch.randn(3, 5))

    def test_requires_at_least_two_samples(self):
        streaming = basis.StreamingMeanCovariance(dimension=4)
        streaming.update(torch.randn(1, 4))
        with self.assertRaises(ValueError):
            streaming.finalize()


class EigendecompositionTests(unittest.TestCase):
    def test_descending_order_and_reconstruction(self):
        torch.manual_seed(1)
        patches = torch.randn(200, 5)
        _, covariance = basis.direct_mean_covariance(patches)
        eigenvalues, components = basis.eigendecompose_covariance(covariance)
        self.assertTrue(torch.all(eigenvalues[:-1] >= eigenvalues[1:]))
        reconstructed = components @ torch.diag(eigenvalues) @ components.T
        torch.testing.assert_close(reconstructed, covariance, atol=1e-8, rtol=1e-6)

    def test_orthogonality_error_near_zero(self):
        torch.manual_seed(2)
        patches = torch.randn(200, 5)
        _, covariance = basis.direct_mean_covariance(patches)
        _, components = basis.eigendecompose_covariance(covariance)
        self.assertLess(basis.orthogonality_error(components), 1e-8)


class BuildPCABasisTests(unittest.TestCase):
    def test_empty_manifest_raises(self):
        manifest = basis.DatasetManifest(source="synthetic_test_fixture", entries=())
        with self.assertRaises(ValueError):
            basis.build_pca_basis(
                manifest, lambda paths: torch.zeros(len(paths), 1, 4, 4),
                patch_size=2, channels=1, patches_per_image=1, sampling_seed=0, synthetic=True,
            )

    def test_encoder_shape_mismatch_raises(self):
        manifest = _make_synthetic_manifest(3)

        def bad_encoder(paths):
            return torch.randn(len(paths), 5, 4, 4)  # wrong channel count

        with self.assertRaises(ValueError):
            basis.build_pca_basis(
                manifest, bad_encoder, patch_size=2, channels=1, patches_per_image=1,
                sampling_seed=0, synthetic=True,
            )

    def test_builds_valid_orthonormal_basis_and_metadata(self):
        manifest = _make_synthetic_manifest(6)
        encoder = _make_smooth_encoder(channels=2, height=8, width=8)
        result = basis.build_pca_basis(
            manifest, encoder, patch_size=3, channels=2, patches_per_image=5,
            sampling_seed=123, synthetic=True, chunk_size=4,
        )
        d = 2 * 3 * 3
        self.assertEqual(result.components.shape, (d, d))
        self.assertLess(basis.orthogonality_error(result.components), 1e-4)
        self.assertEqual(result.num_samples, 6 * 5)
        self.assertTrue(result.metadata["synthetic"])
        self.assertEqual(result.metadata["format_version"], basis.BASIS_FORMAT_VERSION)
        self.assertEqual(result.metadata["manifest_hash"], basis.manifest_hash(manifest))

    def test_same_chunk_boundaries_give_identical_result(self):
        # Per-chunk patch sampling is seeded by chunk_start (derived_seed(sampling_seed,
        # NAMESPACE, chunk_start)), so two chunk_size values that both put every image
        # in a single chunk (chunk_start=0 only) must reproduce the same result bit for
        # bit; two chunk_size values that place different chunk boundaries are NOT
        # expected to agree, since they draw different (equally valid) random patches.
        manifest = _make_synthetic_manifest(8)
        encoder = _make_smooth_encoder(channels=1, height=6, width=6)
        single_chunk_a = basis.build_pca_basis(
            manifest, encoder, patch_size=2, channels=1, patches_per_image=3,
            sampling_seed=7, synthetic=True, chunk_size=100,
        )
        single_chunk_b = basis.build_pca_basis(
            manifest, encoder, patch_size=2, channels=1, patches_per_image=3,
            sampling_seed=7, synthetic=True, chunk_size=1000,
        )
        torch.testing.assert_close(single_chunk_a.eigenvalues, single_chunk_b.eigenvalues)
        torch.testing.assert_close(single_chunk_a.mean, single_chunk_b.mean)

    def test_different_chunk_boundaries_still_process_every_image(self):
        manifest = _make_synthetic_manifest(8)
        encoder = _make_smooth_encoder(channels=1, height=6, width=6)
        multi_chunk = basis.build_pca_basis(
            manifest, encoder, patch_size=2, channels=1, patches_per_image=3,
            sampling_seed=7, synthetic=True, chunk_size=3,
        )
        single_chunk = basis.build_pca_basis(
            manifest, encoder, patch_size=2, channels=1, patches_per_image=3,
            sampling_seed=7, synthetic=True, chunk_size=100,
        )
        self.assertEqual(multi_chunk.num_samples, single_chunk.num_samples)
        self.assertEqual(multi_chunk.num_samples, 8 * 3)


class SplitHalfStabilityTests(unittest.TestCase):
    def test_leading_subspace_is_reasonably_stable_for_smooth_data(self):
        manifest = _make_synthetic_manifest(120)
        encoder = _make_smooth_encoder(channels=1, height=10, width=10, base_seed=99)
        result = basis.split_half_stability(
            manifest, encoder, patch_size=3, channels=1, patches_per_image=40,
            sampling_seed=1, split_seed=2, synthetic=True, num_leading_components=1,
        )
        # A random pair of unit vectors in this dimension would concentrate near pi/2;
        # smoothed synthetic data should give a leading eigenvector far more aligned
        # than that between two disjoint image halves.
        self.assertLess(result.max_angle_leading, 1.2)
        self.assertTrue(torch.all(result.angles_full >= 0.0))
        self.assertTrue(torch.all(result.angles_full <= torch.pi / 2 + 1e-6))


class SaveLoadBasisTests(unittest.TestCase):
    def _build(self, synthetic=True):
        manifest = _make_synthetic_manifest(4)
        encoder = _make_smooth_encoder(channels=1, height=6, width=6)
        return basis.build_pca_basis(
            manifest, encoder, patch_size=2, channels=1, patches_per_image=3,
            sampling_seed=5, synthetic=synthetic,
        )

    def test_roundtrip(self):
        import tempfile
        from pathlib import Path

        built = self._build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basis.pt"
            basis.save_basis(built, path)
            loaded = basis.load_basis(path, allow_synthetic=True)
        torch.testing.assert_close(loaded.components, built.components)
        torch.testing.assert_close(loaded.eigenvalues, built.eigenvalues)
        self.assertEqual(loaded.patch_size, built.patch_size)
        self.assertEqual(loaded.channels, built.channels)

    def test_refuses_synthetic_basis_without_allow_flag(self):
        import tempfile
        from pathlib import Path

        built = self._build(synthetic=True)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basis.pt"
            basis.save_basis(built, path)
            with self.assertRaises(ValueError):
                basis.load_basis(path, allow_synthetic=False)

    def test_rejects_patch_size_mismatch(self):
        import tempfile
        from pathlib import Path

        built = self._build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basis.pt"
            basis.save_basis(built, path)
            with self.assertRaises(ValueError):
                basis.load_basis(path, allow_synthetic=True, expected_patch_size=99)

    def test_rejects_channels_mismatch(self):
        import tempfile
        from pathlib import Path

        built = self._build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basis.pt"
            basis.save_basis(built, path)
            with self.assertRaises(ValueError):
                basis.load_basis(path, allow_synthetic=True, expected_channels=99)

    def test_rejects_format_version_mismatch(self):
        import tempfile
        from pathlib import Path

        built = self._build()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "basis.pt"
            basis.save_basis(built, path)
            payload = torch.load(path, weights_only=False)
            payload["metadata"]["format_version"] = "some_other_version"
            torch.save(payload, path)
            with self.assertRaises(ValueError):
                basis.load_basis(path, allow_synthetic=True)


class FrequencyCentroidTests(unittest.TestCase):
    def test_dc_vector_has_zero_centroid(self):
        patch_size, channels = 4, 1
        d = channels * patch_size * patch_size
        constant_vector = torch.ones(d) / (d ** 0.5)
        centroid = basis.basis_vector_patch_frequency_centroid(constant_vector, patch_size, channels)
        self.assertAlmostEqual(centroid, 0.0, places=6)

    def test_checkerboard_vector_has_larger_centroid_than_dc(self):
        patch_size, channels = 4, 1
        checkerboard = torch.tensor([[(-1.0) ** (r + c) for c in range(patch_size)] for r in range(patch_size)])
        checkerboard_vector = checkerboard.reshape(-1)
        checkerboard_vector = checkerboard_vector / checkerboard_vector.norm()
        centroid = basis.basis_vector_patch_frequency_centroid(checkerboard_vector, patch_size, channels)
        self.assertGreater(centroid, 0.0)

    def test_basis_frequency_centroids_shape(self):
        patch_size, channels = 3, 2
        d = channels * patch_size * patch_size
        components = torch.eye(d)
        centroids = basis.basis_frequency_centroids(components, patch_size, channels)
        self.assertEqual(centroids.shape, (d,))
        self.assertFalse(torch.isnan(centroids).any())


if __name__ == "__main__":
    unittest.main()
