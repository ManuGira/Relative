"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    translate2D, rotate2D, scale2D, shear2D, trs2D, trks2D,
)

class TestTRS2D:
    """Tests for the trs2D combined transformation function."""

    def test_trs_identity(self):
        """Test TRS with identity transformations."""
        M = trs2D(0, 0, 0, 1, 1)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trs_translation_only(self):
        """Test TRS with only translation."""
        M = trs2D(3, 2, 0, 1, 1)
        expected = translate2D(3, 2)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trs_rotation_only(self):
        """Test TRS with only rotation."""
        angle = np.pi / 4
        M = trs2D(0, 0, angle, 1, 1)
        expected = rotate2D(angle)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trs_scale_only(self):
        """Test TRS with only scaling."""
        M = trs2D(0, 0, 0, 2, 3)
        expected = scale2D(2, 3)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trs_order_matters(self):
        """Test that TRS applies transformations in the correct order: T * R * S."""
        tx, ty = 5, 3
        angle = np.pi / 2
        sx, sy = 2, 2
        
        M = trs2D(tx, ty, angle, sx, sy)
        
        # Manually compute T * R * S
        T = translate2D(tx, ty)
        R = rotate2D(angle)
        S = scale2D(sx, sy)
        expected = T @ R @ S
        
        np.testing.assert_array_almost_equal(M, expected)

    def test_trs_transform_point(self):
        """Test TRS transformation on a point: scale, then rotate, then translate."""
        # Start with point (1, 0)
        point = np.array([1, 0, 1])
        
        # Scale by 2, rotate 90°, translate by (3, 2)
        M = trs2D(3, 2, np.pi / 2, 2, 2)
        result = M @ point
        
        # Manual calculation:
        # Scale: (1, 0) -> (2, 0)
        # Rotate 90°: (2, 0) -> (0, 2)
        # Translate: (0, 2) -> (3, 4)
        expected = np.array([3, 4, 1])
        
        np.testing.assert_array_almost_equal(result, expected)

    def test_trs_full_transformation(self):
        """Test a complete TRS transformation with all non-identity values."""
        M = trs2D(10, 5, np.pi / 3, 1.5, 2.0)
        
        # Verify it's a valid 3x3 matrix
        assert M.shape == (3, 3)
        
        # Verify the bottom row is [0, 0, 1]
        np.testing.assert_array_almost_equal(M[2, :], [0, 0, 1])

    def test_trs_matrix_properties(self):
        """Test that TRS matrix has correct properties for affine transformation."""
        M = trs2D(5, 3, np.pi / 4, 2, 1.5)
        
        # Bottom row should always be [0, 0, 1]
        assert M[2, 0] == 0
        assert M[2, 1] == 0
        assert M[2, 2] == 1
        
        # Shape should be 3x3
        assert M.shape == (3, 3)


class TestTRKS2D:
    """Tests for the trks2D combined transformation function."""

    def test_trks_identity(self):
        """Test TRKS with identity transformations."""
        M = trks2D(0, 0, 0, 0, 0, 1, 1)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trks_translation_only(self):
        """Test TRKS with only translation."""
        M = trks2D(3, 2, 0, 0, 0, 1, 1)
        expected = translate2D(3, 2)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trks_rotation_only(self):
        """Test TRKS with only rotation."""
        angle = np.pi / 4
        M = trks2D(0, 0, angle, 0, 0, 1, 1)
        expected = rotate2D(angle)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trks_shear_only(self):
        """Test TRKS with only shear."""
        M = trks2D(0, 0, 0, 0.5, 0.3, 1, 1)
        expected = shear2D(0.5, 0.3)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trks_scale_only(self):
        """Test TRKS with only scaling."""
        M = trks2D(0, 0, 0, 0, 0, 2, 3)
        expected = scale2D(2, 3)
        np.testing.assert_array_almost_equal(M, expected)

    def test_trks_order_matters(self):
        """Test that TRKS applies transformations in the correct order: T * R * K * S."""
        tx, ty = 5, 3
        angle = np.pi / 2
        kx, ky = 0.5, 0.3
        sx, sy = 2, 2
        
        M = trks2D(tx, ty, angle, kx, ky, sx, sy)
        
        # Manually compute T * R * K * S
        T = translate2D(tx, ty)
        R = rotate2D(angle)
        K = shear2D(kx, ky)
        S = scale2D(sx, sy)
        expected = T @ R @ K @ S
        
        np.testing.assert_array_almost_equal(M, expected)

    def test_trks_equals_trs_when_no_shear(self):
        """Test that TRKS equals TRS when shear is zero."""
        tx, ty = 10, 5
        angle = np.pi / 6
        sx, sy = 1.5, 2.0
        
        M_trks = trks2D(tx, ty, angle, 0, 0, sx, sy)
        M_trs = trs2D(tx, ty, angle, sx, sy)
        
        np.testing.assert_array_almost_equal(M_trks, M_trs)

    def test_trks_transform_point(self):
        """Test TRKS transformation on a point."""
        # Start with point (1, 0)
        point = np.array([1, 0, 1])
        
        # Scale by 2, shear kx=0.5, rotate 90°, translate by (3, 2)
        M = trks2D(3, 2, np.pi / 2, 0.5, 0, 2, 2)
        result = M @ point
        
        # Manual calculation:
        # Scale: (1, 0) -> (2, 0)
        # Shear (kx=0.5): x += y*0.5 = 0, so (2, 0) -> (2, 0)
        # Rotate 90°: (2, 0) -> (0, 2)
        # Translate: (0, 2) -> (3, 4)
        expected = np.array([3, 4, 1])
        
        np.testing.assert_array_almost_equal(result, expected)

    def test_trks_matrix_properties(self):
        """Test that TRKS matrix has correct properties for affine transformation."""
        M = trks2D(5, 3, np.pi / 4, 0.2, 0.1, 2, 1.5)
        
        # Bottom row should always be [0, 0, 1]
        assert M[2, 0] == 0
        assert M[2, 1] == 0
        assert M[2, 2] == 1
        
        # Shape should be 3x3
        assert M.shape == (3, 3)

