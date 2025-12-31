"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import translate2D, rotate2D, scale2D, trs2D


class TestTranslate2D:
    """Tests for the translate2D function."""

    def test_translate_zero(self):
        """Test translation matrix with zero translation."""
        T = translate2D(0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_positive(self):
        """Test translation with positive values."""
        T = translate2D(3, 5)
        expected = np.array([[1, 0, 3],
                            [0, 1, 5],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_negative(self):
        """Test translation with negative values."""
        T = translate2D(-2, -4)
        expected = np.array([[1, 0, -2],
                            [0, 1, -4],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(T, expected)

    def test_translate_point(self):
        """Test that translation matrix correctly transforms a point."""
        T = translate2D(3, 2)
        point = np.array([1, 1, 1])  # Homogeneous coordinates
        result = T @ point
        expected = np.array([4, 3, 1])  # Point moved by (3, 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_translate_vector(self):
        """Test that translation matrix doesn't affect vectors (w=0)."""
        T = translate2D(3, 2)
        vector = np.array([1, 1, 0])  # Homogeneous coordinates for vector
        result = T @ vector
        expected = np.array([1, 1, 0])  # Vector unchanged
        np.testing.assert_array_almost_equal(result, expected)


class TestRotate2D:
    """Tests for the rotate2D function."""

    def test_rotate_zero(self):
        """Test rotation matrix with zero angle."""
        R = rotate2D(0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_90_degrees(self):
        """Test rotation by 90 degrees (π/2 radians)."""
        R = rotate2D(np.pi / 2)
        expected = np.array([[0, -1, 0],
                            [1,  0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_180_degrees(self):
        """Test rotation by 180 degrees (π radians)."""
        R = rotate2D(np.pi)
        expected = np.array([[-1,  0, 0],
                            [ 0, -1, 0],
                            [ 0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_270_degrees(self):
        """Test rotation by 270 degrees (3π/2 radians)."""
        R = rotate2D(3 * np.pi / 2)
        expected = np.array([[0,  1, 0],
                            [-1, 0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_360_degrees(self):
        """Test rotation by 360 degrees (2π radians) returns to identity."""
        R = rotate2D(2 * np.pi)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_45_degrees(self):
        """Test rotation by 45 degrees (π/4 radians)."""
        R = rotate2D(np.pi / 4)
        sqrt2_over_2 = np.sqrt(2) / 2
        expected = np.array([[sqrt2_over_2, -sqrt2_over_2, 0],
                            [sqrt2_over_2,  sqrt2_over_2, 0],
                            [0,             0,            1]])
        np.testing.assert_array_almost_equal(R, expected)

    def test_rotate_point(self):
        """Test that rotation matrix correctly rotates a point."""
        R = rotate2D(np.pi / 2)  # 90 degrees
        point = np.array([1, 0, 1])  # Point at (1, 0)
        result = R @ point
        expected = np.array([0, 1, 1])  # Point rotated to (0, 1)
        np.testing.assert_array_almost_equal(result, expected)

    def test_rotate_negative_angle(self):
        """Test rotation with negative angle (clockwise)."""
        R = rotate2D(-np.pi / 2)
        expected = np.array([[0,  1, 0],
                            [-1, 0, 0],
                            [0,  0, 1]])
        np.testing.assert_array_almost_equal(R, expected)


class TestScale2D:
    """Tests for the scale2D function."""

    def test_scale_identity(self):
        """Test scaling matrix with unit scale."""
        S = scale2D(1, 1)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_uniform(self):
        """Test uniform scaling."""
        S = scale2D(2, 2)
        expected = np.array([[2, 0, 0],
                            [0, 2, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_non_uniform(self):
        """Test non-uniform scaling."""
        S = scale2D(3, 2)
        expected = np.array([[3, 0, 0],
                            [0, 2, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_zero(self):
        """Test scaling to zero."""
        S = scale2D(0, 0)
        expected = np.array([[0, 0, 0],
                            [0, 0, 0],
                            [0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_negative(self):
        """Test negative scaling (reflection)."""
        S = scale2D(-1, 1)
        expected = np.array([[-1, 0, 0],
                            [ 0, 1, 0],
                            [ 0, 0, 1]])
        np.testing.assert_array_almost_equal(S, expected)

    def test_scale_point(self):
        """Test that scaling matrix correctly scales a point."""
        S = scale2D(2, 3)
        point = np.array([4, 5, 1])
        result = S @ point
        expected = np.array([8, 15, 1])  # Point scaled by (2, 3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_scale_fractional(self):
        """Test scaling with fractional values."""
        S = scale2D(0.5, 0.25)
        expected = np.array([[0.5,  0,    0],
                            [0,    0.25, 0],
                            [0,    0,    1]])
        np.testing.assert_array_almost_equal(S, expected)


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
