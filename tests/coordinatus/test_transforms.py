"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    translate2D, rotate2D, scale2D, shear2D, trs2D, trks2D,
    projection, projection2D_to_1D, projection3D_to_2D
)


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


class TestShear2D:
    """Tests for the shear2D function."""

    def test_shear_identity(self):
        """Test shear matrix with zero shear."""
        K = shear2D(0, 0)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_x_only(self):
        """Test shear in x direction only."""
        K = shear2D(0.5, 0)
        expected = np.array([[1, 0.5, 0],
                            [0, 1,   0],
                            [0, 0,   1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_y_only(self):
        """Test shear in y direction only."""
        K = shear2D(0, 0.5)
        expected = np.array([[1,   0, 0],
                            [0.5, 1, 0],
                            [0,   0, 1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_both_axes(self):
        """Test shear in both x and y directions."""
        K = shear2D(0.3, 0.7)
        expected = np.array([[1,   0.3, 0],
                            [0.7, 1,   0],
                            [0,   0,   1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_negative(self):
        """Test shear with negative values."""
        K = shear2D(-0.5, -0.25)
        expected = np.array([[1,     -0.5, 0],
                            [-0.25, 1,    0],
                            [0,     0,    1]])
        np.testing.assert_array_almost_equal(K, expected)

    def test_shear_point_x(self):
        """Test that shear matrix correctly shears a point in x direction."""
        K = shear2D(1, 0)  # Shear x by y amount
        point = np.array([0, 2, 1])  # Point at (0, 2)
        result = K @ point
        expected = np.array([2, 2, 1])  # x shifted by y*kx = 2*1 = 2
        np.testing.assert_array_almost_equal(result, expected)

    def test_shear_point_y(self):
        """Test that shear matrix correctly shears a point in y direction."""
        K = shear2D(0, 1)  # Shear y by x amount
        point = np.array([3, 0, 1])  # Point at (3, 0)
        result = K @ point
        expected = np.array([3, 3, 1])  # y shifted by x*ky = 3*1 = 3
        np.testing.assert_array_almost_equal(result, expected)

    def test_shear_vector(self):
        """Test that shear matrix correctly shears a vector."""
        K = shear2D(0.5, 0.5)
        vector = np.array([2, 2, 0])  # Vector (w=0)
        result = K @ vector
        expected = np.array([3, 3, 0])  # x += y*0.5, y += x*0.5
        np.testing.assert_array_almost_equal(result, expected)


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


class TestProjection:
    """Tests for the projection function."""

    def test_projection_2d_to_1d_remove_x(self):
        """Test projection from 2D to 1D by removing x-axis (axis 0)."""
        P = projection(2, 0)
        
        # Should be a 2x3 matrix (removes one dimension from 2D + homogeneous)
        assert P.shape == (2, 3)
        
        # Should be identity with axis 0 row removed
        expected = np.array([[0, 1, 0],
                            [0, 0, 1]])
        np.testing.assert_array_equal(P, expected)

    def test_projection_2d_to_1d_remove_y(self):
        """Test projection from 2D to 1D by removing y-axis (axis 1)."""
        P = projection(2, 1)
        
        # Should be a 2x3 matrix
        assert P.shape == (2, 3)
        
        # Should be identity with axis 1 row removed
        expected = np.array([[1, 0, 0],
                            [0, 0, 1]])
        np.testing.assert_array_equal(P, expected)

    def test_projection_3d_to_2d_remove_x(self):
        """Test projection from 3D to 2D by removing x-axis (axis 0)."""
        P = projection(3, 0)
        
        # Should be a 3x4 matrix
        assert P.shape == (3, 4)
        
        # Should be identity with axis 0 row removed
        expected = np.array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(P, expected)

    def test_projection_3d_to_2d_remove_y(self):
        """Test projection from 3D to 2D by removing y-axis (axis 1)."""
        P = projection(3, 1)
        
        # Should be a 3x4 matrix
        assert P.shape == (3, 4)
        
        expected = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(P, expected)

    def test_projection_3d_to_2d_remove_z(self):
        """Test projection from 3D to 2D by removing z-axis (axis 2)."""
        P = projection(3, 2)
        
        # Should be a 3x4 matrix
        assert P.shape == (3, 4)
        
        expected = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(P, expected)

    def test_projection_2d_point_remove_x(self):
        """Test projection of a 2D point onto 1D by removing x-axis."""
        P = projection(2, 0)
        point = np.array([3, 5, 1])  # Point at (3, 5) in homogeneous coords
        
        result = P @ point
        
        # Should keep only y and weight
        expected = np.array([5, 1])
        np.testing.assert_array_equal(result, expected)

    def test_projection_2d_point_remove_y(self):
        """Test projection of a 2D point onto 1D by removing y-axis."""
        P = projection(2, 1)
        point = np.array([3, 5, 1])  # Point at (3, 5) in homogeneous coords
        
        result = P @ point
        
        # Should keep only x and weight
        expected = np.array([3, 1])
        np.testing.assert_array_equal(result, expected)

    def test_projection_3d_point_remove_z(self):
        """Test projection of a 3D point onto 2D by removing z-axis."""
        P = projection(3, 2)
        point = np.array([2, 4, 7, 1])  # Point at (2, 4, 7) in homogeneous coords
        
        result = P @ point
        
        # Should keep x, y, and weight
        expected = np.array([2, 4, 1])
        np.testing.assert_array_equal(result, expected)

    def test_projection_vector_not_affected_by_weight(self):
        """Test that projection works correctly for vectors (w=0)."""
        P = projection(2, 0)
        vector = np.array([3, 5, 0])  # Vector in homogeneous coords
        
        result = P @ vector
        
        # Should keep only y component and weight
        expected = np.array([5, 0])
        np.testing.assert_array_equal(result, expected)

    def test_projection_assertion_error(self):
        """Test that projection raises assertion error for invalid axis."""
        try:
            projection(2, 2)  # axis >= initial_dim
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "Initial dimension must be greater than the axis" in str(e)

    def test_projection_multiple_points_2d_to_1d(self):
        """Test projection of multiple 2D points to 1D."""
        P = projection(2, 1)  # Remove y-axis
        
        # Multiple points: (1,2), (3,4), (5,6) in homogeneous coords
        points = np.array([[1, 3, 5],
                          [2, 4, 6],
                          [1, 1, 1]])
        
        result = P @ points
        
        # Should keep only x-coordinates and weights
        expected = np.array([[1, 3, 5],
                            [1, 1, 1]])
        np.testing.assert_array_equal(result, expected)


class TestProjection2DTo1D:
    """Tests for the projection2D_to_1D convenience function."""

    def test_projection2d_to_1d_axis_0(self):
        """Test 2D to 1D projection removing x-axis."""
        P = projection2D_to_1D(0)
        
        assert P.shape == (2, 3)
        expected = projection(2, 0)
        np.testing.assert_array_equal(P, expected)

    def test_projection2d_to_1d_axis_1(self):
        """Test 2D to 1D projection removing y-axis."""
        P = projection2D_to_1D(1)
        
        assert P.shape == (2, 3)
        expected = projection(2, 1)
        np.testing.assert_array_equal(P, expected)

    def test_projection2d_to_1d_transform_point(self):
        """Test that projection2D_to_1D correctly transforms a point."""
        P = projection2D_to_1D(0)  # Remove x-axis, keep y
        point = np.array([10, 20, 1])
        
        result = P @ point
        
        expected = np.array([20, 1])  # Only y-coordinate remains
        np.testing.assert_array_equal(result, expected)


class TestProjection3DTo2D:
    """Tests for the projection3D_to_2D convenience function."""

    def test_projection3d_to_2d_axis_0(self):
        """Test 3D to 2D projection removing x-axis."""
        P = projection3D_to_2D(0)
        
        assert P.shape == (3, 4)
        expected = projection(3, 0)
        np.testing.assert_array_equal(P, expected)

    def test_projection3d_to_2d_axis_1(self):
        """Test 3D to 2D projection removing y-axis."""
        P = projection3D_to_2D(1)
        
        assert P.shape == (3, 4)
        expected = projection(3, 1)
        np.testing.assert_array_equal(P, expected)

    def test_projection3d_to_2d_axis_2(self):
        """Test 3D to 2D projection removing z-axis."""
        P = projection3D_to_2D(2)
        
        assert P.shape == (3, 4)
        expected = projection(3, 2)
        np.testing.assert_array_equal(P, expected)

    def test_projection3d_to_2d_transform_point(self):
        """Test that projection3D_to_2D correctly transforms a point."""
        P = projection3D_to_2D(2)  # Remove z-axis, keep x and y
        point = np.array([5, 10, 15, 1])
        
        result = P @ point
        
        expected = np.array([5, 10, 1])  # Only x and y remain
        np.testing.assert_array_equal(result, expected)

    def test_projection3d_to_2d_transform_vector(self):
        """Test that projection3D_to_2D correctly transforms a vector."""
        P = projection3D_to_2D(0)  # Remove x-axis
        vector = np.array([1, 2, 3, 0])  # Vector in homogeneous coords
        
        result = P @ vector
        
        expected = np.array([2, 3, 0])  # Only y, z, and weight remain
        np.testing.assert_array_equal(result, expected)
