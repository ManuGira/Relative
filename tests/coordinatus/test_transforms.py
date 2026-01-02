"""Unit tests for transformation matrix functions."""

import numpy as np
from coordinatus.transforms import (
    translate2D, rotate2D, scale2D, shear2D, trs2D, trks2D,
    swap_axes,
    reduce_dim, 
    project_xyz_to_xy, 
    project_xyz_to_xz, 
    project_xyz_to_yz,
    project_xyz_to_x,
    project_xyz_to_y,
    project_xyz_to_z,
    project_xy_to_x, 
    project_xy_to_y,
    augment_dim,
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


class TestSwapAxes:
    """Tests for the swap_axes function."""

    def test_swap_axes_2d_identity(self):
        """Test swapping an axis with itself returns identity."""
        S = swap_axes(2, 0, 0)
        expected = np.eye(3)
        np.testing.assert_array_equal(S, expected)

    def test_swap_axes_2d(self):
        """Test swapping x and y axes in 2D."""
        S = swap_axes(2, 0, 1)
        
        # Shape should be 3x3 (2D + homogeneous)
        assert S.shape == (3, 3)
        
        # Should swap first two rows
        expected = np.array([[0, 1, 0],
                            [1, 0, 0],
                            [0, 0, 1]])
        np.testing.assert_array_equal(S, expected)

    def test_swap_axes_3d_x_y(self):
        """Test swapping x and y axes in 3D."""
        S = swap_axes(3, 0, 1)
        
        # Shape should be 4x4 (3D + homogeneous)
        assert S.shape == (4, 4)
        
        expected = np.array([[0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(S, expected)

    def test_swap_axes_3d_x_z(self):
        """Test swapping x and z axes in 3D."""
        S = swap_axes(3, 0, 2)
        
        expected = np.array([[0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [1, 0, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(S, expected)

    def test_swap_axes_3d_y_z(self):
        """Test swapping y and z axes in 3D."""
        S = swap_axes(3, 1, 2)
        
        expected = np.array([[1, 0, 0, 0],
                            [0, 0, 1, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(S, expected)

    def test_swap_axes_transform_point_2d(self):
        """Test that swap_axes correctly transforms a 2D point."""
        S = swap_axes(2, 0, 1)
        point = np.array([3, 5, 1])  # Point at (3, 5)
        
        result = S @ point
        
        expected = np.array([5, 3, 1])  # Swapped to (5, 3)
        np.testing.assert_array_equal(result, expected)

    def test_swap_axes_transform_point_3d(self):
        """Test that swap_axes correctly transforms a 3D point."""
        S = swap_axes(3, 0, 2)  # Swap x and z
        point = np.array([1, 2, 3, 1])  # Point at (1, 2, 3)
        
        result = S @ point
        
        expected = np.array([3, 2, 1, 1])  # Swapped to (3, 2, 1)
        np.testing.assert_array_equal(result, expected)

    def test_swap_axes_assertion_error_axis1_out_of_bounds(self):
        """Test that swap_axes raises assertion error for invalid axis1."""
        try:
            swap_axes(2, 2, 0)  # axis1 >= dim
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "axis1 out of bounds" in str(e)

    def test_swap_axes_assertion_error_axis2_out_of_bounds(self):
        """Test that swap_axes raises assertion error for invalid axis2."""
        try:
            swap_axes(2, 0, 3)  # axis2 >= dim
            assert False, "Should have raised AssertionError"
        except AssertionError as e:
            assert "axis2 out of bounds" in str(e)


class TestReduceDim:
    """Tests for the reduce_dim function."""

    def test_reduce_dim_2d_to_1d(self):
        """Test reducing from 2D to 1D by removing the last (y) axis."""
        R = reduce_dim(2)
        
        # Should be 2x3 matrix (removes y-axis, keeps x and weight)
        assert R.shape == (2, 3)
        
        # Should keep first row and last row (x and weight)
        expected = np.array([[1, 0, 0],
                            [0, 0, 1]])
        np.testing.assert_array_equal(R, expected)

    def test_reduce_dim_3d_to_2d(self):
        """Test reducing from 3D to 2D by removing the last (z) axis."""
        R = reduce_dim(3)
        
        # Should be 3x4 matrix (removes z-axis, keeps x, y, and weight)
        assert R.shape == (3, 4)
        
        # Should keep first two rows and last row (x, y, and weight)
        expected = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(R, expected)

    def test_reduce_dim_transform_point_2d(self):
        """Test that reduce_dim correctly transforms a 2D point to 1D."""
        R = reduce_dim(2)
        point = np.array([3, 5, 1])  # Point at (3, 5)
        
        result = R @ point
        
        expected = np.array([3, 1])  # Keep only x and weight
        np.testing.assert_array_equal(result, expected)

    def test_reduce_dim_transform_point_3d(self):
        """Test that reduce_dim correctly transforms a 3D point to 2D."""
        R = reduce_dim(3)
        point = np.array([2, 4, 7, 1])  # Point at (2, 4, 7)
        
        result = R @ point
        
        expected = np.array([2, 4, 1])  # Keep only x, y, and weight
        np.testing.assert_array_equal(result, expected)

    def test_reduce_dim_vector(self):
        """Test that reduce_dim correctly handles vectors."""
        R = reduce_dim(3)
        vector = np.array([1, 2, 3, 0])  # Vector with w=0
        
        result = R @ vector
        
        expected = np.array([1, 2, 0])  # Keep x, y, and weight
        np.testing.assert_array_equal(result, expected)


class TestProjectXYToX:
    """Tests for the project_xy_to_x function."""

    def test_project_xy_to_x_shape(self):
        """Test that project_xy_to_x returns correct shape."""
        P = project_xy_to_x()
        assert P.shape == (2, 3)

    def test_project_xy_to_x_matrix_structure(self):
        """Test the matrix structure of project_xy_to_x."""
        P = project_xy_to_x()
        expected = np.array([[1, 0, 0],
                            [0, 0, 1]])
        np.testing.assert_array_equal(P, expected)

    def test_project_xy_to_x_transform_point(self):
        """Test projecting a 2D point to x-axis."""
        P = project_xy_to_x()
        point = np.array([10, 20, 1])
        
        result = P @ point
        
        expected = np.array([10, 1])  # Keep only x
        np.testing.assert_array_equal(result, expected)

    def test_project_xy_to_x_transform_vector(self):
        """Test projecting a 2D vector to x-axis."""
        P = project_xy_to_x()
        vector = np.array([5, 7, 0])
        
        result = P @ vector
        
        expected = np.array([5, 0])  # Keep only x component
        np.testing.assert_array_equal(result, expected)


class TestProjectXYToY:
    """Tests for the project_xy_to_y function."""

    def test_project_xy_to_y_shape(self):
        """Test that project_xy_to_y returns correct shape."""
        P = project_xy_to_y()
        assert P.shape == (2, 3)

    def test_project_xy_to_y_transform_point(self):
        """Test projecting a 2D point to y-axis."""
        P = project_xy_to_y()
        point = np.array([10, 20, 1])
        
        result = P @ point
        
        expected = np.array([20, 1])  # Keep only y
        np.testing.assert_array_equal(result, expected)

    def test_project_xy_to_y_transform_vector(self):
        """Test projecting a 2D vector to y-axis."""
        P = project_xy_to_y()
        vector = np.array([5, 7, 0])
        
        result = P @ vector
        
        expected = np.array([7, 0])  # Keep only y component
        np.testing.assert_array_equal(result, expected)


class TestProjectXYZToXY:
    """Tests for the project_xyz_to_xy function."""

    def test_project_xyz_to_xy_shape(self):
        """Test that project_xyz_to_xy returns correct shape."""
        P = project_xyz_to_xy()
        assert P.shape == (3, 4)

    def test_project_xyz_to_xy_matrix_structure(self):
        """Test the matrix structure of project_xyz_to_xy."""
        P = project_xyz_to_xy()
        expected = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(P, expected)

    def test_project_xyz_to_xy_transform_point(self):
        """Test projecting a 3D point to xy-plane."""
        P = project_xyz_to_xy()
        point = np.array([5, 10, 15, 1])
        
        result = P @ point
        
        expected = np.array([5, 10, 1])  # Remove z
        np.testing.assert_array_equal(result, expected)

    def test_project_xyz_to_xy_transform_vector(self):
        """Test projecting a 3D vector to xy-plane."""
        P = project_xyz_to_xy()
        vector = np.array([1, 2, 3, 0])
        
        result = P @ vector
        
        expected = np.array([1, 2, 0])  # Remove z component
        np.testing.assert_array_equal(result, expected)


class TestProjectXYZToXZ:
    """Tests for the project_xyz_to_xz function."""

    def test_project_xyz_to_xz_shape(self):
        """Test that project_xyz_to_xz returns correct shape."""
        P = project_xyz_to_xz()
        assert P.shape == (3, 4)

    def test_project_xyz_to_xz_transform_point(self):
        """Test projecting a 3D point to xz-plane."""
        P = project_xyz_to_xz()
        point = np.array([5, 10, 15, 1])
        
        result = P @ point
        
        expected = np.array([5, 15, 1])  # Remove y, keep x and z
        np.testing.assert_array_equal(result, expected)

    def test_project_xyz_to_xz_transform_vector(self):
        """Test projecting a 3D vector to xz-plane."""
        P = project_xyz_to_xz()
        vector = np.array([1, 2, 3, 0])
        
        result = P @ vector
        
        expected = np.array([1, 3, 0])  # Remove y component
        np.testing.assert_array_equal(result, expected)


class TestProjectXYZToYZ:
    """Tests for the project_xyz_to_yz function."""

    def test_project_xyz_to_yz_shape(self):
        """Test that project_xyz_to_yz returns correct shape."""
        P = project_xyz_to_yz()
        assert P.shape == (3, 4)

    def test_project_xyz_to_yz_transform_point(self):
        """Test projecting a 3D point to yz-plane."""
        P = project_xyz_to_yz()
        point = np.array([5, 10, 15, 1])
        
        result = P @ point
        
        expected = np.array([10, 15, 1])  # Remove x, keep y and z
        np.testing.assert_array_equal(result, expected)

    def test_project_xyz_to_yz_transform_vector(self):
        """Test projecting a 3D vector to yz-plane."""
        P = project_xyz_to_yz()
        vector = np.array([1, 2, 3, 0])
        
        result = P @ vector
        
        expected = np.array([2, 3, 0])  # Remove x component
        np.testing.assert_array_equal(result, expected)


class TestProjectXYZToX:
    """Tests for the project_xyz_to_x function."""

    def test_project_xyz_to_x_shape(self):
        """Test that project_xyz_to_x returns correct shape."""
        P = project_xyz_to_x()
        assert P.shape == (2, 4)

    def test_project_xyz_to_x_transform_point(self):
        """Test projecting a 3D point to x-axis."""
        P = project_xyz_to_x()
        point = np.array([5, 10, 15, 1])
        
        result = P @ point
        
        expected = np.array([5, 1])  # Keep only x
        np.testing.assert_array_equal(result, expected)

    def test_project_xyz_to_x_transform_vector(self):
        """Test projecting a 3D vector to x-axis."""
        P = project_xyz_to_x()
        vector = np.array([1, 2, 3, 0])
        
        result = P @ vector
        
        expected = np.array([1, 0])  # Keep only x component
        np.testing.assert_array_equal(result, expected)


class TestProjectXYZToY:
    """Tests for the project_xyz_to_y function."""

    def test_project_xyz_to_y_shape(self):
        """Test that project_xyz_to_y returns correct shape."""
        P = project_xyz_to_y()
        assert P.shape == (2, 4)

    def test_project_xyz_to_y_transform_point(self):
        """Test projecting a 3D point to y-axis."""
        P = project_xyz_to_y()
        point = np.array([5, 10, 15, 1])
        
        result = P @ point
        
        expected = np.array([10, 1])  # Keep only y
        np.testing.assert_array_equal(result, expected)

    def test_project_xyz_to_y_transform_vector(self):
        """Test projecting a 3D vector to y-axis."""
        P = project_xyz_to_y()
        vector = np.array([1, 2, 3, 0])
        
        result = P @ vector
        
        expected = np.array([2, 0])  # Keep only y component
        np.testing.assert_array_equal(result, expected)


class TestProjectXYZToZ:
    """Tests for the project_xyz_to_z function."""

    def test_project_xyz_to_z_shape(self):
        """Test that project_xyz_to_z returns correct shape."""
        P = project_xyz_to_z()
        assert P.shape == (2, 4)

    def test_project_xyz_to_z_transform_point(self):
        """Test projecting a 3D point to z-axis."""
        P = project_xyz_to_z()
        point = np.array([5, 10, 15, 1])
        
        result = P @ point
        
        expected = np.array([15, 1])  # Keep only z
        np.testing.assert_array_equal(result, expected)

    def test_project_xyz_to_z_transform_vector(self):
        """Test projecting a 3D vector to z-axis."""
        P = project_xyz_to_z()
        vector = np.array([1, 2, 3, 0])
        
        result = P @ vector
        
        expected = np.array([3, 0])  # Keep only z component
        np.testing.assert_array_equal(result, expected)


class TestAugmentDim:
    """Tests for the augment_dim function."""

    def test_augment_dim_shape_1d(self):
        """Test augment_dim shape for 1D."""
        A = augment_dim(1)
        
        # Should be (initial_dim + 2) x (initial_dim + 1) = 3x2
        assert A.shape == (3, 2)

    def test_augment_dim_shape_2d(self):
        """Test augment_dim shape for 2D."""
        A = augment_dim(2)
        
        # Should be (initial_dim + 2) x (initial_dim + 1) = 4x3
        assert A.shape == (4, 3)

    def test_augment_dim_shape_3d(self):
        """Test augment_dim shape for 3D."""
        A = augment_dim(3)
        
        # Should be (initial_dim + 2) x (initial_dim + 1) = 5x4
        assert A.shape == (5, 4)

    def test_augment_dim_structure_1d(self):
        """Test the structure of augment_dim matrix for 1D."""
        A = augment_dim(1)
        
        # Should be 3x2 matrix with identity structure but middle column removed
        expected = np.array([[1, 0],
                            [0, 0],
                            [0, 1]])
        np.testing.assert_array_equal(A, expected)

    def test_augment_dim_structure_2d(self):
        """Test the structure of augment_dim matrix for 2D."""
        A = augment_dim(2)
        
        # Should be 4x3 matrix - identity with middle column (index 2) removed
        expected = np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0],
                            [0, 0, 1]])
        np.testing.assert_array_equal(A, expected)

    def test_augment_dim_structure_3d(self):
        """Test the structure of augment_dim matrix for 3D."""
        A = augment_dim(3)
        
        # Should be 5x4 matrix - identity with middle column (index 3) removed
        expected = np.array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 0],
                            [0, 0, 0, 1]])
        np.testing.assert_array_equal(A, expected)

    def test_augment_dim_transform_1d_point(self):
        """Test augmenting a 1D point in homogeneous coords."""
        A = augment_dim(1)
        point_1d = np.array([5, 1])  # 1D point in homogeneous coords
        
        result = A @ point_1d
        
        # Should add a zero dimension in the middle
        expected = np.array([5, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_augment_dim_transform_2d_point(self):
        """Test augmenting a 2D point in homogeneous coords."""
        A = augment_dim(2)
        point_2d = np.array([3, 7, 1])  # 2D point in homogeneous coords
        
        result = A @ point_2d
        
        # Should add a zero dimension in the middle
        expected = np.array([3, 7, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_augment_dim_transform_3d_point(self):
        """Test augmenting a 3D point in homogeneous coords."""
        A = augment_dim(3)
        point_3d = np.array([2, 4, 6, 1])  # 3D point in homogeneous coords
        
        result = A @ point_3d
        
        # Should add a zero dimension in the middle
        expected = np.array([2, 4, 6, 0, 1])
        np.testing.assert_array_equal(result, expected)

    def test_augment_dim_vector_1d(self):
        """Test augmenting a 1D vector."""
        A = augment_dim(1)
        vector = np.array([5, 0])  # 1D vector (w=0)
        
        result = A @ vector
        
        # Should add a zero dimension
        expected = np.array([5, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_augment_dim_vector_2d(self):
        """Test augmenting a 2D vector."""
        A = augment_dim(2)
        vector = np.array([1, 2, 0])  # 2D vector (w=0)
        
        result = A @ vector
        
        # Should add a zero dimension
        expected = np.array([1, 2, 0, 0])
        np.testing.assert_array_equal(result, expected)

    def test_augment_dim_multiple_points(self):
        """Test augmenting multiple points at once."""
        A = augment_dim(2)
        
        # Multiple 2D points in homogeneous coords (3xN array)
        points = np.array([[1, 2, 3],
                          [4, 5, 6],
                          [1, 1, 1]])
        
        result = A @ points
        
        # Should add a row of zeros before the last row
        expected = np.array([[1, 2, 3],
                            [4, 5, 6],
                            [0, 0, 0],
                            [1, 1, 1]])
        np.testing.assert_array_equal(result, expected)

    def test_augment_dim_inserts_dimension_before_homogeneous(self):
        """Test that augment_dim inserts the new dimension before the homogeneous coordinate."""
        A = augment_dim(2)
        
        # The structure should have zeros in the (initial_dim) row
        assert A[2, 0] == 0
        assert A[2, 1] == 0
        assert A[2, 2] == 0
        
        # The last row should be [0, 0, 1] (homogeneous coordinate preserved)
        assert A[3, 0] == 0
        assert A[3, 1] == 0
        assert A[3, 2] == 1

    def test_augment_dim_composition_with_reduce_dim(self):
        """Test composing augment_dim with reduce_dim."""
        A = augment_dim(2)  # 4x3
        R = reduce_dim(3)   # 3x4
        
        # R @ A should give identity (reducing then augmenting)
        composed = R @ A
        
        expected = np.eye(3)
        np.testing.assert_array_equal(composed, expected)

    def test_augment_dim_zero_dimension_location(self):
        """Test that the zero dimension is inserted at the correct position."""
        A = augment_dim(1)
        
        # For initial_dim=1, the zero should be at index 1 (row 1)
        assert A[1, 0] == 0
        assert A[1, 1] == 0
        
        A = augment_dim(2)
        # For initial_dim=2, the zero should be at index 2 (row 2)
        assert A[2, 0] == 0
        assert A[2, 1] == 0
        assert A[2, 2] == 0
