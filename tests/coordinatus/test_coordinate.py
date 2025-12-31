"""Unit tests for the Coordinate, Point, and Vector classes."""

import numpy as np
from coordinatus.coordinate import Coordinate, Point, Vector, transform_coordinate
from coordinatus.frame import Frame
from coordinatus.types import CoordinateType
from coordinatus.transforms import translate2D, rotate2D, scale2D


class TestTransformCoordinate:
    """Tests for the transform_coordinate utility function."""

    def test_transform_point_translation(self):
        """Test transforming a point with translation."""
        transform = translate2D(5, 3)
        coords = np.array([1, 2])
        result = transform_coordinate(transform, coords, CoordinateType.POINT)
        
        # Point should be translated
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_translation(self):
        """Test that translation doesn't affect vectors."""
        transform = translate2D(5, 3)
        coords = np.array([1, 2])
        result = transform_coordinate(transform, coords, CoordinateType.VECTOR)
        
        # Vector should be unchanged by translation
        expected = np.array([1, 2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_point_rotation(self):
        """Test transforming a point with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([1, 0])
        result = transform_coordinate(transform, coords, CoordinateType.POINT)
        
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_rotation(self):
        """Test transforming a vector with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([1, 0])
        result = transform_coordinate(transform, coords, CoordinateType.VECTOR)
        
        # Vector should also rotate
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_point_scale(self):
        """Test transforming a point with scaling."""
        transform = scale2D(2, 3)
        coords = np.array([4, 5])
        result = transform_coordinate(transform, coords, CoordinateType.POINT)
        
        expected = np.array([8, 15])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_scale(self):
        """Test transforming a vector with scaling."""
        transform = scale2D(2, 3)
        coords = np.array([4, 5])
        result = transform_coordinate(transform, coords, CoordinateType.VECTOR)
        
        # Vector should also be scaled
        expected = np.array([8, 15])
        np.testing.assert_array_almost_equal(result, expected)


class TestCoordinateInit:
    """Tests for Coordinate initialization."""

    def test_coordinate_init_no_system(self):
        """Test creating a coordinate without a frame creates identity frame."""
        coord = Coordinate(
            coordinate_type=CoordinateType.POINT,
            local_coords=np.array([1, 2]),
            frame=None
        )
        
        assert coord.coordinate_type == CoordinateType.POINT
        np.testing.assert_array_equal(coord.local_coords, [1, 2])
        assert coord.frame is not None
        assert coord.frame.parent is None
        np.testing.assert_array_equal(coord.frame.transform, np.eye(3))

    def test_coordinate_init_with_system(self):
        """Test creating a coordinate with a custom frame."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        coord = Coordinate(
            coordinate_type=CoordinateType.VECTOR,
            local_coords=np.array([2, 3]),
            frame=frame
        )
        
        assert coord.coordinate_type == CoordinateType.VECTOR
        np.testing.assert_array_equal(coord.local_coords, [2, 3])
        assert coord.frame is frame


class TestPointInit:
    """Tests for Point initialization."""

    def test_point_init_basic(self):
        """Test creating a basic point."""
        point = Point(local_coords=np.array([3, 4]))
        
        assert point.coordinate_type == CoordinateType.POINT
        np.testing.assert_array_equal(point.local_coords, [3, 4])
        assert point.frame is not None

    def test_point_init_with_system(self):
        """Test creating a point with a custom frame."""
        frame = Frame(transform=translate2D(10, 20), parent=None)
        point = Point(local_coords=np.array([1, 1]), frame=frame)
        
        assert point.coordinate_type == CoordinateType.POINT
        assert point.frame is frame


class TestVectorInit:
    """Tests for Vector initialization."""

    def test_vector_init_basic(self):
        """Test creating a basic vector."""
        vector = Vector(local_coords=np.array([1, 0]))
        
        assert vector.coordinate_type == CoordinateType.VECTOR
        np.testing.assert_array_equal(vector.local_coords, [1, 0])
        assert vector.frame is not None

    def test_vector_init_with_system(self):
        """Test creating a vector with a custom frame."""
        frame = Frame(transform=rotate2D(np.pi / 4), parent=None)
        vector = Vector(local_coords=np.array([1, 1]), frame=frame)
        
        assert vector.coordinate_type == CoordinateType.VECTOR
        assert vector.frame is frame


class TestCoordinateToAbsolute:
    """Tests for converting coordinates to absolute space."""

    def test_to_global_no_parent(self):
        """Test to_absolute when frame has no parent (root frame)."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        point = Point(local_coords=np.array([1, 2]), frame=frame)
        
        result = point.to_absolute()
        
        # Point at (1, 2) in frame with translation (5, 3)
        # Absolute coordinates should be (6, 5)
        np.testing.assert_array_almost_equal(result.local_coords, [6, 5])
        # Result should be in identity/absolute frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))

    def test_to_global_one_level(self):
        """Test to_absolute with one parent level."""
        parent = Frame(transform=translate2D(10, 5), parent=None)
        child_frame = Frame(transform=translate2D(3, 2), parent=parent)
        point = Point(local_coords=np.array([1, 1]), frame=child_frame)
        
        result = point.to_absolute()
        
        # Point at (1,1) in child
        # Child is at (3,2) relative to parent
        # So point is at (4,3) in parent
        # Parent is at (10,5) in absolute
        # So point is at (14,8) in absolute
        np.testing.assert_array_almost_equal(result.local_coords, [14, 8])
        # Result should be in absolute/identity frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))

    def test_to_global_multiple_levels(self):
        """Test to_absolute with nested hierarchy."""
        root = Frame(transform=translate2D(100, 100), parent=None)
        middle = Frame(transform=translate2D(10, 10), parent=root)
        leaf = Frame(transform=translate2D(1, 1), parent=middle)
        
        point = Point(local_coords=np.array([0, 0]), frame=leaf)
        result = point.to_absolute()
        
        # Point at origin in leaf frame
        # Leaf is at (1, 1) in middle, middle is at (10, 10) in root, root is at (100, 100) in absolute
        # So point is at (111, 111) in absolute coordinates
        np.testing.assert_array_almost_equal(result.local_coords, [111, 111])
        # Result should be in absolute/identity frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))

    def test_to_global_with_rotation(self):
        """Test to_absolute with rotated coordinate frame."""
        parent = Frame(transform=np.eye(3), parent=None)
        # Child rotated 90 degrees
        child = Frame(transform=rotate2D(np.pi / 2), parent=parent)
        
        # Point at (1, 0) in child frame
        point = Point(local_coords=np.array([1, 0]), frame=child)
        result = point.to_absolute()
        
        # After 90 degree rotation, (1, 0) becomes (0, 1) in absolute
        np.testing.assert_array_almost_equal(result.local_coords, [0, 1])
        # Result should be in absolute/identity frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))

    def test_to_global_vector_ignores_translation(self):
        """Test that vectors ignore translation when going to absolute."""
        parent = Frame(transform=translate2D(10, 20), parent=None)
        child = Frame(transform=translate2D(5, 5), parent=parent)
        
        vector = Vector(local_coords=np.array([1, 0]), frame=child)
        result = vector.to_absolute()
        
        # Vector should maintain direction, translation ignored (weight=0)
        # Vector (1, 0) should remain (1, 0) in absolute since translations don't affect vectors
        np.testing.assert_array_almost_equal(result.local_coords, [1, 0])
        # Result should be in absolute/identity frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))


class TestCoordinateToFrame:
    """Tests for converting coordinates between frames."""

    def test_to_system_same_system(self):
        """Test converting to the same frame."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        point = Point(local_coords=np.array([1, 2]), frame=frame)
        
        result = point.relative_to(frame)
        
        # Should have same coordinates in same frame
        np.testing.assert_array_almost_equal(result.local_coords, [1, 2])
        assert result.frame is frame

    def test_to_system_siblings(self):
        """Test converting between sibling coordinate frames."""
        parent = Frame(transform=np.eye(3), parent=None)
        frame_a = Frame(transform=translate2D(5, 0), parent=parent)
        frame_b = Frame(transform=translate2D(0, 3), parent=parent)
        
        # Point at (0, 0) in frame A
        point = Point(local_coords=np.array([0, 0]), frame=frame_a)
        
        # Convert to frame B
        result = point.relative_to(frame_b)
        
        # Point is at (5, 0) globally
        # In frame B coords (which is at (0, 3)), that's (5, -3)
        np.testing.assert_array_almost_equal(result.local_coords, [5, -3])
        assert result.frame is frame_b

    def test_to_system_parent_to_child(self):
        """Test converting from parent to child frame."""
        parent = Frame(transform=translate2D(10, 5), parent=None)
        child = Frame(transform=translate2D(3, 2), parent=parent)
        
        point = Point(local_coords=np.array([0, 0]), frame=parent)
        result = point.relative_to(child)
        
        # Point at parent origin is at (10, 5) globally
        # Child origin is at (13, 7) globally
        # So parent origin in child coords is (-3, -2)
        np.testing.assert_array_almost_equal(result.local_coords, [-3, -2])
        assert result.frame is child

    def test_to_system_child_to_parent(self):
        """Test converting from child to parent frame."""
        parent = Frame(transform=translate2D(10, 5), parent=None)
        child = Frame(transform=translate2D(3, 2), parent=parent)
        
        point = Point(local_coords=np.array([0, 0]), frame=child)
        result = point.relative_to(parent)
        
        # Point at child origin (13, 7 globally) is at (3, 2) in parent
        np.testing.assert_array_almost_equal(result.local_coords, [3, 2])
        assert result.frame is parent

    def test_to_system_with_rotation(self):
        """Test converting between rotated frames."""
        frame_a = Frame(transform=np.eye(3), parent=None)
        frame_b = Frame(transform=rotate2D(np.pi / 2), parent=None)
        
        # Point at (1, 0) in frame A
        point = Point(local_coords=np.array([1, 0]), frame=frame_a)
        
        # Convert to frame B (rotated 90 degrees)
        result = point.relative_to(frame_b)
        
        # (1, 0) in A becomes (0, -1) in B (inverse rotation)
        np.testing.assert_array_almost_equal(result.local_coords, [0, -1])

    def test_to_system_vector_translation(self):
        """Test that vector conversion ignores translation."""
        frame_a = Frame(transform=translate2D(10, 5), parent=None)
        frame_b = Frame(transform=translate2D(20, 15), parent=None)
        
        vector = Vector(local_coords=np.array([1, 0]), frame=frame_a)
        result = vector.relative_to(frame_b)
        
        # Vector direction should be same regardless of translation
        np.testing.assert_array_almost_equal(result.local_coords, [1, 0])
        assert result.frame is frame_b

    def test_to_system_vector_rotation(self):
        """Test that vector conversion respects rotation."""
        frame_a = Frame(transform=np.eye(3), parent=None)
        frame_b = Frame(transform=rotate2D(np.pi / 2), parent=None)
        
        vector = Vector(local_coords=np.array([1, 0]), frame=frame_a)
        result = vector.relative_to(frame_b)
        
        # Vector should rotate
        np.testing.assert_array_almost_equal(result.local_coords, [0, -1])

    def test_to_system_complex_hierarchy(self):
        """Test conversion in complex hierarchy."""
        root = Frame(transform=np.eye(3), parent=None)
        branch_a = Frame(transform=translate2D(10, 0), parent=root)
        leaf_a = Frame(transform=rotate2D(np.pi / 4), parent=branch_a)
        
        branch_b = Frame(transform=translate2D(0, 10), parent=root)
        
        point = Point(local_coords=np.array([1, 0]), frame=leaf_a)
        result = point.relative_to(branch_b)
        
        # Point goes through: leaf_a -> branch_a -> root -> branch_b
        # In leaf_a: (1, 0)
        # After rotation (π/4): (√2/2, √2/2)
        # After translate (10, 0): (10 + √2/2, √2/2)
        # In branch_b (0, 10): (10 + √2/2, √2/2 - 10)
        sqrt2_over_2 = np.sqrt(2) / 2
        expected = np.array([10 + sqrt2_over_2, sqrt2_over_2 - 10])
        np.testing.assert_array_almost_equal(result.local_coords, expected)


class TestPointAndVectorBehavior:
    """Tests to verify Point and Vector behave differently with transformations."""

    def test_point_affected_by_translation(self):
        """Test that points are affected by translation."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        point = Point(local_coords=np.array([1, 2]), frame=frame)
        
        # When converting to identity frame, point should be translated
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = point.relative_to(identity_frame)
        
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_vector_unaffected_by_translation(self):
        """Test that vectors are unaffected by translation."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        vector = Vector(local_coords=np.array([1, 2]), frame=frame)
        
        # When converting to identity frame, vector should not be translated
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = vector.relative_to(identity_frame)
        
        expected = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_point_scaled_from_origin(self):
        """Test that points are scaled from origin."""
        frame = Frame(transform=scale2D(2, 2), parent=None)
        point = Point(local_coords=np.array([3, 4]), frame=frame)
        
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = point.relative_to(identity_frame)
        
        expected = np.array([6, 8])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_vector_scaled(self):
        """Test that vectors are also scaled."""
        frame = Frame(transform=scale2D(2, 2), parent=None)
        vector = Vector(local_coords=np.array([3, 4]), frame=frame)
        
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = vector.relative_to(identity_frame)
        
        expected = np.array([6, 8])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_point_and_vector_both_rotate(self):
        """Test that both points and vectors rotate the same way."""
        frame = Frame(transform=rotate2D(np.pi / 2), parent=None)
        identity_frame = Frame(transform=np.eye(3), parent=None)
        
        point = Point(local_coords=np.array([1, 0]), frame=frame)
        vector = Vector(local_coords=np.array([1, 0]), frame=frame)
        
        point_result = point.relative_to(identity_frame)
        vector_result = vector.relative_to(identity_frame)
        
        # Both should rotate the same
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(point_result.local_coords, expected)
        np.testing.assert_array_almost_equal(vector_result.local_coords, expected)


class TestDxNCoordinates:
    """Tests for DxN coordinate arrays (multiple points/vectors at once)."""

    def test_transform_coordinate_2xN_points_translation(self):
        """Test transforming multiple points (2xN array) with translation."""
        transform = translate2D(5, 3)
        # Three points: (1, 2), (3, 4), (5, 6)
        coords = np.array([[1, 3, 5], [2, 4, 6]])
        result = transform_coordinate(transform, coords, CoordinateType.POINT)
        
        # All points should be translated
        expected = np.array([[6, 8, 10], [5, 7, 9]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_coordinate_2xN_vectors_translation(self):
        """Test that translation doesn't affect multiple vectors (2xN array)."""
        transform = translate2D(5, 3)
        # Three vectors: (1, 2), (3, 4), (5, 6)
        coords = np.array([[1, 3, 5], [2, 4, 6]])
        result = transform_coordinate(transform, coords, CoordinateType.VECTOR)
        
        # Vectors should be unchanged by translation
        expected = np.array([[1, 3, 5], [2, 4, 6]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_coordinate_2xN_points_rotation(self):
        """Test rotating multiple points (2xN array)."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        # Two points: (1, 0), (0, 1)
        coords = np.array([[1, 0], [0, 1]])
        result = transform_coordinate(transform, coords, CoordinateType.POINT)
        
        # After 90 degree rotation: (1,0) -> (0,1), (0,1) -> (-1,0)
        expected = np.array([[0, -1], [1, 0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_coordinate_2xN_points_scale(self):
        """Test scaling multiple points (2xN array)."""
        transform = scale2D(2, 3)
        # Three points
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        result = transform_coordinate(transform, coords, CoordinateType.POINT)
        
        expected = np.array([[2, 4, 6], [12, 15, 18]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_point_init_2xN(self):
        """Test creating a Point with 2xN array."""
        coords = np.array([[1, 2, 3], [4, 5, 6]])
        point = Point(local_coords=coords)
        
        assert point.coordinate_type == CoordinateType.POINT
        np.testing.assert_array_equal(point.local_coords, coords)
        assert point.local_coords.shape == (2, 3)

    def test_vector_init_2xN(self):
        """Test creating a Vector with 2xN array."""
        coords = np.array([[1, 2], [3, 4]])
        vector = Vector(local_coords=coords)
        
        assert vector.coordinate_type == CoordinateType.VECTOR
        np.testing.assert_array_equal(vector.local_coords, coords)
        assert vector.local_coords.shape == (2, 2)

    def test_point_2xN_to_absolute(self):
        """Test converting multiple points to absolute coordinates."""
        frame = Frame(transform=translate2D(10, 5), parent=None)
        # Two points: (0, 0) and (1, 1)
        coords = np.array([[0, 1], [0, 1]])
        point = Point(local_coords=coords, frame=frame)
        
        result = point.to_absolute()
        
        # Points should be translated
        expected = np.array([[10, 11], [5, 6]])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_vector_2xN_to_absolute(self):
        """Test converting multiple vectors to absolute coordinates."""
        frame = Frame(transform=translate2D(10, 5), parent=None)
        # Two vectors: (1, 0) and (0, 1)
        coords = np.array([[1, 0], [0, 1]])
        vector = Vector(local_coords=coords, frame=frame)
        
        result = vector.to_absolute()
        
        # Vectors should not be translated
        expected = np.array([[1, 0], [0, 1]])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_point_2xN_relative_to(self):
        """Test converting multiple points between frames."""
        frame_a = Frame(transform=translate2D(5, 0), parent=None)
        frame_b = Frame(transform=translate2D(0, 3), parent=None)
        
        # Two points in frame_a: (0, 0) and (1, 1)
        coords = np.array([[0, 1], [0, 1]])
        point = Point(local_coords=coords, frame=frame_a)
        
        result = point.relative_to(frame_b)
        
        # Point (0, 0) in frame_a is at (5, 0) globally, which is (5, -3) in frame_b
        # Point (1, 1) in frame_a is at (6, 1) globally, which is (6, -2) in frame_b
        expected = np.array([[5, 6], [-3, -2]])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_point_2xN_with_rotation(self):
        """Test multiple points with rotation."""
        frame = Frame(transform=rotate2D(np.pi / 2), parent=None)
        # Three points along x-axis: (1,0), (2,0), (3,0)
        coords = np.array([[1, 2, 3], [0, 0, 0]])
        point = Point(local_coords=coords, frame=frame)
        
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = point.relative_to(identity_frame)
        
        # After 90 degree rotation, they should be along y-axis: (0,1), (0,2), (0,3)
        expected = np.array([[0, 0, 0], [1, 2, 3]])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_single_point_still_works(self):
        """Test that single point (2,) arrays still work (backwards compatibility)."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        coords = np.array([1, 2])
        point = Point(local_coords=coords, frame=frame)
        
        result = point.to_absolute()
        
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result.local_coords, expected)
