"""Unit tests for the Coordinate, Point, and Vector classes."""

import numpy as np
from coordinatus.coordinate import Coordinate, Point, Vector, transform_coordinate
from coordinatus.frame import Frame
from coordinatus.types import CoordinateKind
from coordinatus.transforms import translate2D, rotate2D, scale2D


class TestTransformCoordinate:
    """Tests for the transform_coordinate utility function."""

    def test_transform_point_translation(self):
        """Test transforming a point with translation."""
        transform = translate2D(5, 3)
        coords = np.array([1, 2])
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        # Point should be translated
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_translation(self):
        """Test that translation doesn't affect vectors."""
        transform = translate2D(5, 3)
        coords = np.array([1, 2])
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vector should be unchanged by translation
        expected = np.array([1, 2])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_point_rotation(self):
        """Test transforming a point with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([1, 0])
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_rotation(self):
        """Test transforming a vector with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([1, 0])
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vector should also rotate
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_point_scale(self):
        """Test transforming a point with scaling."""
        transform = scale2D(2, 3)
        coords = np.array([4, 5])
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([8, 15])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_vector_scale(self):
        """Test transforming a vector with scaling."""
        transform = scale2D(2, 3)
        coords = np.array([4, 5])
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vector should also be scaled
        expected = np.array([8, 15])
        np.testing.assert_array_almost_equal(result, expected)


class TestCoordinateInit:
    """Tests for Coordinate initialization."""

    def test_coordinate_init_no_system(self):
        """Test creating a coordinate without a frame creates identity frame."""
        coord = Coordinate(
            kind=CoordinateKind.POINT,
            coords=np.array([1, 2]),
            frame=None
        )
        
        assert coord.kind == CoordinateKind.POINT
        np.testing.assert_array_equal(coord.coords, [1, 2])
        assert coord.frame is not None
        assert coord.frame.parent is None
        np.testing.assert_array_equal(coord.frame.transform, np.eye(3))

    def test_coordinate_init_with_system(self):
        """Test creating a coordinate with a custom frame."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        coord = Coordinate(
            kind=CoordinateKind.VECTOR,
            coords=np.array([2, 3]),
            frame=frame
        )
        
        assert coord.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(coord.coords, [2, 3])
        assert coord.frame is frame


class TestPointInit:
    """Tests for Point initialization."""

    def test_point_init_basic(self):
        """Test creating a basic point."""
        point = Point(coords=np.array([3, 4]))
        
        assert point.kind == CoordinateKind.POINT
        np.testing.assert_array_equal(point.coords, [3, 4])
        assert point.frame is not None

    def test_point_init_with_system(self):
        """Test creating a point with a custom frame."""
        frame = Frame(transform=translate2D(10, 20), parent=None)
        point = Point(coords=np.array([1, 1]), frame=frame)
        
        assert point.kind == CoordinateKind.POINT
        assert point.frame is frame

    def test_point_init_with_list(self):
        """Test creating a point with a list."""
        point = Point(coords=[3, 4])
        
        assert point.kind == CoordinateKind.POINT
        np.testing.assert_array_equal(point.coords, [3, 4])
        assert isinstance(point.coords, np.ndarray)

    def test_point_init_with_tuple(self):
        """Test creating a point with a tuple."""
        point = Point(coords=(3, 4))
        
        assert point.kind == CoordinateKind.POINT
        np.testing.assert_array_equal(point.coords, [3, 4])
        assert isinstance(point.coords, np.ndarray)


class TestVectorInit:
    """Tests for Vector initialization."""

    def test_vector_init_basic(self):
        """Test creating a basic vector."""
        vector = Vector(coords=np.array([1, 0]))
        
        assert vector.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(vector.coords, [1, 0])
        assert vector.frame is not None

    def test_vector_init_with_system(self):
        """Test creating a vector with a custom frame."""
        frame = Frame(transform=rotate2D(np.pi / 4), parent=None)
        vector = Vector(coords=np.array([1, 1]), frame=frame)
        
        assert vector.kind == CoordinateKind.VECTOR
        assert vector.frame is frame

    def test_vector_init_with_list(self):
        """Test creating a vector with a list."""
        vector = Vector(coords=[1, 0])
        
        assert vector.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(vector.coords, [1, 0])
        assert isinstance(vector.coords, np.ndarray)

    def test_vector_init_with_tuple(self):
        """Test creating a vector with a tuple."""
        vector = Vector(coords=(1, 0))
        
        assert vector.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(vector.coords, [1, 0])
        assert isinstance(vector.coords, np.ndarray)


class TestCoordinateToAbsolute:
    """Tests for converting coordinates to absolute space."""

    def test_to_global_no_parent(self):
        """Test to_absolute when frame has no parent (root frame)."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        point = Point(coords=np.array([1, 2]), frame=frame)
        
        result = point.to_absolute()
        
        # Point at (1, 2) in frame with translation (5, 3)
        # Absolute coordinates should be (6, 5)
        np.testing.assert_array_almost_equal(result.coords, [6, 5])
        # Result should be in identity/absolute frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))

    def test_to_global_one_level(self):
        """Test to_absolute with one parent level."""
        parent = Frame(transform=translate2D(10, 5), parent=None)
        child_frame = Frame(transform=translate2D(3, 2), parent=parent)
        point = Point(coords=np.array([1, 1]), frame=child_frame)
        
        result = point.to_absolute()
        
        # Point at (1,1) in child
        # Child is at (3,2) relative to parent
        # So point is at (4,3) in parent
        # Parent is at (10,5) in absolute
        # So point is at (14,8) in absolute
        np.testing.assert_array_almost_equal(result.coords, [14, 8])
        # Result should be in absolute/identity frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))

    def test_to_global_multiple_levels(self):
        """Test to_absolute with nested hierarchy."""
        root = Frame(transform=translate2D(100, 100), parent=None)
        middle = Frame(transform=translate2D(10, 10), parent=root)
        leaf = Frame(transform=translate2D(1, 1), parent=middle)
        
        point = Point(coords=np.array([0, 0]), frame=leaf)
        result = point.to_absolute()
        
        # Point at origin in leaf frame
        # Leaf is at (1, 1) in middle, middle is at (10, 10) in root, root is at (100, 100) in absolute
        # So point is at (111, 111) in absolute coordinates
        np.testing.assert_array_almost_equal(result.coords, [111, 111])
        # Result should be in absolute/identity frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))

    def test_to_global_with_rotation(self):
        """Test to_absolute with rotated coordinate frame."""
        parent = Frame(transform=np.eye(3), parent=None)
        # Child rotated 90 degrees
        child = Frame(transform=rotate2D(np.pi / 2), parent=parent)
        
        # Point at (1, 0) in child frame
        point = Point(coords=np.array([1, 0]), frame=child)
        result = point.to_absolute()
        
        # After 90 degree rotation, (1, 0) becomes (0, 1) in absolute
        np.testing.assert_array_almost_equal(result.coords, [0, 1])
        # Result should be in absolute/identity frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))

    def test_to_global_vector_ignores_translation(self):
        """Test that vectors ignore translation when going to absolute."""
        parent = Frame(transform=translate2D(10, 20), parent=None)
        child = Frame(transform=translate2D(5, 5), parent=parent)
        
        vector = Vector(coords=np.array([1, 0]), frame=child)
        result = vector.to_absolute()
        
        # Vector should maintain direction, translation ignored (weight=0)
        # Vector (1, 0) should remain (1, 0) in absolute since translations don't affect vectors
        np.testing.assert_array_almost_equal(result.coords, [1, 0])
        # Result should be in absolute/identity frame
        assert result.frame.parent is None
        np.testing.assert_array_almost_equal(result.frame.transform, np.eye(3))


class TestCoordinateToFrame:
    """Tests for converting coordinates between frames."""

    def test_to_system_same_system(self):
        """Test converting to the same frame."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        point = Point(coords=np.array([1, 2]), frame=frame)
        
        result = point.relative_to(frame)
        
        # Should have same coordinates in same frame
        np.testing.assert_array_almost_equal(result.coords, [1, 2])
        assert result.frame is frame

    def test_to_system_siblings(self):
        """Test converting between sibling coordinate frames."""
        parent = Frame(transform=np.eye(3), parent=None)
        frame_a = Frame(transform=translate2D(5, 0), parent=parent)
        frame_b = Frame(transform=translate2D(0, 3), parent=parent)
        
        # Point at (0, 0) in frame A
        point = Point(coords=np.array([0, 0]), frame=frame_a)
        
        # Convert to frame B
        result = point.relative_to(frame_b)
        
        # Point is at (5, 0) globally
        # In frame B coords (which is at (0, 3)), that's (5, -3)
        np.testing.assert_array_almost_equal(result.coords, [5, -3])
        assert result.frame is frame_b

    def test_to_system_parent_to_child(self):
        """Test converting from parent to child frame."""
        parent = Frame(transform=translate2D(10, 5), parent=None)
        child = Frame(transform=translate2D(3, 2), parent=parent)
        
        point = Point(coords=np.array([0, 0]), frame=parent)
        result = point.relative_to(child)
        
        # Point at parent origin is at (10, 5) globally
        # Child origin is at (13, 7) globally
        # So parent origin in child coords is (-3, -2)
        np.testing.assert_array_almost_equal(result.coords, [-3, -2])
        assert result.frame is child

    def test_to_system_child_to_parent(self):
        """Test converting from child to parent frame."""
        parent = Frame(transform=translate2D(10, 5), parent=None)
        child = Frame(transform=translate2D(3, 2), parent=parent)
        
        point = Point(coords=np.array([0, 0]), frame=child)
        result = point.relative_to(parent)
        
        # Point at child origin (13, 7 globally) is at (3, 2) in parent
        np.testing.assert_array_almost_equal(result.coords, [3, 2])
        assert result.frame is parent

    def test_to_system_with_rotation(self):
        """Test converting between rotated frames."""
        frame_a = Frame(transform=np.eye(3), parent=None)
        frame_b = Frame(transform=rotate2D(np.pi / 2), parent=None)
        
        # Point at (1, 0) in frame A
        point = Point(coords=np.array([1, 0]), frame=frame_a)
        
        # Convert to frame B (rotated 90 degrees)
        result = point.relative_to(frame_b)
        
        # (1, 0) in A becomes (0, -1) in B (inverse rotation)
        np.testing.assert_array_almost_equal(result.coords, [0, -1])

    def test_to_system_vector_translation(self):
        """Test that vector conversion ignores translation."""
        frame_a = Frame(transform=translate2D(10, 5), parent=None)
        frame_b = Frame(transform=translate2D(20, 15), parent=None)
        
        vector = Vector(coords=np.array([1, 0]), frame=frame_a)
        result = vector.relative_to(frame_b)
        
        # Vector direction should be same regardless of translation
        np.testing.assert_array_almost_equal(result.coords, [1, 0])
        assert result.frame is frame_b

    def test_to_system_vector_rotation(self):
        """Test that vector conversion respects rotation."""
        frame_a = Frame(transform=np.eye(3), parent=None)
        frame_b = Frame(transform=rotate2D(np.pi / 2), parent=None)
        
        vector = Vector(coords=np.array([1, 0]), frame=frame_a)
        result = vector.relative_to(frame_b)
        
        # Vector should rotate
        np.testing.assert_array_almost_equal(result.coords, [0, -1])

    def test_to_system_complex_hierarchy(self):
        """Test conversion in complex hierarchy."""
        root = Frame(transform=np.eye(3), parent=None)
        branch_a = Frame(transform=translate2D(10, 0), parent=root)
        leaf_a = Frame(transform=rotate2D(np.pi / 4), parent=branch_a)
        
        branch_b = Frame(transform=translate2D(0, 10), parent=root)
        
        point = Point(coords=np.array([1, 0]), frame=leaf_a)
        result = point.relative_to(branch_b)
        
        # Point goes through: leaf_a -> branch_a -> root -> branch_b
        # In leaf_a: (1, 0)
        # After rotation (π/4): (√2/2, √2/2)
        # After translate (10, 0): (10 + √2/2, √2/2)
        # In branch_b (0, 10): (10 + √2/2, √2/2 - 10)
        sqrt2_over_2 = np.sqrt(2) / 2
        expected = np.array([10 + sqrt2_over_2, sqrt2_over_2 - 10])
        np.testing.assert_array_almost_equal(result.coords, expected)


class TestDxNArraySupport:
    """Tests for DxN array support - working with multiple points/vectors."""

    def test_transform_multiple_points_translation(self):
        """Test transforming multiple points with translation."""
        transform = translate2D(5, 3)
        # 2D array with 3 points: [x1, x2, x3] and [y1, y2, y3]
        coords = np.array([[1, 2, 3], [2, 4, 6]])  # shape (2, 3)
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        # All points should be translated
        expected = np.array([[6, 7, 8], [5, 7, 9]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_multiple_vectors_translation(self):
        """Test that translation doesn't affect multiple vectors."""
        transform = translate2D(5, 3)
        coords = np.array([[1, 2, 3], [2, 4, 6]])  # shape (2, 3)
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vectors should be unchanged by translation
        expected = np.array([[1, 2, 3], [2, 4, 6]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_multiple_points_rotation(self):
        """Test transforming multiple points with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([[1, 0, 1], [0, 1, 1]])  # 3 points: (1,0), (0,1), (1,1)
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        # After 90 degree rotation: (1,0)->(0,1), (0,1)->(-1,0), (1,1)->(-1,1)
        expected = np.array([[0, -1, -1], [1, 0, 1]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_multiple_vectors_rotation(self):
        """Test transforming multiple vectors with rotation."""
        transform = rotate2D(np.pi / 2)  # 90 degrees
        coords = np.array([[1, 0], [0, 1]])  # 2 vectors: (1,0) and (0,1)
        result = transform_coordinate(transform, coords, CoordinateKind.VECTOR)
        
        # Vectors should also rotate
        expected = np.array([[0, -1], [1, 0]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_transform_multiple_points_scale(self):
        """Test transforming multiple points with scaling."""
        transform = scale2D(2, 3)
        coords = np.array([[1, 2], [3, 4]])  # 2 points: (1,3) and (2,4)
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([[2, 4], [9, 12]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_3d_points_translation(self):
        """Test transforming 3D points (though transform is 2D embedded in 3D)."""
        # For 3D we'd need a 4x4 transform, but let's test the concept
        # This test documents current 2D-only behavior
        transform = translate2D(5, 3)
        coords = np.array([[1, 2], [3, 4]])  # 2 points in 2D
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([[6, 7], [6, 7]])
        np.testing.assert_array_almost_equal(result, expected)

    def test_point_with_multiple_coordinates(self):
        """Test Point class with multiple coordinates."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        # Create a point with multiple coordinates
        coords = np.array([[1, 2, 3], [2, 4, 6]])  # 3 points
        point = Point(coords=coords, frame=frame)
        
        result = point.to_absolute()
        
        # All points should be translated
        expected = np.array([[6, 7, 8], [5, 7, 9]])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_vector_with_multiple_coordinates(self):
        """Test Vector class with multiple coordinates."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        # Create a vector with multiple coordinates
        coords = np.array([[1, 2, 3], [2, 4, 6]])  # 3 vectors
        vector = Vector(coords=coords, frame=frame)
        
        result = vector.to_absolute()
        
        # Vectors should be unchanged by translation
        expected = np.array([[1, 2, 3], [2, 4, 6]])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_single_point_still_works(self):
        """Test that single point (2,) still works after DxN implementation."""
        transform = translate2D(5, 3)
        coords = np.array([1, 2])  # shape (2,)
        result = transform_coordinate(transform, coords, CoordinateKind.POINT)
        
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result, expected)


class TestCoordinateOperators:
    """Tests for mathematical operators on Coordinate objects."""

    def test_coordinate_addition(self):
        """Test adding two coordinates."""
        point1 = Point([1, 2])
        point2 = Point([3, 4])
        
        result = point1 + point2
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 6])
        assert result.kind == CoordinateKind.POINT
        assert result.frame is point1.frame

    def test_coordinate_subtraction(self):
        """Test subtracting two coordinates."""
        point1 = Point([5, 7])
        point2 = Point([2, 3])
        
        result = point1 - point2
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [3, 4])
        assert result.kind == CoordinateKind.POINT

    def test_coordinate_scalar_multiplication(self):
        """Test multiplying coordinate by scalar."""
        point = Point([2, 3])
        
        result = point * 2
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 6])
        assert result.kind == CoordinateKind.POINT

    def test_coordinate_scalar_multiplication_reverse(self):
        """Test multiplying scalar by coordinate."""
        point = Point([2, 3])
        
        result = 2 * point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 6])

    def test_coordinate_division(self):
        """Test dividing coordinate by scalar."""
        point = Point([4, 6])
        
        result = point / 2
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [2, 3])

    def test_coordinate_negation(self):
        """Test negating a coordinate."""
        point = Point([1, -2])
        
        result = -point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [-1, 2])

    def test_coordinate_indexing(self):
        """Test indexing into a coordinate."""
        point = Point([5, 10])
        
        assert point[0] == 5
        assert point[1] == 10

    def test_coordinate_item_assignment(self):
        """Test assigning to coordinate items."""
        point = Point([1, 2])
        point[0] = 5
        
        assert point[0] == 5
        np.testing.assert_array_equal(point.coords, [5, 2])

    def test_coordinate_length(self):
        """Test getting length of coordinate."""
        point = Point([1, 2])
        
        assert len(point) == 2

    def test_coordinate_as_numpy_array(self):
        """Test using coordinate in numpy operations."""
        point = Point([1, 2])
        
        # Should work with numpy functions via __array__
        result = np.sin(point)
        expected = np.sin(np.array([1, 2]))
        np.testing.assert_array_almost_equal(result, expected)

    def test_coordinate_equality(self):
        """Test coordinate equality comparison."""
        point1 = Point([1, 2])
        point2 = Point([1, 2])
        point3 = Point([3, 4])
        
        assert point1 == point2
        assert point1 != point3

    def test_coordinate_absolute_value(self):
        """Test absolute value of coordinate."""
        point = Point([-3, 4])
        
        result = abs(point)
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [3, 4])

    def test_multiple_coordinates_indexing(self):
        """Test indexing with multiple coordinates."""
        point = Point([[1, 2, 3], [4, 5, 6]])  # DxN array
        
        # Index first dimension
        np.testing.assert_array_equal(point[0], [1, 2, 3])
        # Index specific element
        assert point[1, 2] == 6

    def test_operations_preserve_frame(self):
        """Test that operations preserve the coordinate frame."""
        frame = Frame(transform=translate2D(5, 3))
        point = Point([1, 2], frame=frame)
        
        result = point * 2
        assert result.frame is frame

    def test_vector_operations(self):
        """Test operations on vectors preserve vector type."""
        vector = Vector([1, 0])
        
        result = vector * 3
        assert isinstance(result, Coordinate)
        assert result.kind == CoordinateKind.VECTOR
        np.testing.assert_array_equal(result.coords, [3, 0])

    def test_different_frames_addition_raises_error(self):
        """Test that adding coordinates from different frames raises an error."""
        frame1 = Frame(transform=translate2D(5, 0))
        frame2 = Frame(transform=translate2D(0, 5))
        
        point1 = Point([1, 2], frame=frame1)
        point2 = Point([3, 4], frame=frame2)
        
        with np.testing.assert_raises(ValueError) as exc_info:
            _ = point1 + point2
        assert "different frames" in str(exc_info.exception).lower()

    def test_different_frames_subtraction_raises_error(self):
        """Test that subtracting coordinates from different frames raises an error."""
        frame1 = Frame(transform=translate2D(5, 0))
        frame2 = Frame(transform=translate2D(0, 5))
        
        point1 = Point([1, 2], frame=frame1)
        point2 = Point([3, 4], frame=frame2)
        
        with np.testing.assert_raises(ValueError) as exc_info:
            _ = point1 - point2
        assert "different frames" in str(exc_info.exception).lower()

    def test_different_frames_multiplication_raises_error(self):
        """Test that multiplying coordinates from different frames raises an error."""
        frame1 = Frame(transform=translate2D(5, 0))
        frame2 = Frame(transform=translate2D(0, 5))
        
        point1 = Point([1, 2], frame=frame1)
        point2 = Point([3, 4], frame=frame2)
        
        with np.testing.assert_raises(ValueError) as exc_info:
            _ = point1 * point2
        assert "different frames" in str(exc_info.exception).lower()

    def test_different_frames_division_raises_error(self):
        """Test that dividing coordinates from different frames raises an error."""
        frame1 = Frame(transform=translate2D(5, 0))
        frame2 = Frame(transform=translate2D(0, 5))
        
        point1 = Point([4, 6], frame=frame1)
        point2 = Point([2, 3], frame=frame2)
        
        with np.testing.assert_raises(ValueError) as exc_info:
            _ = point1 / point2
        assert "different frames" in str(exc_info.exception).lower()

    def test_same_frame_operations_work(self):
        """Test that operations between coordinates in the same frame work correctly."""
        frame = Frame(transform=translate2D(5, 0))
        
        point1 = Point([1, 2], frame=frame)
        point2 = Point([3, 4], frame=frame)
        
        # All these should work
        result_add = point1 + point2
        result_sub = point1 - point2
        result_mul = point1 * point2
        result_div = point2 / point1
        
        np.testing.assert_array_equal(result_add.coords, [4, 6])
        np.testing.assert_array_equal(result_sub.coords, [-2, -2])
        np.testing.assert_array_equal(result_mul.coords, [3, 8])
        np.testing.assert_array_equal(result_div.coords, [3, 2])

    def test_coordinate_with_array_addition(self):
        """Test adding a coordinate with a plain array."""
        point = Point([1, 2])
        array = np.array([3, 4])
        
        result = point + array
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 6])

    def test_coordinate_with_array_subtraction(self):
        """Test subtracting a plain array from coordinate."""
        point = Point([5, 7])
        array = np.array([2, 3])
        
        result = point - array
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [3, 4])

    def test_right_addition_with_array(self):
        """Test right addition (array + coordinate)."""
        point = Point([1, 2])
        array = np.array([3, 4])
        
        result = array + point
        # When numpy array is on the left, numpy handles it and returns array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [4, 6])

    def test_right_subtraction_with_array(self):
        """Test right subtraction (array - coordinate)."""
        point = Point([1, 2])
        array = np.array([5, 7])
        
        result = array - point
        # When numpy array is on the left, numpy handles it and returns array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [4, 5])

    def test_right_division_with_array(self):
        """Test right division (array / coordinate)."""
        point = Point([2, 4])
        array = np.array([8, 12])
        
        result = array / point
        # When numpy array is on the left, numpy handles it and returns array
        assert isinstance(result, np.ndarray)
        np.testing.assert_array_equal(result, [4, 3])

    def test_array_conversion_with_dtype(self):
        """Test __array__ method with dtype specification."""
        point = Point([1.5, 2.7])
        
        # Convert to integer array
        int_array = np.asarray(point, dtype=int)
        np.testing.assert_array_equal(int_array, [1, 2])

    def test_repr(self):
        """Test string representation of coordinate."""
        point = Point([1, 2])
        repr_str = repr(point)
        
        assert "Point" in repr_str
        assert "coords" in repr_str

    def test_equality_with_array(self):
        """Test equality comparison with plain array."""
        point = Point([1, 2])
        array = np.array([1, 2])
        
        assert point == array

    def test_radd_with_scalar(self):
        """Test right addition with scalar (scalar + coordinate)."""
        point = Point([1, 2])
        
        # This should trigger __radd__
        result = 5 + point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [6, 7])

    def test_rsub_with_scalar(self):
        """Test right subtraction with scalar (scalar - coordinate)."""
        point = Point([1, 2])
        
        # This should trigger __rsub__
        result = 10 - point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [9, 8])

    def test_rtruediv_with_scalar(self):
        """Test right division with scalar (scalar / coordinate)."""
        point = Point([2, 4])
        
        # This should trigger __rtruediv__
        result = 8 / point
        assert isinstance(result, Coordinate)
        np.testing.assert_array_equal(result.coords, [4, 2])


class TestPointAndVectorBehavior:
    """Tests to verify Point and Vector behave differently with transformations."""

    def test_point_affected_by_translation(self):
        """Test that points are affected by translation."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        point = Point(coords=np.array([1, 2]), frame=frame)
        
        # When converting to identity frame, point should be translated
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = point.relative_to(identity_frame)
        
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_vector_unaffected_by_translation(self):
        """Test that vectors are unaffected by translation."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        vector = Vector(coords=np.array([1, 2]), frame=frame)
        
        # When converting to identity frame, vector should not be translated
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = vector.relative_to(identity_frame)
        
        expected = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_point_scaled_from_origin(self):
        """Test that points are scaled from origin."""
        frame = Frame(transform=scale2D(2, 2), parent=None)
        point = Point(coords=np.array([3, 4]), frame=frame)
        
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = point.relative_to(identity_frame)
        
        expected = np.array([6, 8])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_vector_scaled(self):
        """Test that vectors are also scaled."""
        frame = Frame(transform=scale2D(2, 2), parent=None)
        vector = Vector(coords=np.array([3, 4]), frame=frame)
        
        identity_frame = Frame(transform=np.eye(3), parent=None)
        result = vector.relative_to(identity_frame)
        
        expected = np.array([6, 8])
        np.testing.assert_array_almost_equal(result.coords, expected)

    def test_point_and_vector_both_rotate(self):
        """Test that both points and vectors rotate the same way."""
        frame = Frame(transform=rotate2D(np.pi / 2), parent=None)
        identity_frame = Frame(transform=np.eye(3), parent=None)
        
        point = Point(coords=np.array([1, 0]), frame=frame)
        vector = Vector(coords=np.array([1, 0]), frame=frame)
        
        point_result = point.relative_to(identity_frame)
        vector_result = vector.relative_to(identity_frame)
        
        # Both should rotate the same
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(point_result.coords, expected)
        np.testing.assert_array_almost_equal(vector_result.coords, expected)
