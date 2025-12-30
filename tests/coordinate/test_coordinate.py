"""Unit tests for the Coordinate, Point, and Vector classes."""

import numpy as np
import pytest
from coordinate.coordinate import Coordinate, Point, Vector, transform_coordinate
from coordinate.system import System, system_factory
from coordinate.types import CoordinateType
from coordinate.transforms import translate2D, rotate2D, scale2D, trs2D


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
        """Test creating a coordinate without a system creates identity system."""
        coord = Coordinate(
            coordinate_type=CoordinateType.POINT,
            local_coords=np.array([1, 2]),
            system=None
        )
        
        assert coord.coordinate_type == CoordinateType.POINT
        np.testing.assert_array_equal(coord.local_coords, [1, 2])
        assert coord.system is not None
        assert coord.system.parent is None
        np.testing.assert_array_equal(coord.system.transform, np.eye(3))

    def test_coordinate_init_with_system(self):
        """Test creating a coordinate with a custom system."""
        system = System(transform=translate2D(5, 3), parent=None)
        coord = Coordinate(
            coordinate_type=CoordinateType.VECTOR,
            local_coords=np.array([2, 3]),
            system=system
        )
        
        assert coord.coordinate_type == CoordinateType.VECTOR
        np.testing.assert_array_equal(coord.local_coords, [2, 3])
        assert coord.system is system


class TestPointInit:
    """Tests for Point initialization."""

    def test_point_init_basic(self):
        """Test creating a basic point."""
        point = Point(local_coords=np.array([3, 4]))
        
        assert point.coordinate_type == CoordinateType.POINT
        np.testing.assert_array_equal(point.local_coords, [3, 4])
        assert point.system is not None

    def test_point_init_with_system(self):
        """Test creating a point with a custom system."""
        system = System(transform=translate2D(10, 20), parent=None)
        point = Point(local_coords=np.array([1, 1]), system=system)
        
        assert point.coordinate_type == CoordinateType.POINT
        assert point.system is system


class TestVectorInit:
    """Tests for Vector initialization."""

    def test_vector_init_basic(self):
        """Test creating a basic vector."""
        vector = Vector(local_coords=np.array([1, 0]))
        
        assert vector.coordinate_type == CoordinateType.VECTOR
        np.testing.assert_array_equal(vector.local_coords, [1, 0])
        assert vector.system is not None

    def test_vector_init_with_system(self):
        """Test creating a vector with a custom system."""
        system = System(transform=rotate2D(np.pi / 4), parent=None)
        vector = Vector(local_coords=np.array([1, 1]), system=system)
        
        assert vector.coordinate_type == CoordinateType.VECTOR
        assert vector.system is system


class TestCoordinateToGlobal:
    """Tests for converting coordinates to global space."""

    def test_to_global_no_parent(self):
        """Test to_global when system has no parent (root system)."""
        system = System(transform=translate2D(5, 3), parent=None)
        point = Point(local_coords=np.array([1, 2]), system=system)
        
        result = point.to_global()
        
        # Point at (1, 2) in system with translation (5, 3)
        # Global coordinates should be (6, 5)
        np.testing.assert_array_almost_equal(result.local_coords, [6, 5])
        # Result should be in identity/global system
        assert result.system.parent is None
        np.testing.assert_array_almost_equal(result.system.transform, np.eye(3))

    def test_to_global_one_level(self):
        """Test to_global with one parent level."""
        parent = System(transform=translate2D(10, 5), parent=None)
        child_system = System(transform=translate2D(3, 2), parent=parent)
        point = Point(local_coords=np.array([1, 1]), system=child_system)
        
        result = point.to_global()
        
        # Point at (1,1) in child
        # Child is at (3,2) relative to parent
        # So point is at (4,3) in parent
        # Parent is at (10,5) in global
        # So point is at (14,8) in global
        np.testing.assert_array_almost_equal(result.local_coords, [14, 8])
        # Result should be in global/identity system
        assert result.system.parent is None
        np.testing.assert_array_almost_equal(result.system.transform, np.eye(3))

    def test_to_global_multiple_levels(self):
        """Test to_global with nested hierarchy."""
        root = System(transform=translate2D(100, 100), parent=None)
        middle = System(transform=translate2D(10, 10), parent=root)
        leaf = System(transform=translate2D(1, 1), parent=middle)
        
        point = Point(local_coords=np.array([0, 0]), system=leaf)
        result = point.to_global()
        
        # Point at origin in leaf system
        # Leaf is at (1, 1) in middle, middle is at (10, 10) in root, root is at (100, 100) in global
        # So point is at (111, 111) in global coordinates
        np.testing.assert_array_almost_equal(result.local_coords, [111, 111])
        # Result should be in global/identity system
        assert result.system.parent is None
        np.testing.assert_array_almost_equal(result.system.transform, np.eye(3))

    def test_to_global_with_rotation(self):
        """Test to_global with rotated coordinate system."""
        parent = System(transform=np.eye(3), parent=None)
        # Child rotated 90 degrees
        child = System(transform=rotate2D(np.pi / 2), parent=parent)
        
        # Point at (1, 0) in child system
        point = Point(local_coords=np.array([1, 0]), system=child)
        result = point.to_global()
        
        # After 90 degree rotation, (1, 0) becomes (0, 1) in global
        np.testing.assert_array_almost_equal(result.local_coords, [0, 1])
        # Result should be in global/identity system
        assert result.system.parent is None
        np.testing.assert_array_almost_equal(result.system.transform, np.eye(3))

    def test_to_global_vector_ignores_translation(self):
        """Test that vectors ignore translation when going to global."""
        parent = System(transform=translate2D(10, 20), parent=None)
        child = System(transform=translate2D(5, 5), parent=parent)
        
        vector = Vector(local_coords=np.array([1, 0]), system=child)
        result = vector.to_global()
        
        # Vector should maintain direction, translation ignored (weight=0)
        # Vector (1, 0) should remain (1, 0) in global since translations don't affect vectors
        np.testing.assert_array_almost_equal(result.local_coords, [1, 0])
        # Result should be in global/identity system
        assert result.system.parent is None
        np.testing.assert_array_almost_equal(result.system.transform, np.eye(3))


class TestCoordinateToSystem:
    """Tests for converting coordinates between systems."""

    def test_to_system_same_system(self):
        """Test converting to the same system."""
        system = System(transform=translate2D(5, 3), parent=None)
        point = Point(local_coords=np.array([1, 2]), system=system)
        
        result = point.to_system(system)
        
        # Should have same coordinates in same system
        np.testing.assert_array_almost_equal(result.local_coords, [1, 2])
        assert result.system is system

    def test_to_system_siblings(self):
        """Test converting between sibling coordinate systems."""
        parent = System(transform=np.eye(3), parent=None)
        system_a = System(transform=translate2D(5, 0), parent=parent)
        system_b = System(transform=translate2D(0, 3), parent=parent)
        
        # Point at (0, 0) in system A
        point = Point(local_coords=np.array([0, 0]), system=system_a)
        
        # Convert to system B
        result = point.to_system(system_b)
        
        # Point is at (5, 0) globally
        # In system B coords (which is at (0, 3)), that's (5, -3)
        np.testing.assert_array_almost_equal(result.local_coords, [5, -3])
        assert result.system is system_b

    def test_to_system_parent_to_child(self):
        """Test converting from parent to child system."""
        parent = System(transform=translate2D(10, 5), parent=None)
        child = System(transform=translate2D(3, 2), parent=parent)
        
        point = Point(local_coords=np.array([0, 0]), system=parent)
        result = point.to_system(child)
        
        # Point at parent origin is at (10, 5) globally
        # Child origin is at (13, 7) globally
        # So parent origin in child coords is (-3, -2)
        np.testing.assert_array_almost_equal(result.local_coords, [-3, -2])
        assert result.system is child

    def test_to_system_child_to_parent(self):
        """Test converting from child to parent system."""
        parent = System(transform=translate2D(10, 5), parent=None)
        child = System(transform=translate2D(3, 2), parent=parent)
        
        point = Point(local_coords=np.array([0, 0]), system=child)
        result = point.to_system(parent)
        
        # Point at child origin (13, 7 globally) is at (3, 2) in parent
        np.testing.assert_array_almost_equal(result.local_coords, [3, 2])
        assert result.system is parent

    def test_to_system_with_rotation(self):
        """Test converting between rotated systems."""
        system_a = System(transform=np.eye(3), parent=None)
        system_b = System(transform=rotate2D(np.pi / 2), parent=None)
        
        # Point at (1, 0) in system A
        point = Point(local_coords=np.array([1, 0]), system=system_a)
        
        # Convert to system B (rotated 90 degrees)
        result = point.to_system(system_b)
        
        # (1, 0) in A becomes (0, -1) in B (inverse rotation)
        np.testing.assert_array_almost_equal(result.local_coords, [0, -1])

    def test_to_system_vector_translation(self):
        """Test that vector conversion ignores translation."""
        system_a = System(transform=translate2D(10, 5), parent=None)
        system_b = System(transform=translate2D(20, 15), parent=None)
        
        vector = Vector(local_coords=np.array([1, 0]), system=system_a)
        result = vector.to_system(system_b)
        
        # Vector direction should be same regardless of translation
        np.testing.assert_array_almost_equal(result.local_coords, [1, 0])
        assert result.system is system_b

    def test_to_system_vector_rotation(self):
        """Test that vector conversion respects rotation."""
        system_a = System(transform=np.eye(3), parent=None)
        system_b = System(transform=rotate2D(np.pi / 2), parent=None)
        
        vector = Vector(local_coords=np.array([1, 0]), system=system_a)
        result = vector.to_system(system_b)
        
        # Vector should rotate
        np.testing.assert_array_almost_equal(result.local_coords, [0, -1])

    def test_to_system_complex_hierarchy(self):
        """Test conversion in complex hierarchy."""
        root = System(transform=np.eye(3), parent=None)
        branch_a = System(transform=translate2D(10, 0), parent=root)
        leaf_a = System(transform=rotate2D(np.pi / 4), parent=branch_a)
        
        branch_b = System(transform=translate2D(0, 10), parent=root)
        
        point = Point(local_coords=np.array([1, 0]), system=leaf_a)
        result = point.to_system(branch_b)
        
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
        system = System(transform=translate2D(5, 3), parent=None)
        point = Point(local_coords=np.array([1, 2]), system=system)
        
        # When converting to identity system, point should be translated
        identity_system = System(transform=np.eye(3), parent=None)
        result = point.to_system(identity_system)
        
        expected = np.array([6, 5])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_vector_unaffected_by_translation(self):
        """Test that vectors are unaffected by translation."""
        system = System(transform=translate2D(5, 3), parent=None)
        vector = Vector(local_coords=np.array([1, 2]), system=system)
        
        # When converting to identity system, vector should not be translated
        identity_system = System(transform=np.eye(3), parent=None)
        result = vector.to_system(identity_system)
        
        expected = np.array([1, 2])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_point_scaled_from_origin(self):
        """Test that points are scaled from origin."""
        system = System(transform=scale2D(2, 2), parent=None)
        point = Point(local_coords=np.array([3, 4]), system=system)
        
        identity_system = System(transform=np.eye(3), parent=None)
        result = point.to_system(identity_system)
        
        expected = np.array([6, 8])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_vector_scaled(self):
        """Test that vectors are also scaled."""
        system = System(transform=scale2D(2, 2), parent=None)
        vector = Vector(local_coords=np.array([3, 4]), system=system)
        
        identity_system = System(transform=np.eye(3), parent=None)
        result = vector.to_system(identity_system)
        
        expected = np.array([6, 8])
        np.testing.assert_array_almost_equal(result.local_coords, expected)

    def test_point_and_vector_both_rotate(self):
        """Test that both points and vectors rotate the same way."""
        system = System(transform=rotate2D(np.pi / 2), parent=None)
        identity_system = System(transform=np.eye(3), parent=None)
        
        point = Point(local_coords=np.array([1, 0]), system=system)
        vector = Vector(local_coords=np.array([1, 0]), system=system)
        
        point_result = point.to_system(identity_system)
        vector_result = vector.to_system(identity_system)
        
        # Both should rotate the same
        expected = np.array([0, 1])
        np.testing.assert_array_almost_equal(point_result.local_coords, expected)
        np.testing.assert_array_almost_equal(vector_result.local_coords, expected)
