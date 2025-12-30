"""Unit tests for the System class."""

import numpy as np
import pytest
from coordinate.system import System, system_factory
from coordinate.transforms import translate2D, rotate2D, scale2D, trs2D


class TestSystemInit:
    """Tests for System initialization."""

    def test_system_init_no_parent(self):
        """Test creating a system without a parent."""
        transform = np.eye(3)
        system = System(transform=transform, parent=None)
        
        assert system.parent is None
        np.testing.assert_array_equal(system.transform, transform)

    def test_system_init_with_parent(self):
        """Test creating a system with a parent."""
        parent_transform = translate2D(5, 3)
        parent = System(transform=parent_transform, parent=None)
        
        child_transform = rotate2D(np.pi / 4)
        child = System(transform=child_transform, parent=parent)
        
        assert child.parent is parent
        np.testing.assert_array_equal(child.transform, child_transform)
        np.testing.assert_array_equal(child.parent.transform, parent_transform)


class TestGlobalTransform:
    """Tests for the global_transform method."""

    def test_global_transform_no_parent(self):
        """Test global transform when there's no parent (should return own transform)."""
        transform = translate2D(3, 2)
        system = System(transform=transform, parent=None)
        
        result = system.global_transform()
        expected = transform
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_transform_one_parent(self):
        """Test global transform with one parent level."""
        # Parent translates by (10, 5)
        parent = System(transform=translate2D(10, 5), parent=None)
        
        # Child translates by (3, 2) relative to parent
        child = System(transform=translate2D(3, 2), parent=parent)
        
        result = child.global_transform()
        # Expected: parent @ child
        expected = translate2D(10, 5) @ translate2D(3, 2)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Verify a point transforms correctly
        point = np.array([0, 0, 1])
        transformed = result @ point
        # Point should be at (13, 7) in global space
        np.testing.assert_array_almost_equal(transformed, [13, 7, 1])

    def test_global_transform_nested_hierarchy(self):
        """Test global transform with multiple nested parents."""
        # Grandparent: translate (10, 0)
        grandparent = System(transform=translate2D(10, 0), parent=None)
        
        # Parent: translate (5, 0) relative to grandparent
        parent = System(transform=translate2D(5, 0), parent=grandparent)
        
        # Child: translate (2, 0) relative to parent
        child = System(transform=translate2D(2, 0), parent=parent)
        
        result = child.global_transform()
        
        # Expected: grandparent @ parent @ child
        expected = translate2D(10, 0) @ translate2D(5, 0) @ translate2D(2, 0)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Point at origin should end up at (17, 0)
        point = np.array([0, 0, 1])
        transformed = result @ point
        np.testing.assert_array_almost_equal(transformed, [17, 0, 1])

    def test_global_transform_with_rotation_and_scale(self):
        """Test global transform with rotation and scaling."""
        # Parent: scale by 2
        parent = System(transform=scale2D(2, 2), parent=None)
        
        # Child: rotate 90 degrees
        child = System(transform=rotate2D(np.pi / 2), parent=parent)
        
        result = child.global_transform()
        expected = scale2D(2, 2) @ rotate2D(np.pi / 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_transform_complex_hierarchy(self):
        """Test global transform with complex transformations at each level."""
        # Root: translate and rotate
        root = System(transform=trs2D(10, 5, np.pi / 4, 1, 1), parent=None)
        
        # Middle: scale
        middle = System(transform=scale2D(2, 2), parent=root)
        
        # Leaf: translate
        leaf = System(transform=translate2D(3, 0), parent=middle)
        
        result = leaf.global_transform()
        expected = trs2D(10, 5, np.pi / 4, 1, 1) @ scale2D(2, 2) @ translate2D(3, 0)
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeConvertTransform:
    """Tests for the compute_convert_transform method."""

    def test_convert_transform_same_system(self):
        """Test conversion from a system to itself (should be identity)."""
        system = System(transform=translate2D(5, 3), parent=None)
        
        result = system.compute_convert_transform(system)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_transform_siblings(self):
        """Test conversion between sibling systems."""
        parent = System(transform=np.eye(3), parent=None)
        
        # System A: translate by (5, 0)
        system_a = System(transform=translate2D(5, 0), parent=parent)
        
        # System B: translate by (0, 3)
        system_b = System(transform=translate2D(0, 3), parent=parent)
        
        # Convert from A to B
        result = system_a.compute_convert_transform(system_b)
        
        # To go from A to B: go to global, then to B
        # Global of A: (5, 0)
        # Inverse of B: (-0, -3)
        # So point at (0,0) in A is at (5, 0) globally, which is (5, -3) in B
        point_in_a = np.array([0, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [5, -3, 1])

    def test_convert_transform_parent_to_child(self):
        """Test conversion from parent to child coordinate system."""
        parent = System(transform=translate2D(10, 5), parent=None)
        child = System(transform=translate2D(3, 2), parent=parent)
        
        # Convert from parent to child
        result = parent.compute_convert_transform(child)
        
        # Point at (0, 0) in parent is at (0, 0) globally (since parent has global (10,5))
        # In child coordinates, we need to invert the child's global transform
        point_in_parent = np.array([0, 0, 1])
        point_in_child = result @ point_in_parent
        
        # Parent's origin is at (10, 5) globally
        # Child's origin is at (13, 7) globally
        # So parent origin in child coords is at (-3, -2)
        np.testing.assert_array_almost_equal(point_in_child, [-3, -2, 1])

    def test_convert_transform_child_to_parent(self):
        """Test conversion from child to parent coordinate system."""
        parent = System(transform=translate2D(10, 5), parent=None)
        child = System(transform=translate2D(3, 2), parent=parent)
        
        # Convert from child to parent
        result = child.compute_convert_transform(parent)
        
        # Point at (0, 0) in child is at (13, 7) globally
        # In parent coordinates, that's (3, 2)
        point_in_child = np.array([0, 0, 1])
        point_in_parent = result @ point_in_child
        np.testing.assert_array_almost_equal(point_in_parent, [3, 2, 1])

    def test_convert_transform_with_rotation(self):
        """Test conversion with rotated coordinate systems."""
        # System A: no transformation
        system_a = System(transform=np.eye(3), parent=None)
        
        # System B: rotated 90 degrees
        system_b = System(transform=rotate2D(np.pi / 2), parent=None)
        
        # Convert from A to B
        result = system_a.compute_convert_transform(system_b)
        
        # Point (1, 0) in A should be (0, -1) in B (rotated -90 degrees)
        point_in_a = np.array([1, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [0, -1, 1])

    def test_convert_transform_nested_systems(self):
        """Test conversion between systems in different branches of hierarchy."""
        root = System(transform=np.eye(3), parent=None)
        
        # Branch A
        branch_a = System(transform=translate2D(10, 0), parent=root)
        
        # Branch B
        branch_b = System(transform=translate2D(0, 10), parent=root)
        
        # Convert from branch_a to branch_b
        result = branch_a.compute_convert_transform(branch_b)
        
        # Point at (0, 0) in branch_a is at (10, 0) globally
        # In branch_b coords, that's (10, -10)
        point_in_a = np.array([0, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [10, -10, 1])


class TestSystemFactory:
    """Tests for the system_factory function."""

    def test_factory_identity(self):
        """Test factory with identity transformation."""
        system = system_factory(parent=None, tx=0, ty=0, angle_rad=0, sx=1, sy=1)
        
        np.testing.assert_array_almost_equal(system.transform, np.eye(3))
        assert system.parent is None

    def test_factory_translation(self):
        """Test factory with translation."""
        system = system_factory(parent=None, tx=5, ty=3)
        
        expected = translate2D(5, 3)
        np.testing.assert_array_almost_equal(system.transform, expected)

    def test_factory_rotation(self):
        """Test factory with rotation."""
        system = system_factory(parent=None, angle_rad=np.pi / 2)
        
        expected = rotate2D(np.pi / 2)
        np.testing.assert_array_almost_equal(system.transform, expected)

    def test_factory_scale(self):
        """Test factory with scaling."""
        system = system_factory(parent=None, sx=2, sy=3)
        
        expected = scale2D(2, 3)
        np.testing.assert_array_almost_equal(system.transform, expected)

    def test_factory_full_trs(self):
        """Test factory with all TRS parameters."""
        system = system_factory(
            parent=None,
            tx=10, ty=5,
            angle_rad=np.pi / 4,
            sx=2, sy=1.5
        )
        
        expected = trs2D(10, 5, np.pi / 4, 2, 1.5)
        np.testing.assert_array_almost_equal(system.transform, expected)

    def test_factory_with_parent(self):
        """Test factory with a parent system."""
        parent = System(transform=translate2D(100, 100), parent=None)
        child = system_factory(parent=parent, tx=5, ty=3)
        
        assert child.parent is parent
        expected = translate2D(5, 3)
        np.testing.assert_array_almost_equal(child.transform, expected)

    def test_factory_default_parameters(self):
        """Test factory with default parameters."""
        system = system_factory(parent=None)
        
        # Should create identity transform with default params
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(system.transform, expected)

    def test_factory_builds_hierarchy(self):
        """Test using factory to build a system hierarchy."""
        root = system_factory(parent=None, tx=10, ty=10)
        child = system_factory(parent=root, tx=5, ty=0, angle_rad=np.pi / 2)
        grandchild = system_factory(parent=child, sx=2, sy=2)
        
        # Test hierarchy is connected
        assert child.parent is root
        assert grandchild.parent is child
        
        # Test global transform of grandchild
        global_t = grandchild.global_transform()
        expected = trs2D(10, 10, 0, 1, 1) @ trs2D(5, 0, np.pi / 2, 1, 1) @ trs2D(0, 0, 0, 2, 2)
        np.testing.assert_array_almost_equal(global_t, expected)
