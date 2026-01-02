"""Unit tests for the Frame class."""

import numpy as np
from coordinatus.frame import Frame, create_frame
from coordinatus.transforms import translate2D, rotate2D, scale2D, trs2D


class TestFrameInit:
    """Tests for Frame initialization."""

    def test_frame_init_no_parent(self):
        """Test creating a frame without a parent."""
        transform = np.eye(3)
        frame = Frame(transform=transform, parent=None)
        
        assert frame.parent is None
        np.testing.assert_array_equal(frame.transform, transform)

    def test_frame_init_with_parent(self):
        """Test creating a frame with a parent."""
        parent_transform = translate2D(5, 3)
        parent = Frame(transform=parent_transform, parent=None)
        
        child_transform = rotate2D(np.pi / 4)
        child = Frame(transform=child_transform, parent=parent)
        
        assert child.parent is parent
        np.testing.assert_array_equal(child.transform, child_transform)
        np.testing.assert_array_equal(child.parent.transform, parent_transform)


class TestFrameDimensions:
    """Tests for Frame.D_in and Frame.D_out properties."""

    def test_D_in_2D_frame(self):
        """Test D_in for a 2D frame (3x3 transformation matrix)."""
        frame = Frame(transform=np.eye(3))
        assert frame.D_in == 2

    def test_D_out_2D_frame(self):
        """Test D_out for a 2D frame (3x3 transformation matrix)."""
        frame = Frame(transform=np.eye(3))
        assert frame.D_out == 2

    def test_D_in_3D_frame(self):
        """Test D_in for a 3D frame (4x4 transformation matrix)."""
        frame = Frame(transform=np.eye(4))
        assert frame.D_in == 3

    def test_D_out_3D_frame(self):
        """Test D_out for a 3D frame (4x4 transformation matrix)."""
        frame = Frame(transform=np.eye(4))
        assert frame.D_out == 3

    def test_D_in_with_translation_2D(self):
        """Test D_in remains correct with translated 2D frame."""
        frame = Frame(transform=translate2D(5, 10))
        assert frame.D_in == 2

    def test_D_out_with_translation_2D(self):
        """Test D_out remains correct with translated 2D frame."""
        frame = Frame(transform=translate2D(5, 10))
        assert frame.D_out == 2

    def test_D_in_with_rotation_2D(self):
        """Test D_in remains correct with rotated 2D frame."""
        frame = Frame(transform=rotate2D(np.pi / 4))
        assert frame.D_in == 2

    def test_D_out_with_rotation_2D(self):
        """Test D_out remains correct with rotated 2D frame."""
        frame = Frame(transform=rotate2D(np.pi / 4))
        assert frame.D_out == 2

    def test_D_in_with_parent(self):
        """Test D_in is independent of parent frame."""
        parent = Frame(transform=translate2D(10, 5))
        child = Frame(transform=rotate2D(np.pi / 2), parent=parent)
        assert child.D_in == 2
        assert parent.D_in == 2

    def test_D_out_with_parent(self):
        """Test D_out is independent of parent frame."""
        parent = Frame(transform=translate2D(10, 5))
        child = Frame(transform=rotate2D(np.pi / 2), parent=parent)
        assert child.D_out == 2
        assert parent.D_out == 2

    def test_D_in_D_out_equal_for_standard_transforms(self):
        """Test that D_in equals D_out for standard (non-projection) transformations."""
        frames = [
            Frame(transform=np.eye(3)),
            Frame(transform=translate2D(3, 4)),
            Frame(transform=rotate2D(np.pi / 3)),
            Frame(transform=scale2D(2, 3)),
            Frame(transform=trs2D(5, 10, np.pi / 4, 2, 2)),
        ]
        
        for frame in frames:
            assert frame.D_in == frame.D_out, "D_in and D_out should be equal for standard transforms"

    def test_D_in_D_out_different_for_projection(self):
        """Test that D_in != D_out for dimension-changing transformations (projections)."""
        # Create a 3x4 projection matrix (projects 3D to 2D)
        projection_3d_to_2d = np.array([[1, 0, 0, 0],
                                        [0, 1, 0, 0],
                                        [0, 0, 0, 1]])
        
        frame = Frame(transform=projection_3d_to_2d)
        assert frame.D_in == 3  # Input is 3D
        assert frame.D_out == 2  # Output is 2D
        assert frame.D_in != frame.D_out

    def test_D_in_1D_frame(self):
        """Test D_in for a 1D frame (2x2 transformation matrix)."""
        frame = Frame(transform=np.eye(2))
        assert frame.D_in == 1

    def test_D_out_1D_frame(self):
        """Test D_out for a 1D frame (2x2 transformation matrix)."""
        frame = Frame(transform=np.eye(2))
        assert frame.D_out == 1



class TestFrameEquality:
    """Tests for Frame equality comparison."""

    def test_same_reference_equal(self):
        """Test that same frame object is equal to itself."""
        frame = Frame(transform=translate2D(5, 3))
        
        assert frame == frame
        assert not (frame != frame)

    def test_different_frames_not_equal(self):
        """Test that different frame objects are not equal by default."""
        frame1 = Frame(transform=translate2D(5, 3))
        frame2 = Frame(transform=translate2D(5, 3))
        
        # Different objects, not the same reference
        assert frame1 is not frame2
        assert frame1 != frame2

    def test_identity_frames_equal(self):
        """Test that two identity frames (no parent, identity transform) are equal."""
        frame1 = Frame()  # Default is identity
        frame2 = Frame()  # Another identity
        
        assert frame1 == frame2
        assert not (frame1 != frame2)

    def test_identity_frames_with_explicit_identity_equal(self):
        """Test identity frames created explicitly."""
        frame1 = Frame(transform=np.eye(3), parent=None)
        frame2 = Frame(transform=np.eye(3), parent=None)
        
        assert frame1 == frame2

    def test_identity_and_non_identity_not_equal(self):
        """Test that identity frame is not equal to non-identity frame."""
        identity_frame = Frame()
        translated_frame = Frame(transform=translate2D(5, 3))
        
        assert identity_frame != translated_frame

    def test_frames_with_parents_not_equal(self):
        """Test that frames with parents are not equal (even if transforms are same)."""
        parent = Frame()
        frame1 = Frame(transform=translate2D(5, 3), parent=parent)
        frame2 = Frame(transform=translate2D(5, 3), parent=parent)
        
        # Even though they have same transform and parent, they're different objects
        assert frame1 != frame2

    def test_frame_not_equal_to_non_frame(self):
        """Test that frame is not equal to non-Frame object."""
        frame = Frame()
        
        assert frame is not None
        assert frame != 42
        assert frame != "frame"
        assert frame != np.eye(3)


class TestComputeAbsoluteTransform:
    """Tests for the compute_absolute_transform method."""

    def test_global_transform_no_parent(self):
        """Test absolute transform when there's no parent (should return own transform)."""
        transform = translate2D(3, 2)
        frame = Frame(transform=transform, parent=None)
        
        result = frame.compute_absolute_transform()
        expected = transform
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_transform_one_parent(self):
        """Test absolute transform with one parent level."""
        # Parent translates by (10, 5)
        parent = Frame(transform=translate2D(10, 5), parent=None)
        
        # Child translates by (3, 2) relative to parent
        child = Frame(transform=translate2D(3, 2), parent=parent)
        
        result = child.compute_absolute_transform()
        # Expected: parent @ child
        expected = translate2D(10, 5) @ translate2D(3, 2)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Verify a point transforms correctly
        point = np.array([0, 0, 1])
        transformed = result @ point
        # Point should be at (13, 7) in absolute space
        np.testing.assert_array_almost_equal(transformed, [13, 7, 1])

    def test_global_transform_nested_hierarchy(self):
        """Test absolute transform with multiple nested parents."""
        # Grandparent: translate (10, 0)
        grandparent = Frame(transform=translate2D(10, 0), parent=None)
        
        # Parent: translate (5, 0) relative to grandparent
        parent = Frame(transform=translate2D(5, 0), parent=grandparent)
        
        # Child: translate (2, 0) relative to parent
        child = Frame(transform=translate2D(2, 0), parent=parent)
        
        result = child.compute_absolute_transform()
        
        # Expected: grandparent @ parent @ child
        expected = translate2D(10, 0) @ translate2D(5, 0) @ translate2D(2, 0)
        np.testing.assert_array_almost_equal(result, expected)
        
        # Point at origin should end up at (17, 0)
        point = np.array([0, 0, 1])
        transformed = result @ point
        np.testing.assert_array_almost_equal(transformed, [17, 0, 1])

    def test_global_transform_with_rotation_and_scale(self):
        """Test absolute transform with rotation and scaling."""
        # Parent: scale by 2
        parent = Frame(transform=scale2D(2, 2), parent=None)
        
        # Child: rotate 90 degrees
        child = Frame(transform=rotate2D(np.pi / 2), parent=parent)
        
        result = child.compute_absolute_transform()
        expected = scale2D(2, 2) @ rotate2D(np.pi / 2)
        np.testing.assert_array_almost_equal(result, expected)

    def test_global_transform_complex_hierarchy(self):
        """Test absolute transform with complex transformations at each level."""
        # Root: translate and rotate
        root = Frame(transform=trs2D(10, 5, np.pi / 4, 1, 1), parent=None)
        
        # Middle: scale
        middle = Frame(transform=scale2D(2, 2), parent=root)
        
        # Leaf: translate
        leaf = Frame(transform=translate2D(3, 0), parent=middle)
        
        result = leaf.compute_absolute_transform()
        expected = trs2D(10, 5, np.pi / 4, 1, 1) @ scale2D(2, 2) @ translate2D(3, 0)
        np.testing.assert_array_almost_equal(result, expected)


class TestComputeRelativeTransformTo:
    """Tests for the compute_relative_transform_to method."""

    def test_convert_transform_same_frame(self):
        """Test conversion from a frame to itself (should be identity)."""
        frame = Frame(transform=translate2D(5, 3), parent=None)
        
        result = frame.compute_relative_transform_to(frame)
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(result, expected)

    def test_convert_transform_siblings(self):
        """Test conversion between sibling frames."""
        parent = Frame(transform=np.eye(3), parent=None)
        
        # Frame A: translate by (5, 0)
        frame_a = Frame(transform=translate2D(5, 0), parent=parent)
        
        # Frame B: translate by (0, 3)
        frame_b = Frame(transform=translate2D(0, 3), parent=parent)
        
        # Convert from A to B
        result = frame_a.compute_relative_transform_to(frame_b)
        
        # To go from A to B: go to absolute, then to B
        # Absolute of A: (5, 0)
        # Inverse of B: (-0, -3)
        # So point at (0,0) in A is at (5, 0) in absolute, which is (5, -3) in B
        point_in_a = np.array([0, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [5, -3, 1])

    def test_convert_transform_parent_to_child(self):
        """Test conversion from parent to child coordinate frame."""
        parent = Frame(transform=translate2D(10, 5), parent=None)
        child = Frame(transform=translate2D(3, 2), parent=parent)
        
        # Convert from parent to child
        result = parent.compute_relative_transform_to(child)
        
        # Point at (0, 0) in parent is at (0, 0) in absolute (since parent has absolute (10,5))
        # In child coordinates, we need to invert the child's absolute transform
        point_in_parent = np.array([0, 0, 1])
        point_in_child = result @ point_in_parent
        
        # Parent's origin is at (10, 5) in absolute
        # Child's origin is at (13, 7) in absolute
        # So parent origin in child coords is at (-3, -2)
        np.testing.assert_array_almost_equal(point_in_child, [-3, -2, 1])

    def test_convert_transform_child_to_parent(self):
        """Test conversion from child to parent coordinate frame."""
        parent = Frame(transform=translate2D(10, 5), parent=None)
        child = Frame(transform=translate2D(3, 2), parent=parent)
        
        # Convert from child to parent
        result = child.compute_relative_transform_to(parent)
        
        # Point at (0, 0) in child is at (13, 7) in absolute
        # In parent coordinates, that's (3, 2)
        point_in_child = np.array([0, 0, 1])
        point_in_parent = result @ point_in_child
        np.testing.assert_array_almost_equal(point_in_parent, [3, 2, 1])

    def test_convert_transform_with_rotation(self):
        """Test conversion with rotated coordinate frames."""
        # Frame A: no transformation
        frame_a = Frame(transform=np.eye(3), parent=None)
        
        # Frame B: rotated 90 degrees
        frame_b = Frame(transform=rotate2D(np.pi / 2), parent=None)
        
        # Convert from A to B
        result = frame_a.compute_relative_transform_to(frame_b)
        
        # Point (1, 0) in A should be (0, -1) in B (rotated -90 degrees)
        point_in_a = np.array([1, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [0, -1, 1])

    def test_convert_transform_nested_frames(self):
        """Test conversion between frames in different branches of hierarchy."""
        root = Frame(transform=np.eye(3), parent=None)
        
        # Branch A
        branch_a = Frame(transform=translate2D(10, 0), parent=root)
        
        # Branch B
        branch_b = Frame(transform=translate2D(0, 10), parent=root)
        
        # Convert from branch_a to branch_b
        result = branch_a.compute_relative_transform_to(branch_b)
        
        # Point at (0, 0) in branch_a is at (10, 0) in absolute
        # In branch_b coords, that's (10, -10)
        point_in_a = np.array([0, 0, 1])
        point_in_b = result @ point_in_a
        np.testing.assert_array_almost_equal(point_in_b, [10, -10, 1])


class TestCreateFrame:
    """Tests for the create_frame function."""

    def test_factory_identity(self):
        """Test factory with identity transformation."""
        frame = create_frame(parent=None, tx=0, ty=0, angle_rad=0, sx=1, sy=1)
        
        np.testing.assert_array_almost_equal(frame.transform, np.eye(3))
        assert frame.parent is None

    def test_factory_translation(self):
        """Test factory with translation."""
        frame = create_frame(parent=None, tx=5, ty=3)
        
        expected = translate2D(5, 3)
        np.testing.assert_array_almost_equal(frame.transform, expected)

    def test_factory_rotation(self):
        """Test factory with rotation."""
        frame = create_frame(parent=None, angle_rad=np.pi / 2)
        
        expected = rotate2D(np.pi / 2)
        np.testing.assert_array_almost_equal(frame.transform, expected)

    def test_factory_scale(self):
        """Test factory with scaling."""
        frame = create_frame(parent=None, sx=2, sy=3)
        
        expected = scale2D(2, 3)
        np.testing.assert_array_almost_equal(frame.transform, expected)

    def test_factory_full_trs(self):
        """Test factory with all TRS parameters."""
        frame = create_frame(
            parent=None,
            tx=10, ty=5,
            angle_rad=np.pi / 4,
            sx=2, sy=1.5
        )
        
        expected = trs2D(10, 5, np.pi / 4, 2, 1.5)
        np.testing.assert_array_almost_equal(frame.transform, expected)

    def test_factory_with_parent(self):
        """Test factory with a parent frame."""
        parent = Frame(transform=translate2D(100, 100), parent=None)
        child = create_frame(parent=parent, tx=5, ty=3)
        
        assert child.parent is parent
        expected = translate2D(5, 3)
        np.testing.assert_array_almost_equal(child.transform, expected)

    def test_factory_default_parameters(self):
        """Test factory with default parameters."""
        frame = create_frame(parent=None)
        
        # Should create identity transform with default params
        expected = np.eye(3)
        np.testing.assert_array_almost_equal(frame.transform, expected)

    def test_factory_builds_hierarchy(self):
        """Test using factory to build a frame hierarchy."""
        root = create_frame(parent=None, tx=10, ty=10)
        child = create_frame(parent=root, tx=5, ty=0, angle_rad=np.pi / 2)
        grandchild = create_frame(parent=child, sx=2, sy=2)
        
        # Test hierarchy is connected
        assert child.parent is root
        assert grandchild.parent is child
        
        # Test absolute transform of grandchild
        absolute_t = grandchild.compute_absolute_transform()
        expected = trs2D(10, 10, 0, 1, 1) @ trs2D(5, 0, np.pi / 2, 1, 1) @ trs2D(0, 0, 0, 2, 2)
        np.testing.assert_array_almost_equal(absolute_t, expected)
