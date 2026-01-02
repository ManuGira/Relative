"""Unit tests for the coordinatus package __init__.py."""


class TestPackageImports:
    """Tests for package-level imports."""

    def test_import_main_classes(self):
        """Test that main classes can be imported from package."""
        from coordinatus import CoordinateKind, Frame, Point, Vector, Coordinate
        from coordinatus.transforms import translate2D
        from coordinatus import create_frame
        
        # Verify they are the expected types
        assert CoordinateKind is not None
        assert Frame is not None
        assert Point is not None
        assert Vector is not None
        assert Coordinate is not None
        assert callable(translate2D)
        assert callable(create_frame)

    def test_visualization_import_success(self):
        """Test that visualization module is available when matplotlib is installed."""
        import coordinatus
        
        # Since matplotlib is installed in dev dependencies, this should not be None
        assert coordinatus.visualization is not None
        
        # Test that we can import from it
        from coordinatus.visualization import draw_frame_axes, draw_points
        assert callable(draw_frame_axes)
        assert callable(draw_points)

