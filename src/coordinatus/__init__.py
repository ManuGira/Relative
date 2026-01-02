"""Coordinate package for managing coordinate systems and transformations."""

# Import main classes and functions for convenient access
from .types import CoordinateKind
from .frame import Frame, create_frame
from . import transforms  # allows access to `coordinatus.transforms.translate2D(1, 2)``
from .coordinate import Coordinate, Point, Vector, transform_coordinate

# Visualization is optional - only available if matplotlib is installed
try:
    from . import visualization
except ImportError:  # pragma: no cover
    visualization = None  # type: ignore[assignment]

# Define what's available when using "from coordinate import *"
__all__ = [
    # Nothing to export explicitly, avoinding namespace conflictions
]


