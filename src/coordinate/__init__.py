"""Coordinate package for managing coordinate systems and transformations."""

# Import main classes and functions for convenient access
from .types import CoordinateType
from .transforms import translate2D, rotate2D, scale2D, trs2D
from .system import System, system_factory
from .coordinate import Coordinate, Point, Vector, transform_coordinate

# Define what's available when using "from coordinate import *"
__all__ = [
    # Nothing to export explicitly, avoinding namespace conflictions
]

