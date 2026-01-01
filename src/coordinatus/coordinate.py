"""Coordinate representation classes for points and vectors."""

from typing import Optional
import numpy as np
from numpy.typing import ArrayLike

from .frame import Frame
from .types import CoordinateKind


def transform_coordinate(transform: np.ndarray, coordinates: np.ndarray, kind: CoordinateKind) -> np.ndarray:
    """Applies an affine transformation to a coordinate, respecting point vs vector semantics.
    
    Points and vectors transform differently under affine transformations:
    - Points (weight=1): Affected by translation, rotation, and scaling
    - Vectors (weight=0): Affected only by rotation and scaling, NOT translation
    
    This function converts to homogeneous coordinates, applies the transformation,
    and converts back to Cartesian coordinates.
    
    Args:
        transform: 3x3 affine transformation matrix in homogeneous coordinates.
        coordinates: 2D coordinate as numpy array [x, y] or DxN array where D is dimensions
                    and N is the number of points/vectors.
        kind: CoordinateKind.POINT or CoordinateKind.VECTOR.
    
    Returns:
        Transformed 2D coordinate as numpy array [x', y'] or DxN array of transformed coordinates.
    
    Examples:
        >>> # Single point translation
        >>> transform_coordinate(translate2D(5, 3), np.array([1, 2]), CoordinateKind.POINT)
        array([6., 5.])  # Point moved by (5, 3)
        
        >>> # Single vector translation (no effect)
        >>> transform_coordinate(translate2D(5, 3), np.array([1, 2]), CoordinateKind.VECTOR)
        array([1., 2.])  # Vector unchanged
        
        >>> # Multiple points translation
        >>> transform_coordinate(translate2D(5, 3), np.array([[1, 2], [2, 4]]), CoordinateKind.POINT)
        array([[6., 7.], [5., 7.]])  # All points moved by (5, 3)
    """
    # Check if we have a single coordinate (1D) - if so, reshape to (D, 1)
    is_single = coordinates.ndim == 1
    if is_single:
        coordinates = coordinates.reshape(-1, 1)
    
    # Handle DxN array where D is dimensions and N is number of points
    D, N = coordinates.shape
    
    # Convert to homogeneous coordinates by adding a row of weights
    weight = 1.0 if kind == CoordinateKind.POINT else 0.0
    weights = np.full((1, N), weight)
    homogeneous_coords = np.vstack([coordinates, weights])  # Shape (D+1, N)
    
    # Apply transformation: (3, 3) @ (3, N) -> (3, N)
    transformed_coords = transform @ homogeneous_coords
    
    # Return to Cartesian coordinates
    # Normalize by the weight (last row) if necessary
    weights = transformed_coords[-1, :]
    # Avoid division by zero - only normalize where weight != 0
    non_zero_mask = weights != 0
    if np.any(non_zero_mask):
        transformed_coords[:, non_zero_mask] /= weights[non_zero_mask]
    
    # Return all dimensions except the last (weight) row
    result = transformed_coords[:D, :]
    
    # If input was 1D, return 1D
    if is_single:
        result = result.flatten()
    
    return result


class Coordinate:
    """Base class for representing coordinates (points or vectors) in a coordinate frame.
    
    Coordinates can be defined in any coordinate frame and converted between frames.
    The distinction between points and vectors is crucial:
    - Points: Represent positions, affected by all transformations including translation
    - Vectors: Represent directions/displacements, unaffected by translation
    
    Attributes:
        kind: CoordinateKind.POINT or CoordinateKind.VECTOR
        coords: 2D numpy array [x, y] in the local coordinate frame
        frame: The coordinate frame this coordinate is defined in
    
    Examples:
        >>> frame = Frame(transform=translate2D(5, 3))
        >>> coord = Coordinate(CoordinateKind.POINT, np.array([1, 2]), frame)
        >>> coord = Coordinate(CoordinateKind.POINT, [1, 2], frame)  # list also works
        >>> coord = Coordinate(CoordinateKind.POINT, (1, 2), frame)  # tuple also works
        >>> absolute_coord = coord.to_absolute()
    """

    def __init__(self, kind: CoordinateKind, coords: ArrayLike, frame: Optional[Frame] = None):
        """Initialize a coordinate.
        
        Args:
            kind: CoordinateKind.POINT or CoordinateKind.VECTOR
            coords: Array-like (numpy array, list, tuple, etc.) representing coordinates.
                         Can be [x, y] for a single point/vector or [[x1, x2, ...], [y1, y2, ...]]
                         for multiple points/vectors.
            frame: Coordinate frame this coordinate is defined in.
                   If None, uses absolute/identity frame.
        """
        self.kind = kind
        self.coords = np.asarray(coords)
        self.frame = frame if frame is not None else Frame()

    def __array__(self, dtype=None):
        """Return the underlying numpy array for numpy operations."""
        if dtype is None:
            return self.coords
        return self.coords.astype(dtype)

    def __getitem__(self, key):
        """Support indexing operations like coord[0] or coord[0, 1]."""
        return self.coords[key]

    def __setitem__(self, key, value):
        """Support item assignment like coord[0] = 5."""
        self.coords[key] = value

    def __len__(self):
        """Return length of the coordinate array."""
        return len(self.coords)

    def __repr__(self):
        """String representation of the coordinate."""
        return f"{self.__class__.__name__}(coords={self.coords!r}, frame={self.frame!r})"

    # Arithmetic operators
    def __add__(self, other):
        """Add coordinates or arrays."""
        if isinstance(other, Coordinate):
            if self.frame != other.frame:
                raise ValueError("Cannot add coordinates from different frames. Convert to same frame first.")
            new_coords = self.coords + other.coords
        else:
            new_coords = self.coords + other
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __radd__(self, other):
        """Right addition."""
        new_coords = self.coords + other
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __sub__(self, other):
        """Subtract coordinates or arrays."""
        if isinstance(other, Coordinate):
            if self.frame != other.frame:
                raise ValueError("Cannot subtract coordinates from different frames. Convert to same frame first.")
            new_coords = self.coords - other.coords
        else:
            new_coords = self.coords - other
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __rsub__(self, other):
        """Right subtraction."""
        new_coords = other - self.coords
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __mul__(self, other):
        """Multiply coordinates or arrays."""
        if isinstance(other, Coordinate):
            if self.frame != other.frame:
                raise ValueError("Cannot multiply coordinates from different frames. Convert to same frame first.")
            new_coords = self.coords * other.coords
        else:
            new_coords = self.coords * other
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __rmul__(self, other):
        """Right multiplication."""
        new_coords = self.coords * other
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __truediv__(self, other):
        """Divide coordinates or arrays."""
        if isinstance(other, Coordinate):
            if self.frame != other.frame:
                raise ValueError("Cannot divide coordinates from different frames. Convert to same frame first.")
            new_coords = self.coords / other.coords
        else:
            new_coords = self.coords / other
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __rtruediv__(self, other):
        """Right division."""
        new_coords = other / self.coords
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __neg__(self):
        """Negate coordinates."""
        new_coords = -self.coords
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    def __abs__(self):
        """Absolute value of coordinates."""
        new_coords = np.abs(self.coords)
        return Coordinate(kind=self.kind, coords=new_coords, frame=self.frame)

    # Comparison operators
    def __eq__(self, other):
        """Check equality."""
        if isinstance(other, Coordinate):
            return np.array_equal(self.coords, other.coords)
        return np.array_equal(self.coords, other)

    def __ne__(self, other):
        """Check inequality."""
        return not self.__eq__(other)

    def to_absolute(self) -> 'Coordinate':
        """Converts this coordinate to absolute (identity) coordinate frame.
        
        Applies the cumulative transformation from this coordinate's frame through
        all parent frames to express the coordinate in absolute space.
        
        Returns:
            New Coordinate with coordinates expressed in absolute frame.
        
        Examples:
            >>> root = Frame(transform=translate2D(10, 5))
            >>> child = Frame(transform=translate2D(3, 2), parent=root)
            >>> point = Point(np.array([1, 1]), frame=child)
            >>> absolute_point = point.to_absolute()
            >>> absolute_point.coords  # Should be [14, 8]
        """
        absolute_transform = self.frame.compute_absolute_transform()
        absolute_coords = transform_coordinate(absolute_transform, self.coords, self.kind)
        return Coordinate(coords=absolute_coords, kind=self.kind, frame=None)
        
    def relative_to(self, target_frame: Frame) -> 'Coordinate':
        """Converts this coordinate to a different coordinate frame.
        
        Transforms the coordinate from its current frame to the target frame,
        properly handling the coordinate type (point vs vector) semantics.
        
        Args:
            target_frame: The destination coordinate frame.
        
        Returns:
            New Coordinate with coordinates expressed in the target frame.
        
        Examples:
            >>> frame_a = Frame(transform=translate2D(5, 0))
            >>> frame_b = Frame(transform=translate2D(0, 3))
            >>> point_in_a = Point(np.array([0, 0]), frame=frame_a)
            >>> point_in_b = point_in_a.relative_to(frame_b)
            >>> point_in_b.coords  # Should be [5, -3]
        """
        # Inverse transform from absolute to target frame
        relative_transform = self.frame.compute_relative_transform_to(target_frame)
        relative_coords = transform_coordinate(relative_transform, self.coords, self.kind)
        return Coordinate(coords=relative_coords, kind=self.kind, frame=target_frame)


class Point(Coordinate):
    """Represents a point (position) in a coordinate frame.
    
    Points are affected by all transformations including translation, rotation, and scaling.
    Use this class to represent positions in space.
    
    Args:
        coords: Array-like (numpy array, list, tuple) [x, y] representing the point position,
                     or [[x1, x2, ...], [y1, y2, ...]] for multiple points.
        frame: Coordinate frame this point is defined in. If None, uses absolute frame.
    
    Examples:
        >>> # Point at origin in a translated frame
        >>> frame = Frame(transform=translate2D(10, 5))
        >>> point = Point([0, 0], frame=frame)  # list works
        >>> point = Point((0, 0), frame=frame)  # tuple works
        >>> point = Point(np.array([0, 0]), frame=frame)  # numpy array works
        >>> absolute_point = point.to_absolute()
        >>> absolute_point.coords  # [10, 5] - affected by translation
    """
    
    def __init__(self, coords: ArrayLike, frame: Optional[Frame] = None):
        super().__init__(
            kind=CoordinateKind.POINT,
            coords=coords, 
            frame=frame)
        

class Vector(Coordinate):
    """Represents a vector (direction/displacement) in a coordinate frame.
    
    Vectors are NOT affected by translation, only by rotation and scaling.
    Use this class to represent directions, velocities, or relative displacements.
    
    Args:
        coords: Array-like (numpy array, list, tuple) [x, y] representing the vector components,
                     or [[x1, x2, ...], [y1, y2, ...]] for multiple vectors.
        frame: Coordinate frame this vector is defined in. If None, uses absolute frame.
    
    Examples:
        >>> # Vector in a translated frame
        >>> frame = Frame(transform=translate2D(10, 5))
        >>> vector = Vector([1, 0], frame=frame)  # list works
        >>> vector = Vector((1, 0), frame=frame)  # tuple works
        >>> vector = Vector(np.array([1, 0]), frame=frame)  # numpy array works
        >>> absolute_vector = vector.to_absolute()
        >>> absolute_vector.coords  # Still [1, 0] - unaffected by translation
    """
    
    def __init__(self, coords: ArrayLike, frame: Optional[Frame] = None):
        super().__init__(
            kind=CoordinateKind.VECTOR,
            coords=coords, 
            frame=frame)
