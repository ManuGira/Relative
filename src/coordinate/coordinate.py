"""Coordinate representation classes for points and vectors."""

from typing import Optional
import numpy as np

from .system import System
from .types import CoordinateType


def transform_coordinate(transform: np.ndarray, coordinates: np.ndarray, coordinate_type: CoordinateType) -> np.ndarray:
    """Applies an affine transformation to a coordinate point or vector."""
    # Convert to homogeneous coordinates
    weight = 1.0 if coordinate_type == CoordinateType.POINT else 0.0
    homogeneous_point = np.append(coordinates, weight) 
    transformed_point = transform @ homogeneous_point

    # Return to Cartesian coordinates
    # Normalize if necessary
    weight = transformed_point[2]
    if weight != 0:
        transformed_point /= weight
    return transformed_point[:2]


class Coordinate:
    """Base class for coordinate representations in a coordinate system."""

    def __init__(self, coordinate_type: CoordinateType, local_coords: np.ndarray, system: Optional[System] = None):
        self.coordinate_type = coordinate_type
        self.local_coords = local_coords
        self.system = system if system is not None else System()

    def to_global(self) -> 'Coordinate':
        """Converts this coordinate to the global coordinate system."""
        global_transform = self.system.global_transform()
        global_coords = transform_coordinate(global_transform, self.local_coords, self.coordinate_type)
        return Coordinate(local_coords=global_coords, coordinate_type=self.coordinate_type, system=None)
        
    def to_system(self, target_system: System) -> 'Coordinate':
        """Converts this coordinate to a target coordinate system."""
        # Inverse transform from global to target system
        convert_transform = self.system.compute_convert_transform(target_system)
        new_local_coords = transform_coordinate(convert_transform, self.local_coords, self.coordinate_type)
        return Coordinate(local_coords=new_local_coords, coordinate_type=self.coordinate_type, system=target_system)


class Point(Coordinate):
    """Represents a point in a coordinate system."""
    
    def __init__(self, local_coords: np.ndarray, system: Optional[System] = None):
        super().__init__(
            coordinate_type=CoordinateType.POINT,
            local_coords=local_coords, 
            system=system)


class Vector(Coordinate):
    """Represents a vector in a coordinate system."""
    
    def __init__(self, local_coords: np.ndarray, system: Optional[System] = None):
        super().__init__(
            coordinate_type=CoordinateType.VECTOR,
            local_coords=local_coords, 
            system=system)
