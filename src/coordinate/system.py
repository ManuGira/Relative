"""Coordinate system representation and operations."""

from typing import Optional
import numpy as np

from .transforms import trs2D



class System:
    """Represents a coordinate system with an optional parent system."""
    def __init__(self, transform: Optional[np.ndarray] = None, parent: Optional['System'] = None):
        self.transform = transform if transform is not None else np.eye(3)
        self.parent = parent

    def global_transform(self) -> np.ndarray:
        """Computes the transformation matrix from this system to global space."""
        if self.parent is None:
            return self.transform
        else:
            return self.parent.global_transform() @ self.transform

    def compute_convert_transform(self, target_system: 'System') -> np.ndarray:
        """Computes the transformation matrix from this system to target system."""
        inv_transform = np.linalg.inv(target_system.global_transform())
        return inv_transform @ self.global_transform()


def system_factory(parent: Optional[System], tx: float=0.0, ty: float=0.0, angle_rad: float=0.0, sx: float=1.0, sy: float=1.0) -> System:
    """Factory function to create a System with TRS parameters."""
    transform = trs2D(tx, ty, angle_rad, sx, sy)
    return System(transform=transform, parent=parent)
