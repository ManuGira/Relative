"""Transformation matrix utilities for 2D affine transformations."""

import numpy as np


def translate2D(tx: float, ty: float) -> np.ndarray:
    """Creates a 2D translation matrix."""
    return np.array([[1, 0, tx],
                     [0, 1, ty],
                     [0, 0, 1]])


def rotate2D(angle_rad: float) -> np.ndarray:
    """Creates a 2D rotation matrix."""
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    return np.array([[c, -s, 0],
                     [s,  c, 0],
                     [0,  0, 1]])


def scale2D(sx: float, sy: float) -> np.ndarray:
    """Creates a 2D scaling matrix."""
    return np.array([[sx, 0,  0],
                     [0, sy,  0],
                     [0,  0, 1]])

def shear2D(kx: float, ky: float) -> np.ndarray:
    """Creates a 2D shear matrix."""
    return np.array([[1, kx, 0],
                     [ky, 1, 0],
                     [0,  0, 1]])


def trs2D(tx: float, ty: float, angle_rad: float, sx: float, sy: float) -> np.ndarray:
    """Creates a combined translation, rotation, and scaling matrix."""
    T = translate2D(tx, ty)
    R = rotate2D(angle_rad)
    S = scale2D(sx, sy)
    return T @ R @ S

def trks2D(tx: float, ty: float, angle_rad: float, kx: float, ky: float, sx: float, sy: float) -> np.ndarray:
    """Creates a combined translation, rotation, shear, and scaling matrix."""
    T = translate2D(tx, ty)
    R = rotate2D(angle_rad)
    K = shear2D(kx, ky)
    S = scale2D(sx, sy)
    return T @ R @ K @ S


def projection(initial_dim: int, axis: int) -> np.ndarray:
    """
    Creates a projection matrix that removes the specified axes. 
    Then returned matrix has shape (initial_dim, initial_dim + 1) suitable for homogeneous coordinates.
    0 <= axis < initial_dim
    """
    assert initial_dim > axis, "Initial dimension must be greater than the axis to project."
    proj = np.eye(initial_dim + 1)
    proj = np.delete(proj, axis, axis=0)  # Remove the specified
    return proj


def projection2D_to_1D(axis: int) -> np.ndarray:
    """
    Creates a projection matrix from 2D to 1D by removing the specified axis (0=x, 1=y).
    The returned matrix is a 2x3 matrix suitable for homogeneous coordinates.
    """
    return projection(2, axis)


def projection3D_to_2D(axis: int) -> np.ndarray:
    """
    Creates a projection matrix from 3D to 2D by removing the specified axis (0=x, 1=y, 2=z).
    The returned matrix is a 3x4 matrix suitable for homogeneous coordinates.
    """
    return projection(3, axis)




