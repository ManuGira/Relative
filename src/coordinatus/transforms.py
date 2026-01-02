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

def swap_axes(dim: int, axis1: int, axis2: int) -> np.ndarray:
    """
    Creates a matrix that swaps two axes in the given dimensional space.
    The returned matrix has shape (dim+1, dim+1) suitable for homogeneous coordinates.
    """
    assert 0 <= axis1 < dim, "axis1 out of bounds"
    assert 0 <= axis2 < dim, "axis2 out of bounds"
    swap_matrix = np.eye(dim+1)
    swap_matrix[[axis1, axis2], :] = swap_matrix[[axis2, axis1], :]
    return swap_matrix

def reduce_dim(initial_dim: int) -> np.ndarray:
    """
    Creates a projection matrix that removes the last axis of the space.
    Then returned matrix has shape (initial_dim, initial_dim + 1) suitable for homogeneous coordinates.
    0 <= axis < initial_dim
    """
    proj = np.eye(initial_dim + 1)
    projection_axis = initial_dim - 1
    proj = np.delete(proj, projection_axis, axis=0)
    return proj


def project_xy_to_x() -> np.ndarray:
    """
    Creates a projection matrix from 2D to 1D by removing the y-axis.
    The returned matrix is a 2x3 matrix suitable for homogeneous coordinates.
    """
    return reduce_dim(2)


def project_xy_to_y() -> np.ndarray:
    """
    Creates a projection matrix from 2D to 1D by removing the x-axis.
    The returned matrix is a 2x3 matrix suitable for homogeneous coordinates.
    """
    return reduce_dim(2) @ swap_axes(2, 0, 1)

def project_xyz_to_xy() -> np.ndarray:
    """
    Creates a projection matrix from 3D to 2D by removing the z-axis.
    The returned matrix is a 3x4 matrix suitable for homogeneous coordinates.
    """
    return reduce_dim(3)

def project_xyz_to_xz() -> np.ndarray:
    """
    Creates a projection matrix from 3D to 2D by removing the y-axis.
    The returned matrix is a 3x4 matrix suitable for homogeneous coordinates.
    """
    y_axis = 1
    z_axis = 2
    return reduce_dim(3) @ swap_axes(3, y_axis, z_axis)

def project_xyz_to_yz() -> np.ndarray:
    """
    Creates a projection matrix from 3D to 2D by removing the x-axis.
    The returned matrix is a 3x4 matrix suitable for homogeneous coordinates.
    """
    x_axis = 0
    z_axis = 2
    return swap_axes(2, 0, 1) @reduce_dim(3) @ swap_axes(3, x_axis, z_axis)

def project_xyz_to_x() -> np.ndarray:
    """
    Creates a projection matrix from 3D to 1D by removing the y and z axes.
    The returned matrix is a 1x4 matrix suitable for homogeneous coordinates.
    """
    return reduce_dim(2) @ reduce_dim(3)

def project_xyz_to_y() -> np.ndarray:
    """
    Creates a projection matrix from 3D to 1D by removing the x and z axes.
    The returned matrix is a 1x4 matrix suitable for homogeneous coordinates.
    """
    x_axis = 0
    y_axis = 1
    return reduce_dim(2) @ reduce_dim(3) @ swap_axes(3, x_axis, y_axis)

def project_xyz_to_z() -> np.ndarray:
    """
    Creates a projection matrix from 3D to 1D by removing the x and y axes.
    The returned matrix is a 1x4 matrix suitable for homogeneous coordinates.
    """
    x_axis = 0
    z_axis = 2
    return reduce_dim(2) @ reduce_dim(3) @ swap_axes(3, x_axis, z_axis)

def augment_dim(initial_dim: int) -> np.ndarray:
    """
    Creates a matrix that adds an extra dimension to the space.
    The returned matrix has shape (initial_dim + 2, initial_dim + 1) suitable for homogeneous coordinates.
    """
    add_dim = np.eye(initial_dim + 2)
    add_dim = np.delete(add_dim, initial_dim, axis=1)
    return add_dim
