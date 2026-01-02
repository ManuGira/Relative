"""Dimension manipulation and projection transformation utilities."""

import numpy as np


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


def augment_dim(initial_dim: int) -> np.ndarray:
    """
    Creates a matrix that adds an extra dimension to the space.
    The returned matrix has shape (initial_dim + 2, initial_dim + 1) suitable for homogeneous coordinates.
    """
    add_dim = np.eye(initial_dim + 2)
    add_dim = np.delete(add_dim, initial_dim, axis=1)
    return add_dim


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
    return swap_axes(2, 0, 1) @ reduce_dim(3) @ swap_axes(3, x_axis, z_axis)


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
