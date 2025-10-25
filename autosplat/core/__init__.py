"""
AutoSplat Core Module
=====================

Core components for differentiable 3D Gaussian splatting and rendering.

This module defines the foundational building blocks used across AutoSplat:
- **Camera**: Handles 3D to 2D projections via intrinsic/extrinsic transformations.
- **Gaussian3D**: Represents trainable 3D Gaussian primitives projected into 2D.
- **BlenderLayer**: Composites multiple Gaussians from a given camera viewpoint
  using differentiable alpha blending.

---------------------------------------------------------------------

BlenderLayer
-------------
    Composites multiple Gaussian3D layers using a Camera.

    Workflow:
        1. Sets a 2x3 projection matrix for each Gaussian from the top 2 rows
           of the camera's Jacobian.
        2. Computes distances from camera to each Gaussian mean and sorts them.
        3. Keeps up to max_gaussians nearest Gaussians.
        4. Performs front-to-back alpha blending in the call() method.

    Warning
    -------
    If `camera_trainable=True`, the projection matrix becomes trainable.
    Use this only when multiple views of a single scene are captured; otherwise,
    intrinsic parameters may drift across layers.

---------------------------------------------------------------------

Camera
-------
    Represents a pinhole camera for 3D Gaussian splatting.

    Attributes
    ----------
    camera_id : int
        Unique identifier for the camera.
    location : np.ndarray, shape (3,)
        3D position of the camera in world coordinates.
    rotation_angles : np.ndarray, shape (3,)
        Euler rotation angles (rx, ry, rz) in radians.
    focus : float
        Focal length of the camera.
    c : np.ndarray, shape (2,)
        Principal point coordinates (cx, cy) in pixels or normalized units.
    rotation_matrix : np.ndarray, shape (3, 3)
        Rotation matrix computed from Euler angles.
    jacobian : np.ndarray, shape (3, 3)
        Intrinsic Jacobian matrix for Gaussian splatting.

---------------------------------------------------------------------

Gaussian3D
-----------
    Trainable 3D Gaussian layer that evaluates 2D points (screen coordinates).

    Workflow:
        1. Applies optional external 3x3 rotation and 3D translation
           to the trainable Gaussian mean and rotation.
        2. Projects the transformed mean/covariance into 2D via a 2x3 Jacobian.
        3. Evaluates Gaussian density values at 2D screen-space coordinates.

    Trainable parameters:
        - s : diagonal scales (sigma) in 3D
        - p : quaternion for 3D rotation
        - mu : 3D mean
        - rgb : color
        - alpha : opacity

    Optional parameters:
        - eps : numerical stability constant for covariance inversion
        - use_default_projection : whether to use a default 2x3 projection matrix
"""

from .blender import BlenderLayer
from .camera import Camera
from .gaussian import Gaussian3D

__all__ = [
    "BlenderLayer",
    "Camera",
    "Gaussian3D",
]

