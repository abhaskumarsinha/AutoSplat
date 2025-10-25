import numpy as np

class Camera:
    """
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
    """

    def __init__(
        self,
        camera_id: int = 0,
        location: tuple[float, float, float] = (0.0, 0.0, 0.0),
        rotation_angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
        focus: float = 1.0,
        c: tuple[float, float] = (0.0, 0.0)
    ):
        """
        Initialize a Camera instance.

        Parameters
        ----------
        camera_id : int, default=0
            Unique ID of the camera.
        location : tuple of 3 floats, default=(0,0,0)
            Camera position in world coordinates (x, y, z).
        rotation_angles : tuple of 3 floats, default=(0,0,0)
            Euler angles (rx, ry, rz) in radians.
        focus : float, default=1.0
            Camera focal length.
        c : tuple of 2 floats, default=(0,0)
            Principal point coordinates (cx, cy).
        """
        self.camera_id: int = int(camera_id)
        self.location: np.ndarray = np.array(location, dtype=float)
        self.rotation_angles: np.ndarray = np.array(rotation_angles, dtype=float)
        self.focus: float = float(focus)
        self.c: np.ndarray = np.array(c, dtype=float)

        # Compute rotation matrix and intrinsic Jacobian
        self.rotation_matrix: np.ndarray = self.compute_rotation_matrix()
        self.jacobian: np.ndarray = self.compute_jacobian()

    def compute_rotation_matrix(self) -> np.ndarray:
        """
        Compute the 3x3 rotation matrix from Euler angles.

        Returns
        -------
        R : np.ndarray, shape (3, 3)
            Combined rotation matrix: R = Rz * Ry * Rx
        """
        rx, ry, rz = self.rotation_angles

        # Rotation around X-axis
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ], dtype=float)

        # Rotation around Y-axis
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ], dtype=float)

        # Rotation around Z-axis
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ], dtype=float)

        # Combined rotation: R = Rz * Ry * Rx
        R = Rz @ Ry @ Rx
        return R

    def compute_jacobian(self) -> np.ndarray:
        """
        Compute the intrinsic Jacobian matrix for Gaussian splatting.

        Returns
        -------
        J : np.ndarray, shape (3, 3)
            Intrinsic matrix:
                [[f, 0, cx],
                 [0, f, cy],
                 [0, 0,  1]]
        """
        f = self.focus
        cx, cy = self.c
        J = np.array([
            [f, 0, cx],
            [0, f, cy],
            [0, 0, 1]
        ], dtype=float)
        return J

    def __repr__(self) -> str:
        return (
            f"Camera ID: {self.camera_id}\n"
            f"  Location: {self.location}\n"
            f"  Rotation angles (rad): {self.rotation_angles}\n"
            f"  Rotation matrix:\n{self.rotation_matrix}\n"
            f"  Focus: {self.focus}, Principal point: {self.c}\n"
            f"  Jacobian:\n{self.jacobian}\n"
        )
