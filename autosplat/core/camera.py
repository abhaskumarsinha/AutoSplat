import keras
import numpy as np

class CameraLayer(keras.ops.keras.layers.Layer):
    """
    A differentiable pinhole camera model for 3D Gaussian splatting,
    implemented as a Keras Layer.

    Attributes
    ----------
    camera_id : int
        Unique camera identifier.
    location : keras.ops.Tensor, shape (3,)
        3D position of the camera (non-trainable).
    rotation_angles : keras.ops.Tensor, shape (3,)
        Euler angles (rx, ry, rz) in radians (non-trainable).
    focus : keras.ops.Variable
        Focal length of the camera (trainable if specified).
    c : keras.ops.Variable
        Principal point (cx, cy) (trainable if specified).
    rotation_matrix : keras.ops.Tensor, shape (3, 3)
        Computed rotation matrix.
    jacobian : keras.ops.Tensor, shape (3, 3)
        Intrinsic Jacobian matrix.
    """

    def __init__(
        self,
        camera_id: int = 0,
        location=(0.0, 0.0, 0.0),
        rotation_angles=(0.0, 0.0, 0.0),
        focus=1.0,
        c=(0.0, 0.0),
        train_focus: bool = True,
        train_c: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.camera_id = int(camera_id)

        # Non-trainable attributes
        self.location = np.array(location)
        self.rotation_angles = np.array(rotation_angles)

        # Trainable intrinsics
        self.focus = keras.Variable(
            initial_value=keras.constant(focus, dtype='float32'),
            trainable=train_focus,
            name=f"focus_{camera_id}"
        )

        self.c = keras.Variable(
            initial_value=keras.constant(c, dtype='float32'),
            trainable=train_c,
            name=f"c_{camera_id}"
        )

        # Derived quantities
        self.rotation_matrix = self.compute_rotation_matrix()
        self.jacobian = self.compute_jacobian()

    def compute_rotation_matrix(self):
        """Compute rotation matrix from Euler angles."""
        rx, ry, rz = keras.ops.unstack(self.rotation_angles)

        Rx = keras.ops.stack([
            [1.0, 0.0, 0.0],
            [0.0, keras.ops.cos(rx), -keras.ops.sin(rx)],
            [0.0, keras.ops.sin(rx), keras.ops.cos(rx)]
        ])

        Ry = keras.ops.stack([
            [keras.ops.cos(ry), 0.0, keras.ops.sin(ry)],
            [0.0, 1.0, 0.0],
            [-keras.ops.sin(ry), 0.0, keras.ops.cos(ry)]
        ])

        Rz = keras.ops.stack([
            [keras.ops.cos(rz), -keras.ops.sin(rz), 0.0],
            [keras.ops.sin(rz), keras.ops.cos(rz), 0.0],
            [0.0, 0.0, 1.0]
        ])

        return keras.ops.matmul(keras.ops.matmul(Rz, Ry), Rx)

    def compute_jacobian(self):
        """Compute intrinsic Jacobian matrix."""
        f = self.focus
        cx, cy = keras.ops.unstack(self.c)
        J = keras.ops.stack([
            [f, 0.0, cx],
            [0.0, f, cy],
            [0.0, 0.0, 1.0]
        ])
        return J

    def call(self, inputs=None):
        """
        Optional forward method.
        You can project 3D points here if desired.
        For now, it just returns camera matrices.
        """
        self.rotation_matrix = self.compute_rotation_matrix()
        self.jacobian = self.compute_jacobian()
        return {
            "R": self.rotation_matrix,
            "J": self.jacobian,
            "focus": self.focus,
            "c": self.c
        }

    def get_config(self):
        """Ensure the layer can be serialized."""
        config = super().get_config()
        config.update({
            "camera_id": self.camera_id,
            "location": self.location.numpy().tolist(),
            "rotation_angles": self.rotation_angles.numpy().tolist(),
            "focus": float(self.focus.numpy()),
            "c": self.c.numpy().tolist(),
            "train_focus": self.focus.trainable,
            "train_c": self.c.trainable,
        })
        return config

    def __repr__(self):
        return (
            f"CameraLayer(ID={self.camera_id}, "
            f"focus={self.focus.numpy():.4f}, c={self.c.numpy()}, "
            f"train_focus={self.focus.trainable}, train_c={self.c.trainable})"
        )
