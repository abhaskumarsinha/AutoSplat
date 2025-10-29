import keras
import numpy as np

class Camera(keras.layers.Layer):
    """
    Differentiable pinhole camera model for 3D Gaussian splatting.
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
        self.location = keras.ops.convert_to_tensor(location)
        self.rotation_angles = keras.ops.convert_to_tensor(rotation_angles)

        # Trainable intrinsics
        self.focus = self.add_weight(
            shape=(),
            initializer=keras.initializers.Constant(focus),
            dtype="float32",
            trainable=train_focus,
            name=f"focus_{camera_id}",
        )

        self.c = self.add_weight(
            shape=(2,),
            initializer=keras.initializers.Constant(c),
            dtype="float32",
            trainable=train_c,
            name=f"c_{camera_id}",
        )


        # Derived quantities
        self.rotation_matrix = self.compute_rotation_matrix()
        self.jacobian = self.compute_jacobian()

    def compute_rotation_matrix(self):
        """Compute rotation matrix from Euler angles."""
        rx, ry, rz = keras.ops.unstack(self.rotation_angles)

        Rx = keras.ops.stack(
            [
                [1.0, 0.0, 0.0],
                [0.0, keras.ops.cos(rx), -keras.ops.sin(rx)],
                [0.0, keras.ops.sin(rx), keras.ops.cos(rx)],
            ]
        )

        Ry = keras.ops.stack(
            [
                [keras.ops.cos(ry), 0.0, keras.ops.sin(ry)],
                [0.0, 1.0, 0.0],
                [-keras.ops.sin(ry), 0.0, keras.ops.cos(ry)],
            ]
        )

        Rz = keras.ops.stack(
            [
                [keras.ops.cos(rz), -keras.ops.sin(rz), 0.0],
                [keras.ops.sin(rz), keras.ops.cos(rz), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )

        return keras.ops.matmul(keras.ops.matmul(Rz, Ry), Rx)

    def compute_jacobian(self):
        """Compute intrinsic Jacobian matrix."""
        f = self.focus
        cx, cy = keras.ops.unstack(keras.ops.convert_to_tensor(self.c))
        J = keras.ops.stack(
            [
                [f, 0.0, cx],
                [0.0, f, cy],
                [0.0, 0.0, 1.0],
            ]
        )
        return J

    def call(self, inputs=None):
        """Optional forward pass (returns R, J, focus, and c)."""
        self.rotation_matrix = self.compute_rotation_matrix()
        self.jacobian = self.compute_jacobian()
        return {
            "R": self.rotation_matrix,
            "J": self.jacobian,
            "focus": self.focus,
            "c": self.c,
        }

    def get_config(self):
        """Ensure layer can be serialized."""
        config = super().get_config()

        # Safe conversion to numpy then tolist
        loc_np = keras.ops.convert_to_numpy(self.location)
        rot_np = keras.ops.convert_to_numpy(self.rotation_angles)
        c_np = keras.ops.convert_to_numpy(self.c)

        config.update(
            {
                "camera_id": self.camera_id,
                "location": loc_np.tolist(),
                "rotation_angles": rot_np.tolist(),
                "focus": float(keras.ops.convert_to_numpy(self.focus)),
                "c": c_np.tolist(),
            }
        )
        return config

    def __repr__(self):
        """Readable summary for debugging."""
        focus_val = float(keras.ops.convert_to_numpy(self.focus))
        c_val = keras.ops.convert_to_numpy(self.c)
        train_focus = getattr(self.focus, "trainable", False)
        train_c = getattr(self.c, "trainable", False)
        return (
            f"CameraLayer(ID={self.camera_id}, "
            f"focus={focus_val:.4f}, c={c_val.tolist()}, "
            f"train_focus={train_focus}, train_c={train_c})"
        )
