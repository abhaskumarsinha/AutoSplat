import keras
import numpy as np

class BlenderLayer(keras.layers.Layer):
    """
    BlenderLayer: Composites multiple Gaussian3D layers using a Camera.

    Workflow:
    1. Sets a 2x3 projection matrix for each Gaussian from the top 2 rows of the camera's jacobian.
    2. Computes distances from camera to each Gaussian mean and sorts them by distance.
    3. Keeps up to max_gaussians nearest Gaussians.
    4. Performs front-to-back alpha blending in the call() method.

    Warning
    -------
    If `camera_trainable=True`, the projection matrix of this layer is trainable. 
    Use this only when a single scene has been captured from different cameras. 
    Otherwise, keeping the camera trainable may cause inconsistent focal lengths 
    or other intrinsic parameters across layers.

    Attributes
    ----------
    camera : Camera
        Camera object containing location and intrinsic/extrinsic matrices.
    gaussians : list of Gaussian3D
        List of Gaussian3D layers to blend.
    max_gaussians : int
        Maximum number of Gaussians to keep for blending.
    projection : tensor (2,3)
        Trainable 2x3 projection matrix for all Gaussians.
    sorted_gaussians : list of Gaussian3D
        Sorted list of nearest Gaussians.
    _sorted_distances : list of float
        Distances corresponding to sorted Gaussians.
    alpha_blending : placeholder, currently None
        Intended for storing alpha blending function or data.
    background_color : tensor (3,)
        Background RGB color.
    """

    def __init__(self, camera, gaussians, camera_trainable = False, max_gaussians=128, background_color=(-1, -1, -1), **kwargs):
        """
        Initialize BlenderLayer.

        Parameters
        ----------
        camera : Camera
            Camera used for projection and distance computation.
        gaussians : list of Gaussian3D
            List of Gaussians to blend.
        camera_trainable : Train Cameras?
            Train to find focal lengths and principal axes of camera (default False).
        max_gaussians : int, optional
            Maximum number of Gaussians to keep (default 128).
        background_color : tuple or list of 3 floats, optional
            RGB values for the background (default (-1, -1, -1)).
        kwargs : dict
            Additional keyword arguments passed to keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.camera = camera
        self.max_gaussians = int(max_gaussians)
        self.gaussians = list(gaussians)

        cam_J = getattr(self.camera, "jacobian", None)
        if cam_J is None:
            raise ValueError("camera must have attribute 'jacobian' (intrinsics matrix)")

        proj_2x3 = cam_J[:2, :3]  # still in the graph, derived from trainable vars
        self.projection = proj_2x3

        self.sort_gaussians_by_distance()
        self.alpha_blending = None
        self.background_color = keras.ops.convert_to_tensor(background_color, dtype='float32')

    def sort_gaussians_by_distance(self):
        """
        Sort Gaussians by distance from the camera.

        Computes Euclidean distance between the camera location and each Gaussian's mean,
        sorts in ascending order (nearest first), and keeps up to max_gaussians.
        """
        cam_loc = np.array(getattr(self.camera, "location", (0.0, 0.0, 0.0)), dtype=np.float32)
        distances = []

        for g in self.gaussians:
            mu_val = None
            try:
                mu_val = keras.ops.convert_to_numpy(g.mu)
            except Exception:
                try:
                    mu_val = np.array(g.mu, dtype=np.float32)
                except Exception:
                    mu_val = np.zeros((3,), dtype=np.float32)

            mu_val = np.array(mu_val, dtype=np.float32).reshape(3,)
            dist = float(np.linalg.norm(mu_val - cam_loc))
            distances.append((dist, g))

        distances.sort(key=lambda x: x[0])
        self.sorted_gaussians = [g for (_, g) in distances][:self.max_gaussians]
        self._sorted_distances = [d for (d, _) in distances][:self.max_gaussians]

    def call(self, inputs, *args, **kwargs):
        """
        Perform front-to-back alpha blending of sorted Gaussians.

        Parameters
        ----------
        inputs : tensor, shape [B, N, 2] or [N, 2]
            2D coordinates of pixels to evaluate Gaussian splats.

        Returns
        -------
        color_acc : tensor, shape [B, N, 3]
            Blended RGB color for each input point.
        """
        x_2d = keras.ops.convert_to_tensor(inputs, dtype='float32')
        if x_2d.shape[-1] != 2:
            raise ValueError("inputs must have shape [B, N, 2] or [N, 2]")

        if len(x_2d.shape) == 2:
            x_2d = keras.ops.expand_dims(x_2d, axis=0)

        B, N = x_2d.shape[0], x_2d.shape[1]
        bg = keras.ops.reshape(self.background_color, (1, 1, 3))
        color_acc = keras.ops.broadcast_to(bg, (B, N, 3))
        alpha_acc = keras.ops.zeros((B, N, 1), dtype='float32')

        rotation = keras.ops.convert_to_tensor(self.camera.rotation_matrix, dtype='float32')
        translation = keras.ops.convert_to_tensor(self.camera.location, dtype='float32')

        one = keras.ops.ones_like(alpha_acc, dtype='float32')
        zero = keras.ops.zeros_like(alpha_acc, dtype='float32')

        for g in self.sorted_gaussians:
            inv_rotation = keras.ops.transpose(rotation)
            inv_translation = -keras.ops.matmul(inv_rotation, translation)

            g_val = g(x_2d, projection=self.projection, rotation=inv_rotation, translation=inv_translation)
            g_val = keras.ops.expand_dims(g_val, axis=-1)

            alpha = keras.ops.softplus(g.alpha) / (1 + keras.ops.softplus(g.alpha)) * g_val
            color = g.rgb / (1 + keras.ops.abs(g.rgb))
            color = keras.ops.reshape(color, (1, 1, 3))

            color_acc = color_acc * (one - alpha) + color * alpha
            alpha_acc = alpha_acc + (one - alpha_acc) * alpha

        return color_acc

    def __repr__(self):
        return (f"BlenderLayer(camera={self.camera}, max_gaussians={self.max_gaussians},\n"
                f"             num_gaussians={len(self.gaussians)}, projection=\n{self.projection.numpy()},\n"
                f"             background_color={self.background_color.numpy()})")
