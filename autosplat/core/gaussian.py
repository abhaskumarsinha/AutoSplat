import keras
import numpy as np

class Gaussian3D(keras.layers.Layer):
    """
    3D Gaussian layer for splatting that evaluates only 2D points (screen coordinates).

    Workflow:
    1. Apply an optional external 3x3 rotation and 3-vector translation to the 
       trainable 3D Gaussian (mean and rotation).
    2. Project the resulting 3D mean/covariance into 2D using a 2x3 Jacobian 
       (either provided externally or default).
    3. Evaluate Gaussian values at 2D input points.

    Trainable parameters:
        - s : diagonal scales (sigma) in 3D
        - p : quaternion for 3D rotation
        - mu : 3D mean
        - rgb : color
        - alpha : opacity

    Optional parameters:
        - eps : small value for numerical stability in covariance inversion
        - use_default_projection : whether to use a default 2x3 projection matrix
    """

    def __init__(self, eps=1e-6, use_default_projection=True, **kwargs):
        """
        Initialize a 3D Gaussian.

        Parameters
        ----------
        eps : float
            Small epsilon for numerical stability in covariance inversion.
        use_default_projection : bool
            If True, uses identity-like 2x3 default projection when none is provided.
        kwargs : dict
            Additional keyword arguments passed to keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.eps = float(eps)

        # Trainable diagonal scales (sigma)
        self.s = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=0.05, maxval=0.2)(shape=(3,), dtype='float32'),
            trainable=True, name='s_scale'
        )

        # Trainable quaternion parameters (small init -> near identity)
        self.p = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)(shape=(4,), dtype='float32'),
            trainable=True, name='p_rot'
        )

        # Trainable 3D mean
        self.mu = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)(shape=(3,), dtype='float32'),
            trainable=True, name='mu'
        )

        # Trainable Color
        self.rgb = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)(shape=(3,), dtype='float32'),
            trainable=True, name='rgb'
        )

        # Trainable Alpha
        self.alpha = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=0.7, maxval=1.0)(shape=(1,), dtype='float32'),
            trainable=True, name='alpha'
        )

        # Optional default projection J (2x3) if caller doesn't provide one.
        self.use_default_projection = bool(use_default_projection)
        self.default_projection = keras.Variable([[1., 0., 0.],
                                               [0., 1., 0.]], dtype='float32', trainable = False)  # takes x,y

    def quaternion_to_rotation(self, p):
        """
        Convert 4-vector quaternion to 3x3 rotation matrix.

        Parameters
        ----------
        p : tensor, shape (4,)
            Quaternion vector.

        Returns
        -------
        R : tensor, shape (3,3)
            Rotation matrix corresponding to the unit quaternion.
        """
        norm = keras.ops.sqrt(keras.ops.sum(p * p) + 1e-12)
        q = p / norm

        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]

        row0 = keras.ops.stack([1.0 - 2.0 * (q2 * q2 + q3 * q3),
                                2.0 * (q1 * q2 - q0 * q3),
                                2.0 * (q1 * q3 + q0 * q2)], axis=0)

        row1 = keras.ops.stack([2.0 * (q1 * q2 + q0 * q3),
                                1.0 - 2.0 * (q1 * q1 + q3 * q3),
                                2.0 * (q2 * q3 - q0 * q1)], axis=0)

        row2 = keras.ops.stack([2.0 * (q1 * q3 - q0 * q2),
                                2.0 * (q2 * q3 + q0 * q1),
                                1.0 - 2.0 * (q1 * q1 + q2 * q2)], axis=0)

        R = keras.ops.stack([row0, row1, row2], axis=0)  # shape (3,3)
        return R

    def call(self, x_2d, projection=None, rotation=None, translation=None):
        """
        Evaluate the Gaussian at 2D points.

        Parameters
        ----------
        x_2d : tensor, shape [..., 2]
            Input 2D points for evaluation.
        projection : tensor, shape (2,3), optional
            2x3 projection matrix. If None and use_default_projection=True, uses default.
        rotation : tensor, shape (3,3), optional
            External 3x3 rotation matrix applied over trainable rotation.
        translation : tensor, shape (3,), optional
            External 3-vector translation applied after rotation.

        Returns
        -------
        gaussian : tensor, shape [...]
            Gaussian values at input 2D points.
        """
        # original implementation remains unchanged
        x_2d = keras.ops.convert_to_tensor(x_2d, dtype='float32')
        if keras.ops.ndim(x_2d) is None:
            last_dim = keras.ops.shape(x_2d)[-1]
            if int(last_dim) != 2:
                raise ValueError("x_2d must have last dimension = 2")
        else:
            if x_2d.shape[-1] != 2:
                raise ValueError(f"x_2d must have last dim = 2 (2D points). Got {x_2d.shape}")

        if projection is None:
            if not self.use_default_projection:
                raise ValueError("projection (2x3) must be provided")
            proj = self.default_projection
        else:
            proj = keras.ops.convert_to_tensor(projection, dtype='float32')
            if tuple(proj.shape) != (2, 3):
                raise ValueError(f"projection must be shape (2,3), got {tuple(proj.shape)}")

        if rotation is None:
            rotation_ext = keras.ops.eye(3, dtype='float32')
        else:
            rotation_ext = keras.ops.convert_to_tensor(rotation, dtype='float32')
            if tuple(rotation_ext.shape) != (3, 3):
                raise ValueError("rotation must be shape (3,3)")

        if translation is None:
            translation = keras.ops.zeros((3,), dtype='float32')
        translation = keras.ops.convert_to_tensor(translation, dtype='float32')
        if tuple(translation.shape) != (3,):
            raise ValueError("translation must be shape (3,)")

        R_train = self.quaternion_to_rotation(self.p)
        R_total = keras.ops.matmul(rotation_ext, R_train)
        S_inv2 = keras.ops.diag(1.0 / (self.s * self.s + self.eps))
        mu3 = keras.ops.matmul(R_total, keras.ops.expand_dims(self.mu, -1))
        mu3 = keras.ops.squeeze(mu3, axis=-1) + translation
        u0 = keras.ops.einsum('ij,j->i', proj, mu3)
        tmp = keras.ops.matmul(proj, R_total)
        tmp = keras.ops.matmul(tmp, S_inv2)
        tmp = keras.ops.matmul(tmp, keras.ops.transpose(R_total))
        sigma_inv_2d = keras.ops.matmul(tmp, keras.ops.transpose(proj))
        delta = x_2d - u0
        dist = keras.ops.einsum('...i,ij,...j->...', delta, sigma_inv_2d, delta)
        gaussian = keras.ops.exp(-0.5 * dist)
        return gaussian

    def __repr__(self):
        return (f"Gaussian3D(eps={self.eps}, use_default_projection={self.use_default_projection},\n"
                f"          s={self.s.numpy()}, p={self.p.numpy()}, mu={self.mu.numpy()},\n"
                f"          rgb={self.rgb.numpy()}, alpha={self.alpha.numpy()})")
