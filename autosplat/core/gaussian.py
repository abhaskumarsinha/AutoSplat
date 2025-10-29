import keras
import numpy as np

class Gaussian3D(keras.layers.Layer):
    """
    3D Gaussian layer for differentiable splatting and projection into 2D space.

    This layer represents a trainable 3D Gaussian distribution that can be 
    transformed, projected, and evaluated at given 2D screen-space points.

    ---
    **Workflow**
    1. Apply an optional *external* 3×3 rotation and 3-vector translation 
       to the *trainable* 3D Gaussian mean and rotation.
    2. Project the transformed 3D mean and covariance into 2D using a 
       2×3 Jacobian (either externally provided or a default one).
    3. Evaluate the resulting 2D Gaussian at input points.

    ---
    **Trainable Parameters**
    - `s` : (3,) — Diagonal standard deviations (scales σₓ, σᵧ, σ_z)
    - `p` : (4,) — Quaternion representing 3D rotation
    - `mu` : (3,) — 3D mean of the Gaussian (initializer: "Uniform" or "Normal")
    - `rgb` : (3,) — Per-Gaussian color
    - `alpha` : (1,) — Opacity (transparency control)

    ---
    **Optional Parameters**
    - `eps` : float — Small numerical constant to stabilize matrix inversions.
    - `use_default_projection` : bool — Whether to use an identity-like 2×3 
      projection matrix when none is provided.
    - `mu_initializer` : {"Uniform", "Normal"} — Distribution type for initializing
      the 3D mean. "Uniform" samples from [-1, 1], "Normal" uses stddev = 0.5.
    """

    def __init__(self, eps=1e-6, use_default_projection=True, mu_initializer='random_uniform', max_value = 0.1, **kwargs):
        """
        Initialize a 3D Gaussian.

        Parameters
        ----------
        eps : float, optional (default=1e-6)
            Small epsilon value added for numerical stability in covariance inversion.
        use_default_projection : bool, optional (default=True)
            If True, uses an internal 2×3 default projection when none is provided.
        mu_initializer : str, optional (default="Uniform")
            Initialization mode for the Gaussian mean (`mu`).
            - "random_uniform" → samples uniformly from [-1.0, 1.0]
            - "random_normal"  → samples from a normal distribution with stddev=0.5
        kwargs : dict
            Additional keyword arguments passed to `keras.layers.Layer`.
        """
        super().__init__(**kwargs)
        self.eps = float(eps)

        # Trainable diagonal scales (sigma)
        self.s = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=0.05, maxval=0.2)(shape=(3,), dtype='float32', kernel_constraint=keras.constraints.MaxNorm(max_value)),
            trainable=True, name='s_scale'
        )

        # Trainable quaternion parameters (small init -> near identity)
        self.p = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=-0.05, maxval=0.05)(shape=(4,), dtype='float32'),
            trainable=True, name='p_rot'
        )
        
        if mu_initializer == 'random_uniform':
            # Trainable 3D mean (Uniform)
            self.mu = keras.Variable(
                keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)(
                    shape=(3,), dtype='float32'
                ),
                trainable=True,
                name='mu'
            )
        
        elif mu_initializer == 'random_normal':
            # Trainable 3D mean (Normal)
            self.mu = keras.Variable(
                keras.initializers.RandomNormal(stddev=0.5)(
                    shape=(3,), dtype='float32'
                ),
                trainable=True,
                name='mu'
            )
        
        else:
            raise ValueError(f"Unknown initializer type: {mu_initializer}")

        # Trainable Color
        self.rgb = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=-1.0, maxval=1.0)(shape=(3,), dtype='float32'),
            trainable=True, name='rgb'
        )

        # Trainable Alpha (opacity)
        self.alpha = keras.Variable(
            initializer=keras.initializers.RandomUniform(minval=0.7, maxval=1.0)(shape=(1,), dtype='float32'),
            trainable=True, name='alpha'
        )

        # Optional default projection J (2×3)
        self.use_default_projection = bool(use_default_projection)
        self.default_projection = keras.Variable(
            [[1., 0., 0.],
             [0., 1., 0.]], dtype='float32', trainable=False
        )

    def quaternion_to_rotation(self, p):
        """
        Convert a quaternion into a 3×3 rotation matrix.

        Parameters
        ----------
        p : tensor, shape (4,)
            Quaternion coefficients [q₀, q₁, q₂, q₃].

        Returns
        -------
        R : tensor, shape (3, 3)
            Corresponding rotation matrix.
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

        R = keras.ops.stack([row0, row1, row2], axis=0)
        return R

    def call(self, x_2d, projection=None, rotation=None, translation=None):
        """
        Evaluate the Gaussian at 2D input points.

        Parameters
        ----------
        x_2d : tensor, shape (..., 2)
            Input 2D coordinates at which to evaluate the Gaussian.
        projection : tensor, shape (2, 3), optional
            Projection matrix from 3D to 2D. If None, uses default if enabled.
        rotation : tensor, shape (3, 3), optional
            External rotation applied over the trainable rotation.
        translation : tensor, shape (3,), optional
            External translation vector applied after rotation.

        Returns
        -------
        gaussian : tensor, shape (...)
            Computed Gaussian intensity values for each input 2D point.
        """
        # Implementation unchanged
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
        """
        Return a readable string representation of the Gaussian3D layer.
        """
        return (f"Gaussian3D(eps={self.eps}, use_default_projection={self.use_default_projection},\n"
                f"          s={self.s.numpy()}, p={self.p.numpy()}, mu={self.mu.numpy()},\n"
                f"          rgb={self.rgb.numpy()}, alpha={self.alpha.numpy()})")
