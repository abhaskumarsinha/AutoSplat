import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
import imageio.v3 as iio

def generate_textured_ball_images(res=256, dist=2.0, save=False):
    """
    Generate synthetic textured sphere images from multiple camera views.

    The sphere is procedurally textured using sine-cosine patterns on RGB channels.
    Images are rendered from four canonical camera positions: front, back, left, right.

    Parameters
    ----------
    res : int, optional
        Resolution of the output images (res x res). Default is 256.
    dist : float, optional
        Distance of the camera from the sphere center. Default is 2.0.
    save : bool, optional
        If True, saves each image as a PNG file in the current working directory.
        Filenames are 'front.png', 'back.png', 'left.png', 'right.png'. Default is False.

    Returns
    -------
    images : np.ndarray, shape (4, res, res, 3)
        Stack of 4 RGB images corresponding to the four camera views.
        Channels are in [0, 255] uint8 format.

    Example
    -------
    >>> imgs = generate_textured_ball_images(res=64, dist=1.0, save=True)
    >>> print(imgs.shape)
    (4, 64, 64, 3)
    """
    # Sphere coordinates
    phi, theta = np.mgrid[0:np.pi:256j, 0:2*np.pi:256j]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)

    # Procedural sine-cosine texture
    tex_r = 0.5 + 0.5 * np.sin(5 * theta) * np.cos(3 * phi)
    tex_g = 0.5 + 0.5 * np.cos(4 * theta + phi)
    tex_b = 0.5 + 0.5 * np.sin(phi * 2)
    texture = np.stack([tex_r, tex_g, tex_b], axis=-1)

    # Camera positions
    cameras = [
        ("front", (0, 0, dist), (0, 0, 0)),
        ("back", (0, 0, -dist), (0, 0, 0)),
        ("left", (-dist, 0, 0), (0, 0, 0)),
        ("right", (dist, 0, 0), (0, 0, 0)),
    ]

    images = []

    for name, cam_pos, target in cameras:
        fig = plt.figure(figsize=(2.56, 2.56), dpi=100, facecolor='black')
        ax = fig.add_subplot(111, projection='3d', facecolor='black')

        # Plot the sphere with texture
        ax.plot_surface(
            x, y, z, rstride=1, cstride=1, facecolors=texture,
            linewidth=0, antialiased=False, shade=False
        )

        # Camera view
        ax.view_init(elev=0, azim={
            "front": 180,
            "back": 0,
            "left": -90,
            "right": 90,
        }[name])

        # Limits and hide axes
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.axis("off")

        # Render to image
        fig.canvas.draw()
        img_full = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)[..., :3]

        # Resize to desired resolution
        img = resize(img_full, (res, res, 3), preserve_range=True).astype(np.uint8)
        plt.close(fig)

        images.append(img)
        if save:
            iio.imwrite(f"{name}.png", img)

    return np.stack(images, axis=0)  # [num_cameras, res, res, 3]

