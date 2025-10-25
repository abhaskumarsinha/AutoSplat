import keras
import matplotlib.pyplot as plt
import numpy as np
import imageio

class RenderObject(keras.layers.Layer):
    """
    RenderObject manages multiple BlenderLayers and renders full RGB images.

    Workflow:
    1. Stores a list of BlenderLayer instances.
    2. Generates a full pixel grid for the target image.
    3. Renders each BlenderLayer into a batch of RGB images.
    4. Provides utilities to preview, save, or animate rendered images.

    Attributes
    ----------
    blend_layers : list of BlenderLayer
        List of BlenderLayer instances managed by this renderer.
    height : int
        Height of the render target.
    width : int
        Width of the render target.
    """

    def __init__(self, blend_layers, image_size=(256, 256), **kwargs):
        """
        Initialize RenderObject.

        Parameters
        ----------
        blend_layers : list
            List of BlenderLayer instances to render.
        image_size : tuple of int, optional
            (height, width) of the render target. Default is (256, 256).
        kwargs : dict
            Additional keyword arguments passed to keras.layers.Layer.
        """
        super().__init__(**kwargs)
        self.blend_layers = list(blend_layers)
        self.height, self.width = map(int, image_size)

    def generate_pixel_grid(self, batch_size=1):
        """
        Generate a tensor of 2D coordinates for the image pixels.

        Parameters
        ----------
        batch_size : int, optional
            Number of batches to generate. Default is 1.

        Returns
        -------
        coords : tensor, shape [B, H*W, 2]
            Normalized pixel coordinates in [-1, 1] x [-1, 1].
        """
        H, W = self.height, self.width
        xs = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        ys = np.linspace(-1.0, 1.0, H, dtype=np.float32)
        xv, yv = np.meshgrid(xs, ys)
        coords = np.stack([xv, yv], axis=-1).reshape(-1, 2)
        coords = np.expand_dims(coords, axis=0)
        coords = np.repeat(coords, batch_size, axis=0)
        return keras.ops.convert_to_tensor(coords, dtype="float32")

    def call(self, blend_id, inputs=None):
        """
        Render RGB images using the specified BlenderLayer(s).

        Parameters
        ----------
        blend_id : int, list, or tensor
            Index of BlenderLayer to render. Can be a single int or a batch of indices.
        inputs : optional
            Placeholder for API compatibility. Not used.

        Returns
        -------
        rgb_imgs : tensor, shape [B, H, W, 3]
            Rendered RGB images.
        """
        if isinstance(blend_id, int):
            blend_ids = keras.ops.expand_dims(keras.ops.convert_to_tensor([blend_id], dtype='int32'), 0)
        else:
            blend_ids = keras.ops.convert_to_tensor(blend_id, dtype='int32')
            if keras.ops.ndim(blend_ids) == 1:
                blend_ids = blend_ids
            elif keras.ops.ndim(blend_ids) == 2 and keras.ops.shape(blend_ids)[-1] == 1:
                blend_ids = keras.ops.squeeze(blend_ids, axis=-1)

        B = keras.ops.shape(blend_ids)[0]
        H, W = self.height, self.width
        N = H * W
        x_2d = self.generate_pixel_grid(B)
        rgb_imgs = []

        for b in range(B):
            layer_idx = int(keras.ops.convert_to_numpy(blend_ids[b]))
            layer = self.blend_layers[layer_idx]
            rgb_flat = layer(x_2d[b:b+1])
            rgb_img = keras.ops.reshape(rgb_flat, (1, H, W, 3))
            rgb_imgs.append(rgb_img)

        return keras.ops.concatenate(rgb_imgs, axis=0)

    def preview(self, blend_id=None, save_path=None, make_gif=False, fps=5):
        """
        Preview or save rendered images and optionally create a GIF.

        Parameters
        ----------
        blend_id : int or None
            Index of BlenderLayer to render. If None, renders the first layer.
        save_path : str or None
            Path to save output images/GIF. If None, images are only displayed.
        make_gif : bool, optional
            If True, saves an animated GIF looping over all layers. Default is False.
        fps : int, optional
            Frames per second for GIF. Default is 5.
        """
        os.makedirs(save_path, exist_ok=True) if save_path else None

        def to_img(x):
            x = np.clip((x + 1) / 2, 0, 1)
            return (x * 255).astype(np.uint8)

        if make_gif:
            frames = []
            for i in range(len(self.blend_layers)):
                img = to_img(keras.ops.convert_to_numpy(self.call(i))[0])
                frames.append(img)
            gif_path = os.path.join(save_path or ".", "preview.gif")
            imageio.mimsave(gif_path, frames, fps=fps, loop=0)
            print(f"🎞️ GIF saved at: {gif_path}")
            return

        if blend_id is None:
            blend_id = 0

        rgb_img = keras.ops.convert_to_numpy(self.call(blend_id=blend_id))[0]
        img = to_img(rgb_img)

        if save_path:
            fname = f"render_{blend_id}.png"
            path = os.path.join(save_path, fname)
            imageio.imwrite(path, img)
            print(f"✅ Saved render at {path}")
        else:
            plt.imshow(img)
            plt.title(f"BlenderLayer {blend_id}")
            plt.axis("off")
            plt.show()

    def __repr__(self):
        return (f"RenderObject(height={self.height}, width={self.width}, "
                f"num_blend_layers={len(self.blend_layers)})")
