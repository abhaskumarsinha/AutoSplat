import numpy as np
import imageio.v3 as iio
from tqdm import tqdm

def render_rotation_gif(renderer, blend_id=0, res=256, radius=2.5, steps=360, save_path="rotation.gif", fps=10):
    """
    Visualize a BlenderLayer by creating a rotating GIF around the object.

    The camera moves in a circle around the Y-axis, looking at the origin,
    and renders frames from the renderer to create an animated GIF.

    Args:
        renderer (RenderObject): Your renderer instance.
        blend_id (int): Index of the BlenderLayer to render.
        res (int): Output resolution (height=width=res).
        radius (float): Distance of camera from origin.
        steps (int): Number of frames (degrees per step = 360 / steps).
        save_path (str): Path to save the GIF.
        fps (int): Frames per second for GIF.
    """
    frames = []

    # Save original renderer size
    orig_H, orig_W = renderer.height, renderer.width
    renderer.height = res
    renderer.width = res

    # Assuming camera is in the first BlenderLayer
    cam = renderer.blend_layers[blend_id].camera
    orig_loc = np.array(cam.location, dtype=np.float32)

    angles = np.linspace(0, 360, steps, endpoint=False)
    for deg in tqdm(angles, desc="Rendering rotation GIF"):
        rad = np.deg2rad(deg)

        # Position camera on circle around Y-axis
        cam_x = radius * np.sin(rad)
        cam_z = radius * np.cos(rad)
        cam.location = np.array([cam_x, orig_loc[1], cam_z], dtype=np.float32)

        # Compute look-at rotation matrix
        forward = -cam.location
        forward /= np.linalg.norm(forward)
        up = np.array([0, 1, 0], dtype=np.float32)
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up_corrected = np.cross(forward, right)
        cam.rotation_matrix = np.stack([right, up_corrected, forward], axis=1).astype(np.float32)

        # Render frame
        rgb_img = keras.ops.convert_to_numpy(renderer.call(blend_id=blend_id))[0]
        img = np.clip((rgb_img + 1) / 2, 0, 1)       # normalize [-1,1] -> [0,1]
        img = (img * 255).astype(np.uint8)
        frames.append(img)

    # Restore original renderer size
    renderer.height = orig_H
    renderer.width = orig_W

    # Save GIF
    iio.imwrite(save_path, frames, fps=fps, loop=0)
    print(f"🎞️ GIF saved at: {save_path}")
