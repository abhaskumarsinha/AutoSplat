import os
import numpy as np
import imageio.v3 as iio
from autosplat.core.camera import Camera  # Your Camera class
from copy import deepcopy

def normalize_camera_positions(cameras, scale_factor: float = 1.0):
    """
    Normalize camera locations jointly to fit within [-a, a] range,
    maintaining aspect ratio across all axes.

    Parameters
    ----------
    cameras : list[Camera]
        List of Camera instances to normalize.
    scale_factor : float, default=1.0
        Scale the normalized range to [-scale_factor, scale_factor].

    Returns
    -------
    list[Camera]
        New list of normalized Camera objects.
    """
    if not cameras:
        return []

    # Stack all camera positions
    positions = np.stack([cam.location for cam in cameras], axis=0)

    # Find the max absolute coordinate value (joint normalization)
    max_abs_val = np.max(np.abs(positions))
    if max_abs_val == 0:
        max_abs_val = 1.0  # avoid division by zero

    # Normalize all positions keeping relative scale
    normalized_positions = (positions / max_abs_val) * scale_factor

    # Create new cameras (deepcopy to avoid mutating originals)
    normalized_cameras = []
    for cam, new_pos in zip(cameras, normalized_positions):
        cam_copy = deepcopy(cam)
        cam_copy.location = new_pos
        cam_copy.jacobian = cam_copy.jacobian/(positions / max_abs_val)
        normalized_cameras.append(cam_copy)

    return normalized_cameras


def load_dataset(folder, image_size=(256, 256), skip_seeds=True, rotate_images=False):
    """
    Load images and cameras from dataset folder.
    
    Download *Buddha Dataset* using:
        git clone https://github.com/alicevision/dataset_buddha.git
    and use `buddha` or `buddha_mini6` as the corresponding folder.

    Args:
        folder (str): Path to dataset folder containing *_c.png and *_P.txt files.
        image_size (tuple): Desired output size (H, W) for images.
        skip_seeds (bool): Whether to ignore _seeds.bin files.
        rotate_images (bool): If True, rotates all images by 90° clockwise.

    Returns:
        images (list of np.ndarray): List of images resized to image_size.
        cameras (list of Camera): List of Camera objects.
    """
    images = []
    cameras = []

    # Sort to keep ordering consistent
    files = sorted(os.listdir(folder))
    
    # Find all images
    img_files = [f for f in files if f.endswith('_c.png')]
    
    for img_file in img_files:
        # Corresponding camera file
        base_name = img_file.split('_c.png')[0].rstrip('.')
        cam_file = os.path.join(folder, f"{base_name}_P.txt")
        img_path = os.path.join(folder, img_file)

        # Load image
        img = iio.imread(img_path)

        # Rotate if enabled
        if rotate_images:
            img = np.rot90(img, k=-1)  # 90° clockwise

        # Resize using simple nearest-neighbor / PIL if needed
        if img.shape[:2] != image_size:
            from PIL import Image
            img = np.array(Image.fromarray(img).resize(image_size[::-1], Image.BILINEAR))
        
        images.append(img)

        # Load camera
        cam_matrix = np.loadtxt(cam_file)
        if cam_matrix.shape != (3, 4):
            raise ValueError(f"Unexpected camera shape {cam_matrix.shape} in {cam_file}")

        # Parse rotation + translation
        R = cam_matrix[:, :3]
        t = cam_matrix[:, 3]

        # Compute Euler angles from R if needed
        import scipy.spatial.transform
        r = scipy.spatial.transform.Rotation.from_matrix(R)
        euler_angles = r.as_euler('xyz', degrees=False)
        quat = r.as_quat()  # [x, y, z, w]

        # Compute focal length and principal point if available
        focus = np.linalg.norm(R[0])  # crude approximation
        c = (0.0, 0.0)  # placeholder if not provided

        cam = Camera(
            camera_id=int(base_name),
            location=t,
            rotation_angles=euler_angles,
            focus=focus,
            c=c
        )
        cameras.append(cam)

    return images, cameras
