import os
import numpy as np
import imageio.v3 as iio
from autosplat.core.camera import Camera  # Your Camera class
from copy import deepcopy

from copy import deepcopy
import numpy as np
from typing import Iterable, Tuple

def normalize_cameras_and_intrinsics(
    cameras: Iterable[Camera],
    scale_factor: float = 1.0,
    scale_intrinsics: bool = False,
    inplace: bool = False
) -> Tuple[list, float]:
    """
    Normalize camera locations to fit within [-scale_factor, scale_factor] across all axes,
    and optionally scale intrinsics (focus and principal point) by the same factor.

    Parameters
    ----------
    cameras : iterable of Camera
        Input Camera objects.
    scale_factor : float
        The desired half-range after normalization; positions are scaled so that
        max(abs(positions)) == scale_factor.
    scale_intrinsics : bool
        If True, multiply camera.focus and camera.c by the same scale. Use this
        only when intrinsics are in the same *world* units as camera locations.
    inplace : bool
        If True, modify the provided Camera objects in-place; otherwise a deepcopy
        is returned.

    Returns
    -------
    normalized_cameras : list[Camera]
        New list of normalized Camera objects (or the same objects when inplace=True).
    applied_scale : float
        The numeric scale s that was applied (new_location = old_location * s).
    """
    cams = list(cameras)
    if len(cams) == 0:
        return [], 1.0

    # stack positions: shape (N, 3)
    positions = np.stack([cam.location for cam in cams], axis=0)
    max_abs_val = np.max(np.abs(positions))
    if max_abs_val == 0:
        max_abs_val = 1.0

    s = (scale_factor / max_abs_val)

    normalized = []
    for cam in cams:
        target_cam = cam if inplace else deepcopy(cam)

        # scale translation/location
        target_cam.location = np.asarray(target_cam.location, dtype=float) * s

        # optionally scale intrinsics (focus and principal point) by same s
        if scale_intrinsics:
            # protect against missing attributes or None
            if hasattr(target_cam, "focus") and target_cam.focus is not None:
                try:
                    target_cam.focus = float(target_cam.focus) * s
                except Exception:
                    # keep original if it can't be scaled
                    pass
            if hasattr(target_cam, "c") and target_cam.c is not None:
                try:
                    target_cam.c = np.asarray(target_cam.c, dtype=float) * s
                except Exception:
                    pass

        # recompute cached derived fields if Camera stores them
        if hasattr(target_cam, "compute_rotation_matrix"):
            try:
                target_cam.rotation_matrix = target_cam.compute_rotation_matrix()
            except Exception:
                pass
        if hasattr(target_cam, "compute_jacobian"):
            try:
                target_cam.jacobian = target_cam.compute_jacobian()
            except Exception:
                pass

        normalized.append(target_cam)

    return normalized, s


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
