import os
import numpy as np
import imageio.v3 as iio
from autosplat.core.camera import Camera  # Your Camera class
from copy import deepcopy

from copy import deepcopy
import numpy as np
from typing import Iterable, Tuple

from scipy.linalg import rq
from scipy.spatial.transform import Rotation


def decompose_projection_matrix(P: np.ndarray):
    """
    Decompose 3x4 projection matrix P into intrinsic K (3x3), rotation R (3x3),
    and camera center C (3,).

    Returns (K, R, C) where:
      - P = K @ [R | -R C]
      - K: upper-triangular intrinsic matrix (normalized so K[2,2] == 1)
      - R: rotation matrix (3x3)
      - C: camera center in world coordinates (3,)
    """
    if P.shape != (3, 4):
        raise ValueError("P must be 3x4")

    M = P[:, :3]
    # RQ decomposition yields K (upper-triangular) and R (rotation-like)
    K, R = rq(M)

    # Ensure positive diagonal in K
    diag_sign = np.sign(np.diag(K))
    # Replace zeros with 1 to avoid zero diagonal sign
    diag_sign[diag_sign == 0] = 1.0
    S = np.diag(diag_sign)
    K = K @ S
    R = S @ R

    # Normalize K so that K[2,2] == 1
    if K[2, 2] == 0:
        raise ValueError("Invalid intrinsic K with zero bottom-right entry")
    K = K / K[2, 2]

    # Camera center C computed from P[:,3]
    p3 = P[:, 3]
    # P[:,3] = -K R C  =>  C = - R.T @ K^{-1} @ P[:,3]
    K_inv = np.linalg.inv(K)
    C = -R.T @ (K_inv @ p3)

    # Ensure R is a proper rotation (determinant = +1). If det=-1 flip
    if np.linalg.det(R) < 0:
        R = -R
        K = -K

    return K, R, C


def normalize_cameras_and_intrinsics(cameras, scale_factor: float = 1.0, scale_intrinsics: bool = True, inplace=False):
    """
    Jointly normalize camera locations so max_abs(location) == scale_factor.
    If scale_intrinsics=True, multiply intrinsics (K / focus / c) by same scale s.
    Returns (normalized_cameras, s)
    """
    cams = list(cameras)
    if len(cams) == 0:
        return [], 1.0

    positions = np.stack([cam.location for cam in cams], axis=0)
    max_abs = float(np.max(np.abs(positions)))
    if max_abs == 0:
        max_abs = 1.0
    s = scale_factor / max_abs

    normalized = []
    for cam in cams:
        tgt = cam if inplace else deepcopy(cam)
        tgt.location = np.asarray(tgt.location, dtype=float) * s

        if scale_intrinsics:
            # scale jacobian if present
            if hasattr(tgt, "jacobian") and tgt.jacobian is not None:
                tgt.jacobian = np.asarray(tgt.jacobian, dtype=float) * s
                # update focus and principal point from new jacobian
                tgt.focus = float(tgt.jacobian[0, 0])
                tgt.c = np.array([tgt.jacobian[0, 2], tgt.jacobian[1, 2]])
            else:
                # fall back to scaling focus and c if jacobian isn't present
                if hasattr(tgt, "focus") and tgt.focus is not None:
                    tgt.focus = float(tgt.focus) * s
                if hasattr(tgt, "c") and tgt.c is not None:
                    tgt.c = np.asarray(tgt.c, dtype=float) * s
            # recompute jacobian if required
            if not hasattr(tgt, "jacobian") or tgt.jacobian is None:
                tgt.jacobian = tgt.compute_jacobian()
        else:
            # if not scaling intrinsics, ensure jacobian reflects current focus/c
            tgt.jacobian = tgt.compute_jacobian()

        # recompute rotation matrix (if rotation_angles modified elsewhere)
        tgt.rotation_matrix = tgt.compute_rotation_matrix()

        normalized.append(tgt)

    return normalized


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
