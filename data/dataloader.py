import os
import numpy as np
import imageio.v3 as iio
from autosplat.core.camera import Camera  # Your Camera class
from copy import deepcopy

import os
import numpy as np
import imageio as iio
from PIL import Image
from scipy.linalg import rq
from scipy.spatial.transform import Rotation as R

def decompose_projection_matrix(P):
    """
    Decompose 3x4 projection P into (K, R, C)
    - K: 3x3 intrinsic (K[2,2] normalized to 1, diag positive)
    - R: 3x3 rotation matrix
    - C: 3-vector camera center in world coordinates
    """
    P = np.asarray(P, dtype=float)
    if P.shape != (3, 4):
        raise ValueError(f"Projection matrix must be (3,4), got {P.shape}")

    M = P[:, :3].copy()
    p4 = P[:, 3].copy()

    # RQ decomposition (M = K @ R)
    K, Rmat = rq(M)

    # Force positive diagonal on K
    diag_sign = np.sign(np.diag(K))
    diag_sign[diag_sign == 0] = 1.0
    T = np.diag(diag_sign)
    K = K @ T
    Rmat = T @ Rmat

    # normalize so K[2,2] == 1
    if K[2, 2] == 0:
        raise RuntimeError("Invalid K (K[2,2] == 0)")
    K = K / K[2, 2]

    # camera center C = -R^T * (K^{-1} * p4)
    t_cam = np.linalg.inv(K) @ p4
    C = -Rmat.T @ t_cam

    return K, Rmat, C


def camera_from_decomposed(camera_id, K, Rmat, C):
    """
    Build Camera instance from decomposed parts.
    `K` should already be the final intrinsics in pixel/normalized units
    appropriate for the final resized image and world-scale.
    """
    fx = K[0, 0]
    fy = K[1, 1]
    # Use mean focal to produce a single scalar focus (preserves aspect roughly)
    f = float((fx + fy) / 2.0)
    cx = float(K[0, 2])
    cy = float(K[1, 2])

    rot = R.from_matrix(Rmat)
    euler = rot.as_euler('xyz', degrees=False)

    # Attempt to convert camera_id to int but fallback to string index
    try:
        cam_id_int = int(camera_id)
    except Exception:
        cam_id_int = camera_id

    cam = Camera(
        camera_id=cam_id_int,
        location=C.astype(float),
        rotation_angles=euler.astype(float),
        focus=f,
        c=(cx, cy)
    )
    return cam


def normalize_camera_projection_list(P_list, camera_ids=None, image_scales=None, verbose=False):
    """
    Decompose all P in P_list, compute global centroid and scale so all camera centers lie
    in [-1,1] (global cube). Scale intrinsics by image_scales (if provided) then by the
    same world-scale factor so projection geometry is consistent.

    Args:
        P_list: list of (3,4) projection matrices (np.ndarray)
        camera_ids: optional list of ids to use for Camera.camera_id
        image_scales: optional list of (s_x, s_y) image resize scales for each camera.
                      If None, assumes (1.0, 1.0) for each camera.

    Returns:
        cameras: list[Camera]
        meta: dict with 'centroid' and 'scale'
    """
    n = len(P_list)
    if camera_ids is None:
        camera_ids = list(range(n))
    if image_scales is None:
        image_scales = [(1.0, 1.0)] * n

    Ks = []
    Rmats = []
    Cs = []
    for i, P in enumerate(P_list):
        K, Rmat, C = decompose_projection_matrix(P)
        Ks.append(K)
        Rmats.append(Rmat)
        Cs.append(C)

    Cs = np.stack(Cs, axis=0)  # shape (N,3)

    # compute centroid and scale so all camera centers fit in [-1,1]
    centroid = Cs.mean(axis=0)
    Cs_centered = Cs - centroid[None, :]
    max_abs = np.max(np.abs(Cs_centered))
    scale = 1.0 / max_abs if max_abs != 0 else 1.0

    cameras = []
    for i in range(n):
        K_orig = Ks[i].copy()
        Rmat = Rmats[i].copy()
        C_orig = Cs[i].copy()

        # apply image resize scale to K (does not change world center)
        s_x, s_y = image_scales[i]
        K_scaled = K_orig.copy()
        K_scaled[0, 0] *= s_x  # fx
        K_scaled[1, 1] *= s_y  # fy
        K_scaled[0, 2] *= s_x  # cx
        K_scaled[1, 2] *= s_y  # cy

        # apply world normalization scale to intrinsics
        K_scaled[0, 0] *= scale
        K_scaled[1, 1] *= scale
        K_scaled[0, 2] *= scale
        K_scaled[1, 2] *= scale

        # transform camera center into normalized coordinate frame
        C_new = (C_orig - centroid) * scale

        cam = camera_from_decomposed(camera_id=camera_ids[i], K=K_scaled, Rmat=Rmat, C=C_new)
        cameras.append(cam)

        if verbose:
            print(f"[normalize] id={camera_ids[i]} orig_C={C_orig} -> C_new={C_new}, image_scale=({s_x:.4f},{s_y:.4f})")

    meta = {'centroid': centroid, 'scale': scale}
    return cameras, meta


def load_dataset(folder, image_size=(256, 256), skip_seeds=True, rotate_images=False):
    """
    Load images and normalized cameras from dataset folder.

    Returns:
        images (list[np.ndarray]), cameras (list[Camera])
    The returned `cameras` are normalized globally so all camera locations are in [-1,1],
    and intrinsics (focus, principal point) are scaled accordingly. This preserves the
    same return format as the previous implementation.
    """
    images = []
    cameras = []

    files = sorted(os.listdir(folder))

    # find image files (ending with '_c.png')
    img_files = [f for f in files if f.endswith('_c.png')]
    # optionally ignore seeds-like names (if any weird files present)
    if skip_seeds:
        img_files = [f for f in img_files if '_seeds' not in f]

    P_list = []
    ids = []
    image_scales = []

    for idx, img_file in enumerate(img_files):
        base_name = img_file[:-6]  # remove trailing '_c.png'
        cam_file = os.path.join(folder, f"{base_name}_P.txt")
        img_path = os.path.join(folder, img_file)

        if not os.path.exists(cam_file):
            raise FileNotFoundError(f"Expected camera file {cam_file} for image {img_path}")

        # --- load image ---
        img = iio.imread(img_path)

        # rotate before resizing (keeps pixel scale logic straightforward)
        if rotate_images:
            img = np.rot90(img, k=-1)  # 90° clockwise

        # store original size BEFORE resizing, used to scale intrinsics
        orig_h, orig_w = img.shape[:2]
        new_h, new_w = image_size

        # resize if required (BILINEAR)
        if (orig_h, orig_w) != (new_h, new_w):
            img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

        images.append(img)

        # parse projection matrix P
        P = np.loadtxt(cam_file)
        if P.shape != (3, 4):
            raise ValueError(f"Unexpected camera shape {P.shape} in {cam_file}")

        P_list.append(P)
        ids.append(base_name)  # keep string id; camera_from_decomposed will try int()
        # image scale (sx, sy) that maps original -> requested
        s_x = float(new_w) / float(orig_w)
        s_y = float(new_h) / float(orig_h)
        image_scales.append((s_x, s_y))

    # Decompose + globally normalize intrinsics & camera centers
    if len(P_list) == 0:
        return images, []

    cameras, meta = normalize_camera_projection_list(P_list, camera_ids=ids, image_scales=image_scales)
    # meta contains 'centroid' and 'scale' in case caller needs to transform scene geometry

    return images, cameras
