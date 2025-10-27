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
import re


def normalize_camera(
    cameras,
    image_size=(256, 256),
    world_box_min=(-1.0, -1.0, -1.0),
    world_box_max=(1.0, 1.0, 1.0),
    margin=0.05,
    eps=1e-6,
    verbose=False
):
    """
    Adjust camera intrinsics (focus, principal point, jacobian) so that the unit
    world box defined by world_box_min..world_box_max fits inside the image
    frustum for each camera.

    Modifies Camera objects in-place and returns meta {'image_size', 'margin'}.

    Logic:
      - compute the 8 corners of the world box
      - for each camera:
          - transform corners into camera coordinates: X_cam = R @ (X_world - cam.location)
          - compute ratios rx = |x_cam / z_cam|, ry = |y_cam / z_cam|
            (if z_cam <= eps it's clamped to eps to avoid div-by-zero)
          - find max_rx, max_ry across corners
          - set fx = (W/2) / (max_rx * (1+margin))  (so unit-box fills image with margin)
            similarly fy = (H/2) / (max_ry * (1+margin))
          - set cam.focus = (fx + fy) / 2
          - set cam.c = (W/2, H/2)  (image center)
          - update cam.jacobian accordingly and recompute dependent fields

    Notes:
      - If all z_cam for corners are <= 0 (camera inside box looking away), we
        clamp z to eps to avoid massive focals; result may be unnatural for that camera.
      - margin is fractional extra space (0.05 = 5% extra border).
    """
    W, H = image_size[1], image_size[0]  # image_size is (H, W)
    # prepare box corners
    mins = np.asarray(world_box_min, dtype=float)
    maxs = np.asarray(world_box_max, dtype=float)
    corners = []
    for xi in [mins[0], maxs[0]]:
        for yi in [mins[1], maxs[1]]:
            for zi in [mins[2], maxs[2]]:
                corners.append(np.array([xi, yi, zi], dtype=float))
    corners = np.stack(corners, axis=0)  # (8,3)

    for cam in cameras:
        Rmat = cam.rotation_matrix  # 3x3, should map world->camera: X_cam = R @ (X_world - C)
        C_world = cam.location       # camera center in world coords

        # transform corners into camera coords
        rel = corners - C_world[None, :]         # (8,3)
        corners_cam = (Rmat @ rel.T).T           # (8,3)

        xs = corners_cam[:, 0]
        ys = corners_cam[:, 1]
        zs = corners_cam[:, 2]

        # clamp z to avoid division by zero and handle behind-camera points
        zs_clamped = np.where(zs > eps, zs, eps)

        # ratios
        rx = np.abs(xs / zs_clamped)
        ry = np.abs(ys / zs_clamped)

        max_rx = np.max(rx)
        max_ry = np.max(ry)

        # If both are zero (camera extremely far or degenerate), fallback to small value:
        if max_rx <= 0:
            max_rx = eps
        if max_ry <= 0:
            max_ry = eps

        # compute focal lengths in pixel units so that fx * max_rx == W/2 (half width)
        fx = (W * 0.5) / (max_rx * (1.0 + margin))
        fy = (H * 0.5) / (max_ry * (1.0 + margin))

        # choose single focus scalar as average (keeps aspect)
        f_new = float((fx + fy) / 2.0)

        # set principal point to image center (you can preserve previous offset if you want)
        cx_new = float(W * 0.5)
        cy_new = float(H * 0.5)

        # write back to Camera
        cam.focus = f_new
        cam.c = np.array([cx_new, cy_new], dtype=float)

        # update jacobian format used elsewhere: [[f,0,cx],[0,f,cy],[0,0,1]]
        cam.jacobian = np.array([
            [cam.focus, 0.0, cam.c[0]],
            [0.0, cam.focus, cam.c[1]],
            [0.0, 0.0, 1.0]
        ], dtype=float)

        # ensure rotation_matrix is consistent (no change) and keep as-is
        # If Camera has methods that rely on jacobian etc, recompute those if needed:
        # If your Camera class had compute_jacobian() use that; but we set it directly.

        if verbose:
            print(f"Camera ID: {cam.camera_id}")
            print(f"  max_rx={max_rx:.6g}, max_ry={max_ry:.6g}")
            print(f"  set focus={cam.focus:.6g}, c=({cam.c[0]:.3f},{cam.c[1]:.3f})")
            print(f"  location={cam.location}")
            print()

    meta = {'image_size': image_size, 'margin': margin}
    return cameras, meta


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


def _extract_base_name_from_image(img_file: str) -> str:
    """
    Robustly extract the base name for camera file mapping from an image filename.
    Handles cases like:
      - 00001_c.png
      - 00001._c.png
      - 00001_c.JPG
      - weird.extra.00001._c.png
    Returns the base name without any trailing dots.
    """
    # look for the last occurrence of "_c" (case-insensitive)
    lower = img_file.lower()
    pos = lower.rfind('_c')
    if pos != -1:
        base = img_file[:pos]
    else:
        # fallback: remove the extension, then strip last underscore-group
        base = os.path.splitext(img_file)[0]
        # if there is an underscore, assume last underscore separates id from suffix
        if '_' in base:
            base = base.rsplit('_', 1)[0]

    # remove any stray trailing dots that created the "00001." situation
    base = base.rstrip('.')
    return base

def load_dataset(folder, image_size=(256, 256), skip_seeds=True, rotate_images=False):
    """
    Load images and normalized cameras from dataset folder.

    Returns:
        images (list[np.ndarray]), cameras (list[Camera])
    """
    images = []
    P_list = []
    ids = []
    image_scales = []

    files = sorted(os.listdir(folder))

    # find image files (endswith like '_c.<ext>' case-insensitive)
    img_files = [f for f in files if re.search(r'[_\.]c\.[A-Za-z0-9]+$', f)]
    if skip_seeds:
        img_files = [f for f in img_files if '_seeds' not in f.lower()]

    for img_file in img_files:
        base_name = _extract_base_name_from_image(img_file)
        cam_file = os.path.join(folder, f"{base_name}_P.txt")
        img_path = os.path.join(folder, img_file)

        if not os.path.exists(img_path):
            # safety: skip missing images (should not happen normally)
            continue

        # Load image
        img = iio.imread(img_path)
        if rotate_images:
            img = np.rot90(img, k=-1)

        orig_h, orig_w = img.shape[:2]
        new_h, new_w = image_size

        if (orig_h, orig_w) != (new_h, new_w):
            img = np.array(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))

        images.append(img)

        if not os.path.exists(cam_file):
            raise FileNotFoundError(f"Expected camera file {cam_file} for image {img_path}")

        P = np.loadtxt(cam_file)
        if P.shape != (3, 4):
            raise ValueError(f"Unexpected camera shape {P.shape} in {cam_file}")

        P_list.append(P)
        ids.append(base_name)
        s_x = float(new_w) / float(orig_w)
        s_y = float(new_h) / float(orig_h)
        image_scales.append((s_x, s_y))

    if len(P_list) == 0:
        return images, []

    cameras, meta = normalize_camera_projection_list(P_list, camera_ids=ids, image_scales=image_scales)
    return images, cameras

