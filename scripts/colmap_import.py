# colmap_to_layers.py
import os
import numpy as np
import keras
from scipy.spatial.transform import Rotation as R
import random

from autosplat.core import Camera
from autosplat.core import Gaussian3D


def read_cameras_txt(path):
    cams = {}
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            cam_id = int(parts[0])
            model = parts[1]
            width, height = float(parts[2]), float(parts[3])
            params = list(map(float, parts[4:]))
            cams[cam_id] = {"model": model, "width": width, "height": height, "params": params}
    return cams


def parse_images_txt(path):
    """
    Parse images.txt and return list of entries:
    [{'image_id':int, 'qvec':(4,), 'tvec':(3,), 'cam_id':int, 'name':str}, ...]
    """
    entries = []
    with open(path, "r") as f:
        lines = [l.strip() for l in f if l.strip() and not l.startswith("#")]
    # images.txt uses two lines per image; the first contains the pose & filename
    image_lines = lines[::2]
    for ln in image_lines:
        parts = ln.split()
        image_id = int(parts[0])
        qvec = np.array(list(map(float, parts[1:5])))   # qw qx qy qz
        tvec = np.array(list(map(float, parts[5:8])))   # tx ty tz
        cam_id = int(parts[8])
        name = parts[9]
        entries.append({"image_id": image_id, "qvec": qvec, "tvec": tvec, "cam_id": cam_id, "name": name})
    return entries


def parse_points3d_txt(path, max_points=None):
    """
    Parse points3D.txt and yield (xyz, rgb) tuples.
    """
    pts = []
    with open(path, "r") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            xyz = np.array(list(map(float, parts[1:4])))
            rgb = np.array(list(map(int, parts[4:7])), dtype=np.float32) / 255.0
            pts.append((xyz, rgb))
            if max_points is not None and len(pts) >= max_points:
                break
    return pts


def colmap_qt_to_cam_cw(qvec, tvec):
    """
    Convert COLMAP qvec (qw,qx,qy,qz) and tvec (tx,ty,tz) into
    camera-to-world rotation matrix R_cw and translation vector t_cw.

    COLMAP: x_cam = R_wc * x_world + t_wc  (world->camera)
    We return camera-to-world:
        R_cw = R_wc^T
        t_cw = -R_wc^T @ t_wc
    """
    # SciPy uses quaternion order [x, y, z, w]
    qw, qx, qy, qz = qvec[0], qvec[1], qvec[2], qvec[3]
    rot = R.from_quat([qx, qy, qz, qw])
    R_wc = rot.as_matrix()        # world-to-camera
    t_wc = np.array(tvec).reshape(3, 1)

    R_cw = R_wc.T
    t_cw = (-R_wc.T @ t_wc).reshape(3,)
    return R_cw.astype(np.float32), t_cw.astype(np.float32)


def build_layers_from_colmap(
    colmap_sparse_dir,
    images_dir=None,
    max_cameras=None,
    max_points=None,
    verbose=True,
):
    """
    Load cameras/images/points from a COLMAP sparse/text export and create
    Camera and Gaussian3D Keras layers.

    Parameters
    ----------
    colmap_sparse_dir : str
        Folder containing cameras.txt, images.txt, points3D.txt
    images_dir : str or None
        Optional path where the actual image files live (used to verify existence).
    max_cameras, max_points : int or None
        Limits for import amounts.
    """
    cams_txt = os.path.join(colmap_sparse_dir, "cameras.txt")
    imgs_txt = os.path.join(colmap_sparse_dir, "images.txt")
    pts_txt = os.path.join(colmap_sparse_dir, "points3D.txt")

    cams = read_cameras_txt(cams_txt)
    img_entries = parse_images_txt(imgs_txt)
    pts = parse_points3d_txt(pts_txt, max_points=max_points)

    camera_layers = []
    image_names = []

    if max_cameras is not None:
        img_entries = img_entries[:max_cameras]

    for entry in img_entries:
        img_id = entry["image_id"]
        qvec = entry["qvec"]
        tvec = entry["tvec"]
        cam_id = entry["cam_id"]
        name = entry["name"]

        # Convert COLMAP -> camera-to-world
        R_cw, t_cw = colmap_qt_to_cam_cw(qvec, tvec)

        # Attempt to extract focal length + principal point from cameras.txt (if available)
        caminfo = cams.get(cam_id, None)
        if caminfo:
            params = caminfo["params"]
            # Common formats: [f, cx, cy, ...] or [fx, fy, cx, cy, ...]
            if len(params) >= 3:
                fx = float(params[0])
                if len(params) >= 4:
                    fy = float(params[1]) if len(params) >= 4 else fx
                    cx = float(params[-2])
                    cy = float(params[-1])
                else:
                    fy = fx
                    cx = float(params[1]) if len(params) >= 2 else caminfo["width"] / 2.0
                    cy = float(params[2]) if len(params) >= 3 else caminfo["height"] / 2.0
            else:
                fx = caminfo["width"] / 2.0
                cx, cy = caminfo["width"] / 2.0, caminfo["height"] / 2.0
        else:
            fx, cx, cy = 1.0, 0.0, 0.0

        # Instantiate Camera layer **but** we won't rely on Euler angles.
        cam_layer = Camera(
            camera_id=img_id,
            location=(0.0, 0.0, 0.0),           # temporarily set, will overwrite
            rotation_angles=(0.0, 0.0, 0.0),   # temporarily set, will overwrite
            focus=float(fx),
            c=(float(cx), float(cy)),
            train_focus=False,  # keep as you prefer
            train_c=False
        )

        # Overwrite the rotation matrix & location with exact values from COLMAP conversion
        # Convert to keras tensors (safe ways)
        try:
            cam_layer.rotation_matrix = keras.ops.convert_to_tensor(R_cw, dtype="float32")
            cam_layer.location = keras.ops.convert_to_tensor(t_cw, dtype="float32")
        except Exception:
            # Fallback: assign as numpy arrays (still accessible)
            cam_layer.rotation_matrix = R_cw
            cam_layer.location = t_cw

        # Attach image filename & optional full path
        cam_layer.image_name = name
        if images_dir is not None:
            img_path = os.path.join(images_dir, name)
            cam_layer.image_path = img_path if os.path.exists(img_path) else None
        else:
            cam_layer.image_path = None

        camera_layers.append(cam_layer)
        image_names.append(name)

        if verbose:
            print(f"Loaded camera {img_id} -> {name}  fx={fx:.2f}  R_cw shape={R_cw.shape}")

    # Build Gaussians
    gaussian_layers = []
    for i, (xyz, rgb) in enumerate(pts):
        g = Gaussian3D(max_s_value = 0.05, min_s_value = 0.01)
        # assign trainable weights directly (mu, rgb). This bypasses add_weight re-creation.
        try:
            g.mu.assign(xyz.astype(np.float32))
            g.rgb.assign(rgb.astype(np.float32))
        except Exception:
            # fallback if assign not supported, set attribute (still usable but not tracked)
            g.mu = keras.ops.convert_to_tensor(xyz.astype(np.float32))
            g.rgb = keras.ops.convert_to_tensor(rgb.astype(np.float32))
        gaussian_layers.append(g)
        if verbose and (i < 5):
            print(f"  Gaussian {i}: mu={xyz}, rgb={rgb}")

    if verbose:
        print(f"Imported {len(camera_layers)} cameras, {len(gaussian_layers)} Gaussians.")

    return camera_layers, image_names, gaussian_layers

def compute_scene_normalization(gaussians, cameras):
    """
    Compute a normalization transform so that all 3D points and camera locations
    fit roughly within [-1, 1] cube.

    Parameters
    ----------
    gaussians : list of Gaussian3D
    cameras : list of Camera

    Returns
    -------
    center : np.ndarray, shape (3,)
        Mean center of the scene.
    scale : float
        Scaling factor such that the longest side fits to [-1, 1].
    """
    all_points = []

    # Collect Gaussian centers (mu)
    for g in gaussians:
        if hasattr(g.mu, "numpy"):
            all_points.append(g.mu.numpy())
        else:
            all_points.append(np.array(g.mu))

    # Collect camera locations
    for c in cameras:
        loc = c.location.numpy() if hasattr(c.location, "numpy") else np.array(c.location)
        all_points.append(loc)

    all_points = np.array(all_points)
    center = np.mean(all_points, axis=0)
    max_extent = np.max(np.linalg.norm(all_points - center, axis=1))
    scale = 1.0 / max_extent  # scale to roughly fit in unit ball

    return center, scale


def normalize_scene(gaussians, cameras, center, scale):
    """
    Apply normalization (translate + scale) to all Gaussians and Cameras.

    Modifies in place.

    Parameters
    ----------
    center : (3,)
        Scene centroid to subtract.
    scale : float
        Scaling factor.
    """
    for g in gaussians:
        if hasattr(g.mu, "assign"):
            g.mu.assign((g.mu - center) * scale)
        else:
            g.mu = (np.array(g.mu) - center) * scale

    for c in cameras:
        if hasattr(c.location, "assign"):
            c.location.assign((c.location - center) * scale)
        else:
            c.location = (np.array(c.location) - center) * scale


import os
import numpy as np
from PIL import Image

def load_images_from_list(dir_path, image_names, resize=None, rotate_90=False):
    """
    Load a list of images from a directory in the given order and return as a NumPy tensor.

    Parameters:
    -----------
    dir_path : str
        Path to the directory containing the images.
    image_names : list of str
        List of image filenames to load (in order).
    resize : tuple(int, int) or None
        Optional (height, width) to resize images to. If None, keep original size.
    rotate_90 : bool
        If True, rotate each image 90 degrees clockwise.

    Returns:
    --------
    np.ndarray
        Numpy array of shape (N, H, W, C) with float32 normalized pixel values [0, 1].
    """
    images = []

    for name in image_names:
        path = os.path.join(dir_path, name)
        if not os.path.exists(path):
            print(f"[WARN] Image not found: {path}")
            continue

        try:
            img = Image.open(path).convert("RGB")

            # Rotate 90 degrees clockwise if requested
            if rotate_90:
                img = img.transpose(Image.ROTATE_270)  # 270° CCW == 90° CW

            # Resize if requested
            if resize is not None:
                img = img.resize((resize[1], resize[0]), Image.BICUBIC)

            # Convert to normalized float32 NumPy array
            img_np = np.asarray(img, dtype=np.float32) / 255.0
            images.append(img_np)

        except Exception as e:
            print(f"[ERROR] Could not load {path}: {e}")
            continue

    if not images:
        raise ValueError("No images were loaded. Check file paths or formats.")

    return np.stack(images, axis=0) / 255.0

