import os
import re
import numpy as np
import imageio.v3 as iio
from scipy.linalg import svd
from scipy.spatial.transform import Rotation as R

from autosplat.core import Camera

def _read_par_file_safe(par_path):
    entries = []
    with open(par_path, 'r') as f:
        for lineno, ln in enumerate(f, start=1):
            s = ln.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 22:
                print(f"par: skipping short line {lineno}: {s!r}")
                continue
            imgname = parts[0]
            vals = list(map(float, parts[1:]))
            K = np.array(vals[0:9]).reshape(3, 3)
            Rmat = np.array(vals[9:18]).reshape(3, 3)
            t = np.array(vals[18:21], dtype=float)
            entries.append({'imgname': imgname, 'K': K, 'R': Rmat, 't': t})
    print(f"par: parsed {len(entries)} valid entries from {par_path}")
    return entries

def _orthonormalize_R(Rmat):
    U, s, Vt = svd(Rmat)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def import_temple_ring_to_cameras_fixed(
    dataset_dir,
    par_filename='templeR_par.txt',
    target_image_size=None,   # e.g. (480,640) or None to keep native
    normalize_world=True,
    rotate_ccw: bool = False,   # <-- Counter-clockwise rotation
    train_focus = True,
    train_c = True,
    verbose=True
):
    """
    Import Temple ring dataset into Camera class list.

    rotate_ccw: if True, each image is rotated 90° counter-clockwise,
                and intrinsics are updated accordingly.
    """
    par_path = os.path.join(dataset_dir, par_filename)
    if not os.path.exists(par_path):
        raise FileNotFoundError(par_path)

    entries = _read_par_file_safe(par_path)

    images = []
    cameras = []
    centers = []
    image_scales = []
    orig_h = orig_w = None
    new_h = new_w = None

    for idx, ent in enumerate(entries):
        imgname = ent['imgname']
        img_path = os.path.join(dataset_dir, imgname)
        if not os.path.exists(img_path):
            if verbose:
                print(f"Warning: image {img_path} missing — skipping")
            continue

        img = iio.imread(img_path)
        if img.ndim == 2:
            img = np.stack([img]*3, axis=-1)
        orig_h, orig_w = img.shape[:2]

        # --- resize if requested ---
        if target_image_size is not None:
            new_h, new_w = target_image_size
            if (orig_h, orig_w) != (new_h, new_w):
                from PIL import Image
                img = np.asarray(Image.fromarray(img).resize((new_w, new_h), Image.BILINEAR))
                s_x = new_w / float(orig_w)
                s_y = new_h / float(orig_h)
            else:
                s_x = s_y = 1.0
        else:
            new_h, new_w = orig_h, orig_w
            s_x = s_y = 1.0

        # --- optional 90° counter-clockwise rotation ---
        if rotate_ccw:
            img = np.rot90(img, k=1)  # CCW rotation
            H_before, W_before = new_h, new_w
            new_h, new_w = img.shape[:2]

        images.append(img)
        image_scales.append((s_x, s_y))

        # --- intrinsics & extrinsics from par file ---
        K = ent['K'].astype(float).copy()
        Rfile = ent['R'].astype(float).copy()
        t = ent['t'].astype(float).copy()

        # normalize K
        if abs(K[2,2]) < 1e-12:
            raise RuntimeError("Invalid K[2,2] == 0")
        K = K / K[2,2]

        R_ortho = _orthonormalize_R(Rfile)
        if np.linalg.det(R_ortho) < 0:
            R_ortho = -R_ortho
            K = -K

        C = -R_ortho.T @ t
        centers.append(C)

        R_cam2world = R_ortho.T
        euler = R.from_matrix(R_cam2world).as_euler('xyz', degrees=False)

        # apply scaling
        K_scaled = K.copy()
        K_scaled[0,0] *= s_x
        K_scaled[1,1] *= s_y
        K_scaled[0,2] *= s_x
        K_scaled[1,2] *= s_y

        # --- if rotated CCW, update K ---
        if rotate_ccw:
            # For 90° CCW rotation:
            # u' = v
            # v' = -u + (W_before - 1)
            M = np.array([[0.0,  1.0, 0.0],
                          [-1.0, 0.0, float(W_before - 1)],
                          [0.0,  0.0, 1.0]], dtype=float)
            K_scaled = M @ K_scaled

        fx = float(K_scaled[0,0])
        fy = float(K_scaled[1,1])
        f_mean = (fx + fy) / 2.0
        cx = float(K_scaled[0,2])
        cy = float(K_scaled[1,2])

        m = re.search(r'(\d+)', imgname)
        cam_id = int(m.group(1)) if m else idx

        cam = Camera(
            camera_id=cam_id,
            location=tuple(C.tolist()),
            rotation_angles=tuple(euler.tolist()),
            focus=f_mean,
            c=(cx, cy),
            train_focus = train_focus,
            train_c = train_c
        )
        cam._K_scaled = K_scaled
        cameras.append(cam)

    if len(cameras) == 0:
        return images, [], {'orig_image_size': None}

    # --- normalize world (scale only camera centers) ---
    centers = np.stack(centers, axis=0)
    centroid = centers.mean(axis=0)
    max_abs = np.max(np.abs(centers - centroid))
    scale_world = 1.0 / max_abs if max_abs > 0 else 1.0

    if normalize_world:
        for cam in cameras:
            cam.location = (cam.location.detach().cpu().numpy() - centroid) * scale_world
            K_s = cam._K_scaled
            fx = float(K_s[0,0])
            fy = float(K_s[1,1])
            cam.focus = (fx + fy) / 2.0
            cam.c = np.array([float(K_s[0,2]), float(K_s[1,2])])
            cam.jacobian = cam.compute_jacobian()
            cam.rotation_matrix = cam.compute_rotation_matrix()
            del cam._K_scaled
    else:
        for cam in cameras:
            K_s = cam._K_scaled
            fx = float(K_s[0,0])
            fy = float(K_s[1,1])
            cam.focus = (fx + fy) / 2.0
            cam.c = np.array([float(K_s[0,2]), float(K_s[1,2])])
            cam.jacobian = cam.compute_jacobian()
            cam.rotation_matrix = cam.compute_rotation_matrix()
            del cam._K_scaled

    meta = {
        'num_views': len(cameras),
        'orig_image_size': (orig_h, orig_w),
        'target_image_size': (new_h, new_w),
        'centroid': centroid,
        'scale_world': scale_world,
        'rotated_images': bool(rotate_ccw)
    }

    if verbose:
        print(f"Loaded {len(cameras)} cameras and {len(images)} images.")
        print(f"World centroid = {centroid}, world scale factor applied = {scale_world:.6g}")
        if rotate_ccw:
            print("Images were rotated 90° counter-clockwise; intrinsics updated accordingly.")

    return images, cameras, meta



def shift_world_centroid_to_zero(cameras):
    """
    Translate camera locations so the centroid of all camera centers becomes exactly (0,0,0).
    This function does NOT change intrinsics (focus, c) or rotation.
    """
    if len(cameras) == 0:
        return cameras, np.zeros(3)

    centers = np.stack([cam.location for cam in cameras], axis=0)  # (N,3)
    centroid = centers.mean(axis=0)
    for cam in cameras:
        cam.location = np.array(cam.location) - centroid
    return cameras, centroid

