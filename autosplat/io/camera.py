import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from autosplat.core.camera import Camera

def save_cameras_to_json(cameras, file_path):
    """
    Save a list of Camera objects to a JSON file.
    By default, saves rotation as quaternions for robustness.

    Parameters
    ----------
    cameras : list of Camera
        Cameras to save.
    file_path : str
        Output JSON path.
    """
    cam_data = []
    for cam in cameras:
        # Convert rotation matrix to quaternion
        quat = R.from_matrix(cam.rotation_matrix).as_quat()  # [x, y, z, w]
        cam_entry = {
            "camera_id": cam.camera_id,
            "location": cam.location.tolist(),
            "rotation_quaternion": quat.tolist(),
            "focus": cam.focus,
            "c": cam.c.tolist()
        }
        cam_data.append(cam_entry)

    with open(file_path, "w") as f:
        json.dump(cam_data, f, indent=4)
    print(f"Saved {len(cameras)} cameras to {file_path}")

def load_cameras_from_json(file_path):
    """
    Load a list of Camera objects from a JSON file.
    Supports JSON containing either Euler angles or quaternions.

    Parameters
    ----------
    file_path : str
        Path to JSON file.

    Returns
    -------
    cameras : list of Camera
    """
    cameras = []
    with open(file_path, "r") as f:
        cam_data = json.load(f)

    for entry in cam_data:
        cam_id = entry["camera_id"]
        loc = entry.get("location", (0, 0, 0))
        focus = entry.get("focus", 1.0)
        c = entry.get("c", (0, 0))

        if "rotation_quaternion" in entry:
            quat = np.array(entry["rotation_quaternion"], dtype=float)
            # Convert quaternion to Euler angles
            rot_matrix = R.from_quat(quat).as_matrix()
            euler = R.from_matrix(rot_matrix).as_euler("xyz", degrees=False)
        elif "rotation_angles" in entry:
            euler = np.array(entry["rotation_angles"], dtype=float)
        else:
            euler = (0.0, 0.0, 0.0)

        cam = Camera(
            camera_id=cam_id,
            location=loc,
            rotation_angles=euler,
            focus=focus,
            c=c
        )
        cameras.append(cam)
    return cameras
