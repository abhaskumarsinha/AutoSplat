import os
import numpy as np
import imageio.v3 as iio
from autosplat.core.camera import Camera  # Your Camera class

def load_dataset(folder, image_size=(256, 256), skip_seeds=True):
    """
    Load images and cameras from dataset folder.

    Args:
        folder (str): Path to dataset folder containing *_c.png and *_P.txt files.
        image_size (tuple): Desired output size (H, W) for images.
        skip_seeds (bool): Whether to ignore _seeds.bin files.

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
        base_name = img_file.split('_c.png')[0]
        cam_file = os.path.join(folder, f"{base_name}_P.txt")
        img_path = os.path.join(folder, img_file)

        # Load and resize image
        img = iio.imread(img_path)
        if img.shape[:2] != image_size:
            # Resize using simple nearest-neighbor / PIL if you like
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
        # Here assuming t[2] stores approximate focus / scale
        focus = np.linalg.norm(R[0])  # crude approximation, refine if dataset provides fx, fy
        c = (0.0, 0.0)  # placeholder, if principal point not in file

        cam = Camera(
            camera_id=int(base_name),
            location=t,
            rotation_angles=euler_angles,
            focus=focus,
            c=c
        )
        cameras.append(cam)

    return images, cameras
