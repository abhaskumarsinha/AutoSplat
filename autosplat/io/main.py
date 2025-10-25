import os
import glob
import shutil

def main(
    input_folder,
    output_folder,
    save_cameras=True,
    save_gaussians=True,
    overwrite=False
):
    """
    Load cameras and Gaussians from a folder, then save them to another folder.
    Automatically handles folder creation and avoids overwriting by appending suffixes.

    Parameters
    ----------
    input_folder : str
        Folder containing JSON files for cameras and Gaussians.
    output_folder : str
        Folder to save processed JSON files.
    save_cameras : bool, optional
        Whether to save cameras after loading. Default True.
    save_gaussians : bool, optional
        Whether to save Gaussians after loading. Default True.
    overwrite : bool, optional
        Whether to overwrite existing files. If False, will append suffix to filename.
    """
    os.makedirs(output_folder, exist_ok=True)

    # --- Cameras ---
    cam_files = glob.glob(os.path.join(input_folder, "*camera*.json"))
    for cam_file in cam_files:
        cameras = load_cameras_from_json(cam_file)

        if save_cameras:
            base_name = os.path.basename(cam_file)
            save_path = os.path.join(output_folder, base_name)

            # Handle existing file
            if os.path.exists(save_path) and not overwrite:
                name, ext = os.path.splitext(base_name)
                i = 1
                while os.path.exists(os.path.join(output_folder, f"{name}_{i}{ext}")):
                    i += 1
                save_path = os.path.join(output_folder, f"{name}_{i}{ext}")

            save_cameras_to_json(cameras, save_path)
            print(f"✅ Saved cameras to {save_path}")

    # --- Gaussians ---
    gaussian_files = glob.glob(os.path.join(input_folder, "*gaussian*.json"))
    for gaussian_file in gaussian_files:
        gaussians = load_gaussians_from_json(gaussian_file)

        if save_gaussians:
            base_name = os.path.basename(gaussian_file)
            save_path = os.path.join(output_folder, base_name)

            # Handle existing file
            if os.path.exists(save_path) and not overwrite:
                name, ext = os.path.splitext(base_name)
                i = 1
                while os.path.exists(os.path.join(output_folder, f"{name}_{i}{ext}")):
                    i += 1
                save_path = os.path.join(output_folder, f"{name}_{i}{ext}")

            save_gaussians_to_json(gaussians, save_path)
            print(f"✅ Saved Gaussians to {save_path}")

# Example usage
# main("/path/to/input_folder", "/path/to/output_folder", overwrite=False)
