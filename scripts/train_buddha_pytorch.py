import sys
import os

# Add AutoSplat/ to Python module search path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# This script can best be run with torch backend.
os.environ["KERAS_BACKEND"] = "torch"

import keras
import matplotlib.pyplot as plt

from autosplat.core import Camera
from autosplat.core import Gaussian3D
from autosplat.core import BlenderLayer

from autosplat.utils import RenderObject
from autosplat.visualizer import plot_cameras_plotly

from data.dataloader import load_dataset, normalize_camera_positions

import os
from typing import Tuple, Optional
import torch
import argparse


def train_gaussian_renderer(
    dataset_dir,
    dataset_image_size=(64, 64),
    rotate_image=True,
    normalize_camera_extrinsics=True,
    gaussian_3d_eps=1e-6,
    camera_matrices_trainable=False,
    total_gaussians=1024,
    max_gaussians=128,
    mu_initializer='uniform',
    background_color=(-1.0, -1.0, -1.0),
    render_size=64,  # square only
    render_rotation_gif=True,
    blend_id=0,
    radius=2.0,
    steps=360,
    savepath="./checkpoints/",
    fps=15,
    loss_weighting_ratio=0.0,
    lr=1e-2,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=0,
    coupled_weight_decay=False,
    checkpoint_frequency=50,
    load_from_previous_checkpoint=True,
    torch_compile_mode="default",
    save_training_progress_examples=False,
    total_epochs=100
):
    """
    Main training function for Gaussian-based renderer.

    Parameters
    ----------
    dataset_dir : str
        Path to dataset directory.

    dataset_image_size : tuple(int, int), default=(64, 64)
        Image size to which dataset images are resized.

    rotate_image : bool, default=True
        Whether to rotate images when loading.

    normalize_camera_extrinsics : bool, default=True
        Normalize camera extrinsics during preprocessing.

    gaussian_3d_eps : float, default=1e-6
        Small epsilon value for numerical stability in Gaussian operations.

    camera_matrices_trainable : bool, default=False
        If True, camera extrinsic matrices are optimized during training.

    total_gaussians : int, default=1024
        Total number of Gaussians in the scene.

    max_gaussians : int, default=128
        Maximum number of Gaussians rendered at once.
        Must satisfy total_gaussians > max_gaussians.

    mu_initializer : str, default='uniform'
        Initializer for Gaussian mean ('uniform' or 'normal').

    background_color : tuple(float, float, float), default=(-1.0, -1.0, -1.0)
        Background RGB color used for rendering.

    render_size : int, default=64
        Render output size (square images).

    render_rotation_gif : bool, default=True
        Whether to generate a rotating camera GIF of the scene.

    blend_id : int, default=0
        ID for selecting blend configuration.

    radius : float, default=2.0
        Radius of the camera path used for GIF rotation.

    steps : int, default=360
        Number of rotation steps for generating GIF.

    savepath : str, default="./checkpoints/"
        Path to save checkpoints and outputs.

    fps : int, default=15
        Frames per second for the rendered GIF.

    loss_weighting_ratio : float, default=0.0
        Weighting ratio for auxiliary loss terms.

    lr : float, default=1e-2
        Learning rate for the optimizer.

    betas : tuple(float, float), default=(0.9, 0.999)
        Adam optimizer beta coefficients.

    eps : float, default=1e-8
        Adam optimizer epsilon parameter.

    weight_decay : float, default=0
        Weight decay regularization term.

    coupled_weight_decay : bool, default=False
        Whether to use decoupled weight decay (Adam-style).

    checkpoint_frequency : int, default=50
        Save model checkpoint every n epochs.

    load_from_previous_checkpoint : bool, default=True
        Whether to resume from an existing checkpoint if available.

    torch_compile_mode : str, default='default'
        Torch compile mode (e.g., 'default', 'reduce-overhead', 'max-autotune').

    save_training_progress_examples : bool, default=False
        Whether to save periodic example renderings during training.

    total_epochs : int, default=100
        Total number of epochs for training.
    """

    # === Setup ===
    os.makedirs(savepath, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if loss_weighting_ratio > 0.0:
        try:
            from keras_hub.src.models.depth_anything.depth_anything_depth_estimator import (
                DepthAnythingDepthEstimator,
            )
            depth_estimator = DepthAnythingDepthEstimator.from_preset("depth_anything_v2_small", depth_estimation_type="relative", max_depth=None,)
            depth_estimator.to(device)
        except ImportError:
            raise ImportError(
                "This module requires the unstable Keras Hub. "
                "Install it via: pip install git+https://github.com/keras-team/keras-hub.git"
            )

    images, cameras = load_dataset(dataset_dir, image_size=dataset_image_size, rotate_images=rotate_image_on_load)
    images = np.array(images) / 255.0
    if normalize_camera_extrinsics:
        cameras = normalize_camera_positions(cameras)

    gaussians = []
    for _ in range(total_gaussians):
        gaussians.append(Gaussian3D(eps=gaussian_eps, use_default_projection=True, mu_initializer=mu_initializer))

    scenes = []
    for cam in cameras:
        scenes.append(BlendLayer(cam, gaussians, camera_trainable=camera_matrices_trainable, max_gaussians=max_gaussians, background_color=background_color))

    renderer = RenderObject(blend_layers=scenes, image_size=dataset_image_size).to(device)

    optimizer = torch.optim.Adam(renderer.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, coupled_weight_decay=coupled_weight_decay)
    if torch_compile_mode is not None:
        renderer = torch.compile(renderer, mode=torch_compile_mode)
    optimizer = torch.optim.Adam(renderer.parameters(), lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, coupled_weight_decay=decoupled_weight_decay)
    I_gt = torch.tensor(images, dtype=torch.float32, device=device)

    blend_id_indices = [int(i) for i in range(len(scenes))]

    # --------- WARNING-----------
    # Check if that loads up in to(device)
    # ----------------------------
    if load_from_previous_checkpoint:
        try:
            main(input_folder=savepath, output_folder=None, save_cameras=False, save_gaussians=False, overwrite=False)
        except:
            pass

    def find_depth(images):
        image = keras.ops.image.resize(keras.ops.convert_to_tensor(images), (518, 518))
        depth = depth_estimator.predict(image, device=device)['depths']
        depth = keras.ops.nn.relu(depth)
        depth = (depth - keras.ops.min(depth)) / (
            keras.ops.max(depth) - keras.ops.min(depth)
        )
        return depth

    if loss_weighting_ratio > 0.0:
        y_depths = find_depth(images)
        

    for step in tqdm(range(total_epochs)):
        optimizer.zero_grad()
        I_pred = renderer(blend_id=torch.tensor(blend_id_indices, device=device))        
        if loss_weighting_ratio > 0.0:
            I_pred_depth = find_depth(I_pred)
            loss_render = torch.mean((I_pred - I_gt) ** 2)
            loss_depth = torch.mean((I_pred_depth - y_depths) ** 2)
            loss = (1-loss_weighting_ratio)*loss_render + loss_weighting_ratio*loss_depth
        else:
            loss = torch.mean((I_pred - I_gt) ** 2)
        loss.backward()
        optimizer.step()
        log(f"Step {step:03d}: loss={loss.item():.6f} ")

        if step % 10 == 0:
            for layer in renderer.blend_layers:
                layer.sort_gaussians_by_distance()


        if step % checkpoint_frequency == 0 or step == total_epochs - 1:
            if render_rotation_gif:
                rotation_gif_path = os.path.join(savepath, "rotation.gif")
                render_rotation_gif(renderer=renderer, blend_id=0, res=render_size, radius=radius, steps=steps, save_path=rotation_gif_path, fps=fps)
            if savepath is not None:
                main(input_folder=None, output_folder=savepath, save_cameras=True, save_gaussians=True, overwrite=False)
            if save_training_progress_examples:
                filename = "preview.png"
                base, ext = os.path.splitext(filename)
                save_file = os.path.join(savepath, filename)
                counter = 1
                while os.path.exists(save_file):
                    save_file = os.path.join(savepath, f"{base}_{counter}{ext}")
                    counter += 1
                plt.imsave(save_file, renderer.preview(0))




def get_args():
    parser = argparse.ArgumentParser(description="Train Gaussian-based Renderer")

    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory.')
    parser.add_argument('--dataset_image_size', type=int, nargs=2, default=(64, 64), help='Image size (H W) for dataset images.')
    parser.add_argument('--rotate_image', type=bool, default=True, help='Whether to rotate images when loading.')
    parser.add_argument('--normalize_camera_extrinsics', type=bool, default=True, help='Normalize camera extrinsics during preprocessing.')
    parser.add_argument('--gaussian_3d_eps', type=float, default=1e-6, help='Small epsilon for numerical stability in Gaussian operations.')
    parser.add_argument('--camera_matrices_trainable', type=bool, default=False, help='Optimize camera extrinsics during training.')
    parser.add_argument('--total_gaussians', type=int, default=1024, help='Total number of Gaussians in the scene.')
    parser.add_argument('--max_gaussians', type=int, default=128, help='Maximum number of Gaussians rendered at once.')
    parser.add_argument('--mu_initializer', type=str, choices=['uniform', 'normal'], default='uniform', help='Initializer for Gaussian mean.')
    parser.add_argument('--background_color', type=float, nargs=3, default=(-1.0, -1.0, -1.0), help='Background RGB color.')
    parser.add_argument('--render_size', type=int, default=64, help='Render output size (square).')
    parser.add_argument('--render_rotation_gif', type=bool, default=True, help='Whether to generate rotating camera GIF.')
    parser.add_argument('--blend_id', type=int, default=0, help='ID for selecting blend configuration.')
    parser.add_argument('--radius', type=float, default=2.0, help='Radius of camera path for GIF rotation.')
    parser.add_argument('--steps', type=int, default=360, help='Number of rotation steps for GIF.')
    parser.add_argument('--savepath', type=str, default='./checkpoints/', help='Path to save checkpoints and outputs.')
    parser.add_argument('--fps', type=int, default=15, help='Frames per second for GIF.')
    parser.add_argument('--loss_weighting_ratio', type=float, default=0.0, help='Weighting ratio for auxiliary loss terms.')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate.')
    parser.add_argument('--betas', type=float, nargs=2, default=(0.9, 0.999), help='Adam optimizer beta coefficients.')
    parser.add_argument('--eps', type=float, default=1e-8, help='Adam optimizer epsilon.')
    parser.add_argument('--weight_decay', type=float, default=0, help='Weight decay regularization.')
    parser.add_argument('--coupled_weight_decay', type=bool, default=False, help='Use decoupled weight decay (Adam-style).')
    parser.add_argument('--checkpoint_frequency', type=int, default=50, help='Save checkpoint every n epochs.')
    parser.add_argument('--load_from_previous_checkpoint', type=bool, default=True, help='Resume from existing checkpoint.')
    parser.add_argument('--torch_compile_mode', type=str, default='default', help="Torch compile mode ('default', 'reduce-overhead', 'max-autotune').")
    parser.add_argument('--save_training_progress_examples', type=bool, default=False, help='Save example renderings during training.')
    parser.add_argument('--total_epochs', type=int, default=100, help='Total number of epochs.')

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    train_gaussian_renderer(**vars(args))
