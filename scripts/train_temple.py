import sys, os

# Get the absolute path to the project root (one level above "scripts")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os

# This guide can only be run with the torch backend.
os.environ["KERAS_BACKEND"] = "torch"


import os
import argparse
import torch
import keras
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt

# ----------- TEMP ----------------
from scripts.colmap_import import *
$ ---------------------------------


# ---------------------------
# Argument Parser
# ---------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="AutoSplat Trainer Script")

    parser.add_argument('--dataset_dir', type=str, required=True, help='Path to dataset directory.')
    parser.add_argument('--colmap_dir', type=str, required=True, help='Path to COLMAP Model export.')
    parser.add_argument('--rotate_ccw', type=bool, default=True, help='Rotate temple images counter-clockwise.')
    parser.add_argument('--total_gaussians', type=int, default=32)
    parser.add_argument('--eps_gaussian', type=float, default=1e-6)
    parser.add_argument('--mu_initializer', type=str, default='random_normal')
    parser.add_argument('--default_focus', type=float, default=1.0)
    parser.add_argument('--default_c', type=tuple, default=(0.0, 0.0))
    parser.add_argument('--camera_trainable', type=bool, default=True)
    parser.add_argument('--max_gaussians', type=int, default=32)
    parser.add_argument('--max_s_value', type=float, default=0.3)
    parser.add_argument('--min_s_value', type=float, default=0.05)
    parser.add_argument('--background_color', type=tuple, default=(-1.0, -1.0, -1.0))
    parser.add_argument('--image_size', type=tuple, default=(64, 64))
    parser.add_argument('--depth_loss_ratio', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--path_update_frequency', type=int, default=10)
    parser.add_argument('--blend_id_for_viz', type=int, default=0)
    parser.add_argument('--res_viz', type=int, default=256)
    parser.add_argument('--radius_viz', type=float, default=2.5)
    parser.add_argument('--steps_viz', type=int, default=180)
    parser.add_argument('--save_path_viz', type=str, default='rotation.gif')
    parser.add_argument('--fps_viz', type=int, default=10)
    parser.add_argument('--save_model_rendering', type=int, default=10)
    parser.add_argument('--save_model_freq', type=int, default=1)
    parser.add_argument('--json_camera_filepath', type=str, default='./camera.JSON')
    parser.add_argument('--json_gaussians_filepath', type=str, default='./gaussians.JSON')

    return parser.parse_args()


# ---------------------------
# Logger Setup
# ---------------------------
def setup_logger():
    logger = logging.getLogger("AutoSplat")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ---------------------------
# Main Trainer
# ---------------------------
def main():
    args = parse_args()
    logger = setup_logger()

    os.environ["KERAS_BACKEND"] = "torch"
    logger.info("✅ Keras backend set to Torch.")

    # Lazy imports
    from autosplat.core import Camera, Gaussian3D, BlenderLayer
    from autosplat.utils import RenderObject
    from autosplat.visualizer import render_rotation_gif
    from autosplat.io import save_cameras_to_json, save_gaussians_to_json
    from data.dataloader import import_temple_ring_to_cameras_fixed, shift_world_centroid_to_zero

    # ---------------------------
    # Load Dataset
    # ---------------------------
    logger.info(f"📂 Loading dataset from: {args.dataset_dir}")
    cams, names, gaussians = build_layers_from_colmap(args.colmap_dir, args.dataset_dir, max_cameras=50, max_points=5000)
    logger.info(f"📸 Loaded {len(cameras)} cameras, {len(gaussians)} gaussians and {len(names)} images.")

    images = load_images_from_list(args.dataset_dir, names, (64, 64), True)
    images = keras.ops.convert_to_tensor(images)


    center, scale = compute_scene_normalization(gaussians, cams)
    normalize_scene(gaussians, cams, center, scale)

    gaussians = random.sample(gaussians, 1000)

    for cam in cams:
        cam.focus = args.default_focus
        cam.c = args.default_c
        cam.jacobian = cam.compute_jacobian()
    logger.info("📷 Camera parameters (focus, c, jacobian) computed.")

    # ---------------------------
    # Scenes and Renderer
    # ---------------------------
    logger.info("🧩 Building scenes and renderer...")
    scenes = []
    for cam in cams:
        scenes.append(BlenderLayer(cam, gaussians, True, args.max_gaussians, (-1.0, -1.0, -1.0)))
    renderer = RenderObject(scenes, (64, 64))

    # ---------------------------
    # Device Setup
    # ---------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    renderer.to(device)
    logger.info(f"💻 Using device: {device}")

    # ---------------------------
    # Depth Estimator Setup
    # ---------------------------
    if args.depth_loss_ratio > 0.0:
        try:
            from keras_hub.src.models.depth_anything.depth_anything_depth_estimator import (
                DepthAnythingDepthEstimator,
            )
            logger.info("🧠 Initializing DepthAnythingDepthEstimator...")
            depth_estimator = DepthAnythingDepthEstimator.from_preset(
                "depth_anything_v2_small",
                depth_estimation_type="relative",
                max_depth=None,
            )
            depth_estimator.trainable = False
            depth_estimator.to(device)
            depth_estimator = torch.compile(depth_estimator)
            logger.info("✅ DepthAnythingDepthEstimator successfully loaded and compiled.")

            def find_depth(images):
                image = keras.ops.image.resize(keras.ops.convert_to_tensor(images), (518, 518))
                depth = depth_estimator.predict(image)['depths']
                depth = keras.ops.nn.relu(depth)
                depth = (depth - keras.ops.min(depth)) / (
                    keras.ops.max(depth) - keras.ops.min(depth)
                )
                return keras.ops.image.resize(depth, (64, 64))

            logger.info("📏 Estimating initial depth maps...")
            y_depths = find_depth(images)
            logger.info("✅ Depth maps computed successfully.")

        except ImportError as e:
            logger.error("❌ Failed to import DepthAnythingDepthEstimator.")
            raise ImportError(
                "You need the *unstable* version of keras-hub.\n"
                "Install it with:\n\n"
                "    !pip install git+https://github.com/keras-team/keras-hub.git\n\n"
                f"Original error: {e}"
            )
    else:
        logger.warning("⚠️ Depth loss disabled (depth_loss_ratio <= 0.0). Skipping depth model import.")

    # ---------------------------
    # Training Setup
    # ---------------------------
    logger.info("🔧 Setting up optimizer and ground truth tensors...")
    optimizer = torch.optim.Adam(renderer.parameters(), lr=1e-2)
    I_gt = torch.tensor(images, dtype=torch.float32, device=device) / 255.0
    I_gt = keras.ops.image.resize(I_gt, args.image_size)
    blend_ids = [i for i in range(len(scenes))]
    logger.info("✅ Training setup complete.")

    # ---------------------------
    # Training Loop
    # ---------------------------
    logger.info(f"🚀 Starting training for {args.epochs} epochs...")
    for step in tqdm(range(args.epochs), desc="Training"):
        if step % args.save_model_freq == 0:
            save_gaussians_to_json(g, args.json_gaussians_filepath)
            logger.info(f"💾 Saved Gaussians to {args.json_gaussians_filepath} at step {step}")

        optimizer.zero_grad()
        I_pred = renderer(blend_id=blend_ids)
        loss_pixels = torch.mean((I_pred - I_gt) ** 2)

        if args.depth_loss_ratio > 0.0:
            I_pred_depth = find_depth(I_pred)
            loss_depth = torch.mean((I_pred_depth - y_depths) ** 2)
            loss = args.depth_loss_ratio * loss_depth + (1.0 - args.depth_loss_ratio) * loss_pixels
        else:
            loss = loss_pixels

        loss.backward()
        optimizer.step()

        if step % args.path_update_frequency == 0 or step == args.epochs - 1:
            logger.info(f"Step {step:03d}: loss={loss.item():.6f}  (pixels={loss_pixels.item():.6f})")
            if args.depth_loss_ratio > 0.0:
                logger.info(f"           depth_loss={loss_depth.item():.6f}")
            for scene in renderer.blend_layers:
                scene.sort_gaussians_by_distance()
            renderer.to(device)

        if step % args.save_model_rendering == 0:
            render_rotation_gif(
                renderer,
                args.blend_id_for_viz,
                args.res_viz,
                args.radius_viz,
                args.steps_viz,
                args.save_path_viz,
                args.fps_viz,
            )
            logger.info(f"🎞️ Rendered rotation GIF saved to {args.save_path_viz}")

    logger.info("🏁 Training completed successfully.")


# ---------------------------
# Entry Point
# ---------------------------
if __name__ == "__main__":
    main()
