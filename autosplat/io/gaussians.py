import json
import numpy as np
import keras
from autosplat.core.gaussian import Gaussian3D

def load_gaussians_from_json(json_path):
    """
    Load Gaussians from a JSON file and return a list of Gaussian3D instances.

    Parameters
    ----------
    json_path : str
        Path to the JSON file containing Gaussians.

    Returns
    -------
    gaussians : list of Gaussian3D
        List of Gaussian3D objects initialized with JSON parameters.
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    gaussians = []

    for gdata in data.get("gaussians", []):
        g = Gaussian3D()

        # Load mu
        if "mu" in gdata:
            g.mu.assign(np.array(gdata["mu"], dtype=np.float32))

        # Load RGB
        if "rgb" in gdata:
            g.rgb.assign(np.array(gdata["rgb"], dtype=np.float32))

        # Load alpha
        if "alpha" in gdata:
            g.alpha.assign(np.array([gdata["alpha"]], dtype=np.float32))

        # Load scale / sigma
        if "scale" in gdata:
            g.s.assign(np.array(gdata["scale"], dtype=np.float32))

        # Load rotation quaternion (optional)
        if "rotation" in gdata:
            g.p.assign(np.array(gdata["rotation"], dtype=np.float32))

        gaussians.append(g)

    return gaussians


def save_gaussians_to_json(gaussians, json_path):
    """
    Save a list of Gaussian3D objects to a JSON file.

    Parameters
    ----------
    gaussians : list of Gaussian3D
        List of Gaussian3D objects to save.
    json_path : str
        Path to save the JSON file.
    """
    data = {"gaussians": []}

    for g in gaussians:
        gdata = {
            "mu": g.mu.numpy().tolist(),
            "rgb": g.rgb.numpy().tolist(),
            "alpha": float(g.alpha.numpy()[0]),
            "scale": g.s.numpy().tolist(),
            "rotation": g.p.numpy().tolist()
        }
        data["gaussians"].append(gdata)

    with open(json_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"✅ Saved {len(gaussians)} Gaussians to {json_path}")
