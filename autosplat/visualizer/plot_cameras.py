import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

def plot_cameras_plotly(cameras, arrow_frac=0.15, pad_frac=0.2):
    """
    Robust Plotly 3D camera visualization for Colab.

    Args:
        cameras: list of Camera objects with .location (3,) and .rotation_angles (3,)
        arrow_frac: fraction of scene diagonal that arrows will be (visible size)
        pad_frac: extra padding fraction for axis ranges
    """
    # Collect positions
    positions = np.array([np.asarray(cam.location, dtype=float) for cam in cameras])
    if positions.size == 0:
        raise ValueError("No cameras provided")

    # Compute scene bounds and center
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    center = (mins + maxs) / 2.0
    ranges = maxs - mins
    max_range = max(ranges.max(), 1e-6)  # avoid zero

    # Arrow length chosen as fraction of scene diagonal
    arrow_len = arrow_frac * max_range

    # Axis limits with padding
    pad = pad_frac * max_range
    xlim = [mins[0] - pad + center[0] - center[0], maxs[0] + pad + center[0] - center[0]]
    # above line keeps values relative — but simpler compute absolute:
    xlim = [mins[0] - pad, maxs[0] + pad]
    ylim = [mins[1] - pad, maxs[1] + pad]
    zlim = [mins[2] - pad, maxs[2] + pad]

    fig = go.Figure()

    # Plot camera positions
    fig.add_trace(go.Scatter3d(
        x=positions[:,0], y=positions[:,1], z=positions[:,2],
        mode='markers+text',
        marker=dict(size=4, color='red'),
        text=[f"{cam.camera_id}" for cam in cameras],
        textposition="top center",
        name="camera positions"
    ))

    # For each camera, plot orientation axes using Cone traces (R,G,B)
    for cam in cameras:
        pos = np.asarray(cam.location, dtype=float)
        # rotation matrix from Euler angles (ensure correct convention for your dataset)
        Rmat = R.from_euler('xyz', cam.rotation_angles).as_matrix()

        # local axes vectors: x=right, y=up, z=forward (we'll take camera forward as -Z)
        right_vec = Rmat[:,0]   # local +X
        up_vec    = Rmat[:,1]   # local +Y
        forward_vec = -Rmat[:,2]  # camera often looks along -Z; flip if needed

        # normalize and scale to arrow_len
        def scaled(v):
            v = np.asarray(v, dtype=float)
            n = np.linalg.norm(v)
            if n < 1e-12:
                return np.zeros(3)
            return (v / n) * arrow_len

        r_v = scaled(right_vec)
        u_v = scaled(up_vec)
        f_v = scaled(forward_vec)

        # cones: anchor tail so they emanate from camera position
        # sizeref controls cone head size relative to vector length; tune if needed
        sizeref = arrow_len * 0.3 if arrow_len>0 else 0.1

        # Right (red)
        fig.add_trace(go.Cone(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            u=[r_v[0]], v=[r_v[1]], w=[r_v[2]],
            colorscale=[[0,'red'],[1,'red']],
            showscale=False, sizemode="absolute", sizeref=sizeref, anchor="tail",
            name=f"right {cam.camera_id}"
        ))
        # Up (green)
        fig.add_trace(go.Cone(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            u=[u_v[0]], v=[u_v[1]], w=[u_v[2]],
            colorscale=[[0,'green'],[1,'green']],
            showscale=False, sizemode="absolute", sizeref=sizeref, anchor="tail",
            name=f"up {cam.camera_id}"
        ))
        # Forward (blue)
        fig.add_trace(go.Cone(
            x=[pos[0]], y=[pos[1]], z=[pos[2]],
            u=[f_v[0]], v=[f_v[1]], w=[f_v[2]],
            colorscale=[[0,'blue'],[1,'blue']],
            showscale=False, sizemode="absolute", sizeref=sizeref, anchor="tail",
            name=f"forward {cam.camera_id}"
        ))

    # set axis ranges and ensure equal aspect
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=xlim, title='X'),
            yaxis=dict(range=ylim, title='Y'),
            zaxis=dict(range=zlim, title='Z'),
            aspectmode='data'   # important: equal scaling on all axes
        ),
        width=900,
        height=800,
        margin=dict(l=0,r=0,b=0,t=30),
        showlegend=False
    )

    fig.update_layout(title="Camera positions & orientations (R=red, G=up, B=forward)")
    fig.show()
