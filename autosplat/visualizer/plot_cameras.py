import numpy as np
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation as R

def plot_cameras_plotly(cameras, arrow_frac=0.15, pad_frac=0.2, fov=60):
    """
    Enhanced Plotly 3D camera visualization with clear orientation and frustum.
    
    Args:
        cameras: list of Camera objects with .location (3,) and .rotation_angles (3,)
        arrow_frac: fraction of scene diagonal that arrows will be
        pad_frac: padding around scene
        fov: horizontal field of view (degrees) for frustum visualization
    """
    # --- Gather camera positions ---
    positions = np.array([np.asarray(cam.location, dtype=float) for cam in cameras])
    if positions.size == 0:
        raise ValueError("No cameras provided")

    # --- Scene bounds and scaling ---
    mins = positions.min(axis=0)
    maxs = positions.max(axis=0)
    ranges = maxs - mins
    max_range = max(ranges.max(), 1e-6)
    pad = pad_frac * max_range

    xlim = [mins[0] - pad, maxs[0] + pad]
    ylim = [mins[1] - pad, maxs[1] + pad]
    zlim = [mins[2] - pad, maxs[2] + pad]

    arrow_len = arrow_frac * max_range

    fig = go.Figure()

    # --- Camera positions ---
    fig.add_trace(go.Scatter3d(
        x=positions[:,0], y=positions[:,1], z=positions[:,2],
        mode='markers+text',
        marker=dict(size=4, color='red'),
        text=[f"{cam.camera_id}" for cam in cameras],
        textposition="top center",
        name="Camera centers"
    ))

    # --- Helper to draw frustum ---
    def draw_camera_frustum(fig, pos, Rmat, fov, scale, color='blue'):
        """Draws a simple wireframe pyramid showing camera FOV."""
        z = scale
        w = np.tan(np.deg2rad(fov / 2)) * z
        h = w

        corners = np.array([
            [ w,  h, z],
            [-w,  h, z],
            [-w, -h, z],
            [ w, -h, z],
        ])

        corners_world = (Rmat @ corners.T).T + pos

        # Lines from camera to corners
        for c in corners_world:
            fig.add_trace(go.Scatter3d(
                x=[pos[0], c[0]], y=[pos[1], c[1]], z=[pos[2], c[2]],
                mode='lines', line=dict(color=color, width=2),
                showlegend=False
            ))

        # Connect corners
        idxs = [0,1,2,3,0]
        fig.add_trace(go.Scatter3d(
            x=corners_world[idxs,0], y=corners_world[idxs,1], z=corners_world[idxs,2],
            mode='lines', line=dict(color=color, width=2),
            showlegend=False
        ))

    # --- Plot camera orientations ---
    for cam in cameras:
        pos = np.asarray(cam.location, dtype=float)
        Rmat = R.from_euler('xyz', cam.rotation_angles).as_matrix()

        right_vec = Rmat[:,0]
        up_vec = Rmat[:,1]
        forward_vec = -Rmat[:,2]  # -Z is typically "look" direction

        def scaled(v):
            n = np.linalg.norm(v)
            return (v / n) * arrow_len if n > 1e-12 else np.zeros(3)

        r_v = scaled(right_vec)
        u_v = scaled(up_vec)
        f_v = scaled(forward_vec)

        sizeref = arrow_len * 0.3

        # Local axes cones
        for vec, color, label in [(r_v, 'red', 'right'), (u_v, 'green', 'up'), (f_v, 'blue', 'forward')]:
            fig.add_trace(go.Cone(
                x=[pos[0]], y=[pos[1]], z=[pos[2]],
                u=[vec[0]], v=[vec[1]], w=[vec[2]],
                colorscale=[[0, color], [1, color]],
                showscale=False, sizemode="absolute", sizeref=sizeref,
                anchor="tail", name=f"{label} {cam.camera_id}"
            ))

        # Long forward arrow for clarity
        look_len = arrow_len * 3
        look_dir = (forward_vec / np.linalg.norm(forward_vec)) * look_len
        look_end = pos + look_dir
        fig.add_trace(go.Scatter3d(
            x=[pos[0], look_end[0]],
            y=[pos[1], look_end[1]],
            z=[pos[2], look_end[2]],
            mode='lines',
            line=dict(color='blue', width=6),
            name=f"look_dir {cam.camera_id}",
            showlegend=False
        ))

        # Draw frustum pyramid
        draw_camera_frustum(fig, pos, Rmat, fov=fov, scale=arrow_len*2, color='blue')

    # --- Layout ---
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=xlim, title='X'),
            yaxis=dict(range=ylim, title='Y'),
            zaxis=dict(range=zlim, title='Z'),
            aspectmode='data'
        ),
        width=900,
        height=800,
        margin=dict(l=0, r=0, b=0, t=40),
        title="Camera positions and orientations (R=Right, G=Up, B=Forward)",
        showlegend=False
    )

    fig.show()
