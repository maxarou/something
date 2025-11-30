# src/visualization/viewer.py

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from ..environment.grid3d_to_2d import to_3d


# Store robot trails globally (reset each notebook run)
ROBOT_TRAILS = {}


def visualize_3d(env, explorer, step=0, elev=40, azim=40, figsize=(12, 8)):
    """
    Beautiful upgraded 3D visualizer:
    - Transparent colored floors
    - Robot movement trails
    - Clean obstacle & visited rendering
    - Distinct floor colors
    """
    H = env.H
    W = env.W
    F = env.F

    # Setup trails
    global ROBOT_TRAILS
    for r in explorer.robots:
        if r.id not in ROBOT_TRAILS:
            ROBOT_TRAILS[r.id] = []
        ROBOT_TRAILS[r.id].append(r.pos)

    # Prepare lists
    xs_obs = []; ys_obs = []; zs_obs = []
    xs_vis = []; ys_vis = []; zs_vis = []

    # Iterate unfolded grid
    for gx in range(H):
        for gy in range(env.unfolded_W):
            floor, x, y = to_3d(gx, gy, W)
            X = int(y)
            Y = int(x)
            Z = int(floor)

            if env.is_obstacle(gx, gy):
                xs_obs.append(X); ys_obs.append(Y); zs_obs.append(Z)
            elif explorer.global_visited[gx, gy]:
                xs_vis.append(X); ys_vis.append(Y); zs_vis.append(Z)

    # Colors for floors
    floor_colors = ["#ffcccc", "#ccffcc", "#ccccff", "#ffe6cc", "#e6ccff"]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Transparent colored planes for each floor
    for f in range(F):
        color = floor_colors[f % len(floor_colors)]
        Xg, Yg = np.meshgrid(np.arange(W), np.arange(H))
        Zg = np.full_like(Xg, f)
        ax.plot_surface(
            Xg, Yg, Zg,
            rstride=1, cstride=1,
            color=color,
            alpha=0.15,  # transparent
            linewidth=0
        )
        ax.text(
            W / 2, -1, f,
            f"Floor {f}",
            color="black",
            fontsize=12,
            ha="center"
        )

    # Obstacles (dark)
    ax.scatter(xs_obs, ys_obs, zs_obs, c="black", s=40, alpha=0.9)

    # Visited cells (blue)
    ax.scatter(xs_vis, ys_vis, zs_vis, c="blue", s=25, alpha=0.4)

    # Robots
    robot_colors = ["red", "green", "blue", "orange", "purple", "cyan"]

    for idx, r in enumerate(explorer.robots):
        gx, gy = r.pos
        floor, x, y = to_3d(int(gx), int(gy), W)
        rx, ry, rz = int(y), int(x), int(floor)

        # Robot dot
        ax.scatter([rx], [ry], [rz], s=180, c=robot_colors[idx], marker='o')
        ax.text(rx, ry, rz + 0.2, f"R{r.id}", color="black")

        # Robot trail
        trail = ROBOT_TRAILS[r.id]
        trail_X = []; trail_Y = []; trail_Z = []
        for (tx, ty) in trail:
            fl, xx, yy = to_3d(tx, ty, W)
            trail_X.append(int(yy))
            trail_Y.append(int(xx))
            trail_Z.append(int(fl))

        ax.plot(trail_X, trail_Y, trail_Z, color=robot_colors[idx], linewidth=2, alpha=0.7)

    ax.set_xlabel("Y (col)")
    ax.set_ylabel("X (row)")
    ax.set_zlabel("Floor")
    ax.set_title(f"Upgraded 3D View — Step {step}")

    ax.set_xlim(-1, W + 1)
    ax.set_ylim(-1, H + 1)
    ax.set_zlim(-1, F)

    ax.view_init(elev=elev, azim=azim)
    plt.tight_layout()
    plt.show()
