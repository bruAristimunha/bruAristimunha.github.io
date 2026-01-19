"""
.. _logeig-animation:

LogEig Layer Animation
======================

This animation visualizes how the LogEig layer projects SPD matrices
from the curved Riemannian manifold to a flat tangent space.

.. math::

    \\text{LogEig}(X) = U \\log(\\Lambda) U^T

where :math:`X = U \\Lambda U^T` is the eigendecomposition.

.. contents:: This visualization shows:
   :local:
   :depth: 2

"""

# sphinx_gallery_thumbnail_number = 1

######################################################################
# Understanding LogEig
# --------------------
#
# The SPD manifold is curved - straight lines don't exist. LogEig
# projects points to the tangent space at the identity, where we can
# use Euclidean operations (like classification with linear layers).
#
# Key insight: The matrix logarithm "flattens" the manifold locally.
#

import sys

from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# Handle both direct execution and import
try:
    _current_dir = Path(__file__).parent
except NameError:
    _current_dir = Path.cwd() / "examples" / "visualizations"

if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

from spd_visualization_utils import create_ellipse_patch, setup_spd_plot


######################################################################
# Setup and Data Generation
# -------------------------
#

np.random.seed(42)

n_matrices = 6

# Create SPD matrices with varying eigenvalues
eigval_sets = [
    np.array([0.5, 1.5]),
    np.array([1.0, 2.0]),
    np.array([0.3, 3.0]),
    np.array([1.5, 1.5]),
    np.array([0.8, 2.5]),
    np.array([2.0, 0.5]),
]

# Random rotations
angles = np.linspace(0, np.pi * 0.8, n_matrices)
rotation_matrices = [
    np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]) for a in angles
]

# Create input SPD matrices
input_matrices = []
for eigvals, U in zip(eigval_sets, rotation_matrices):
    X = U @ np.diag(eigvals) @ U.T
    input_matrices.append(X)

# Apply LogEig: log of eigenvalues
output_matrices = []
for eigvals, U in zip(eigval_sets, rotation_matrices):
    log_eigvals = np.log(eigvals)
    Y = U @ np.diag(log_eigvals) @ U.T
    output_matrices.append(Y)

# Colors
colors = plt.cm.plasma(np.linspace(0.2, 0.8, n_matrices))

# Print transformation
print("LogEig: SPD Manifold -> Tangent Space")
print("=" * 50)
for i, (inp_eig, out_eig) in enumerate(zip(eigval_sets, eigval_sets)):
    log_eig = np.log(out_eig)
    print(
        f"Matrix {i+1}: lambda={inp_eig} -> log(lambda)=[{log_eig[0]:.2f}, {log_eig[1]:.2f}]"
    )

######################################################################
# Static Visualization: Log Function
# ----------------------------------
#
# First, let's visualize how the log function transforms eigenvalues.
#

fig_static, axes_static = plt.subplots(1, 2, figsize=(14, 5))

# Log function on eigenvalues
ax1 = axes_static[0]
x = np.linspace(0.1, 4, 200)
y = np.log(x)

ax1.plot(x, y, "b-", linewidth=3, label="log(lambda)")
ax1.axhline(y=0, color="k", linestyle="--", alpha=0.3)
ax1.axvline(x=1, color="k", linestyle="--", alpha=0.3)
ax1.fill_between(
    x[x < 1], y[x < 1], 0, alpha=0.2, color="red", label="Compression (lambda < 1)"
)
ax1.fill_between(
    x[x > 1], 0, y[x > 1], alpha=0.2, color="green", label="Expansion (lambda > 1)"
)

# Mark eigenvalues
for i, eigvals in enumerate(eigval_sets):
    for ev in eigvals:
        ax1.scatter(
            [ev],
            [np.log(ev)],
            s=100,
            c=[colors[i]],
            edgecolors="black",
            linewidth=1.5,
            zorder=5,
        )

ax1.scatter(
    [1],
    [0],
    s=200,
    c="gold",
    marker="*",
    edgecolors="black",
    linewidth=2,
    zorder=10,
    label="Identity (lambda=1, log(lambda)=0)",
)

ax1.set_xlim(0, 4)
ax1.set_ylim(-2.5, 2)
ax1.set_xlabel("Eigenvalue lambda", fontsize=12)
ax1.set_ylabel("log(lambda)", fontsize=12)
ax1.set_title("LogEig: Eigenvalue Transformation", fontsize=13, fontweight="bold")
ax1.legend(loc="lower right", fontsize=9)
ax1.grid(True, alpha=0.3)

# Manifold to Tangent Space diagram
ax2 = axes_static[1]
ax2.set_xlim(-2, 6)
ax2.set_ylim(-2, 4)
ax2.set_aspect("equal")
ax2.axis("off")
ax2.set_title("LogEig: Manifold -> Tangent Space", fontsize=13, fontweight="bold")

# Draw manifold (curved surface)
theta = np.linspace(-0.5, 2, 50)
r = 1.5
x_curve = r * np.cosh(theta)
y_curve = r * np.sinh(theta)
ax2.plot(x_curve - 0.5, y_curve + 1, "b-", linewidth=3, label="SPD Manifold")
ax2.fill_between(x_curve - 0.5, y_curve + 1, -2, alpha=0.1, color="blue")

# Draw tangent plane (flat)
ax2.plot([3, 6], [1, 1], "orange", linewidth=3, label="Tangent Space")
ax2.fill_between([3, 6], [1, 1], [-1, -1], alpha=0.1, color="orange")

# Draw projection arrows
ax2.annotate(
    "",
    xy=(4, 1),
    xytext=(1.5, 2),
    arrowprops=dict(
        arrowstyle="->", color="gray", lw=2, connectionstyle="arc3,rad=-0.2"
    ),
)
ax2.text(2.5, 2.2, "LogEig", fontsize=11, ha="center", fontweight="bold")

# Identity point
ax2.scatter(
    [1], [1], s=200, c="gold", marker="*", edgecolors="black", linewidth=2, zorder=10
)
ax2.text(1, 0.5, "Identity", ha="center", fontsize=10)

# Points
ax2.scatter(
    [0.5, 1.5, 2], [1.5, 2.5, 1.8], s=100, c="blue", edgecolors="black", linewidth=1.5
)
ax2.scatter(
    [3.5, 4.5, 5], [1, 1, 1], s=100, c="orange", edgecolors="black", linewidth=1.5
)

ax2.legend(loc="upper left", fontsize=10)

plt.tight_layout()

######################################################################
# Mathematical Explanation
# ------------------------
#
# LogEig projects SPD matrices to symmetric matrices (tangent space):
#
# 1. **Eigendecomposition**: :math:`X = U \Lambda U^T`
# 2. **Log transformation**: :math:`\log(\Lambda) = \text{diag}(\log \lambda_i)`
# 3. **Reconstruction**: :math:`Y = U \log(\Lambda) U^T`
#
# The tangent space at identity is the space of symmetric matrices,
# which is a flat (Euclidean) vector space. This allows standard
# machine learning operations like linear classification.
#

######################################################################
# 3D Manifold Visualization Helpers
# ---------------------------------
#
# The SPD(2) manifold can be embedded in 3D as a cone-like surface.
#


def draw_spd_cone(ax, alpha=0.2):
    """Draw the SPD cone in 3D."""
    u = np.linspace(0.1, 3, 30)  # trace/2
    v = np.linspace(0, 2 * np.pi, 30)
    U, V = np.meshgrid(u, v)

    R = 0.8 * U  # radius (constraint for SPD)
    X = U + R * np.cos(V)  # a
    Y = R * np.sin(V)  # b
    Z = U - R * np.cos(V)  # c

    ax.plot_surface(X, Y, Z, alpha=alpha, cmap="viridis", edgecolor="none")
    ax.set_xlabel("a (diagonal)", fontsize=10)
    ax.set_ylabel("b (off-diag)", fontsize=10)
    ax.set_zlabel("c (diagonal)", fontsize=10)


def spd_to_3d_coords(spd):
    """Convert 2x2 SPD matrix to 3D coordinates."""
    return spd[0, 0], spd[0, 1], spd[1, 1]


def sym_to_3d_coords(sym):
    """Convert 2x2 symmetric matrix to 3D coordinates (tangent space)."""
    return sym[0, 0], sym[0, 1], sym[1, 1]


######################################################################
# Animation
# ---------
#
# The animation shows the LogEig projection from the curved SPD manifold
# to the flat tangent space.

# Create animation figure
fig_anim = plt.figure(figsize=(16, 6))

# 3D view of manifold (cone) and tangent plane
ax_manifold = fig_anim.add_subplot(1, 3, 1, projection="3d")
# Input ellipses (SPD manifold)
ax_input = fig_anim.add_subplot(1, 3, 2)
# Output (tangent space - symmetric matrices)
ax_output = fig_anim.add_subplot(1, 3, 3)

# Animation parameters
n_frames = 60
pause_frames = 20

# Positions for ellipses
y_positions = np.linspace(2, -2, n_matrices)
centers_input = [(-0.5, y) for y in y_positions]
centers_output = [(0.5, y) for y in y_positions]


def animate(frame):
    """Animation frame update."""
    # Progress
    if frame < pause_frames:
        t = 0.0
        phase = "SPD Manifold (curved)"
    elif frame < pause_frames + n_frames:
        t = (frame - pause_frames) / n_frames
        t = 0.5 * (1 - np.cos(np.pi * t))  # Smooth easing
        phase = f"Projecting to tangent space (t={t:.2f})"
    else:
        t = 1.0
        phase = "Tangent Space (flat)"

    # Clear all axes
    ax_manifold.clear()
    ax_input.clear()
    ax_output.clear()

    # Draw 3D manifold visualization
    ax_manifold.set_title("SPD Manifold (Cone)", fontsize=12, fontweight="bold")
    draw_spd_cone(ax_manifold, alpha=0.15)

    # Draw tangent plane at identity
    plane_size = 2
    xx, yy = np.meshgrid(
        np.linspace(1 - plane_size, 1 + plane_size, 5),
        np.linspace(-plane_size, plane_size, 5),
    )
    zz = np.ones_like(xx)  # z = 1 plane (at identity)

    if t > 0.3:
        ax_manifold.plot_surface(
            xx,
            yy,
            zz,
            alpha=0.2 * min(1, (t - 0.3) / 0.3),
            color="orange",
            edgecolor="none",
        )
        ax_manifold.text(1, 0, 1, "Tangent Space", fontsize=9, color="orange")

    # Plot points on manifold and their projections
    for i, (inp, out) in enumerate(zip(input_matrices, output_matrices)):
        # Point on manifold
        x, y, z = spd_to_3d_coords(inp)
        ax_manifold.scatter(
            [x], [y], [z], s=80, c=[colors[i]], edgecolors="black", linewidth=1
        )

        # Projected point (on tangent space at identity)
        if t > 0:
            x_proj, y_proj, z_proj = sym_to_3d_coords(out)
            # Shift to tangent space at identity (centered at (1, 0, 1))
            x_proj_shifted = 1 + x_proj
            z_proj_shifted = 1 + z_proj

            # Interpolate
            x_curr = x + t * (x_proj_shifted - x)
            y_curr = y + t * (y_proj - y)
            z_curr = z + t * (z_proj_shifted - z)

            ax_manifold.scatter(
                [x_curr],
                [y_curr],
                [z_curr],
                s=80,
                c=[colors[i]],
                marker="^",
                edgecolors="black",
                linewidth=1,
                alpha=t,
            )

            # Draw projection line
            ax_manifold.plot(
                [x, x_curr],
                [y, y_curr],
                [z, z_curr],
                color=colors[i],
                linestyle=":",
                linewidth=1,
                alpha=0.5,
            )

    # Mark identity
    ax_manifold.scatter(
        [1],
        [0],
        [1],
        s=150,
        c="gold",
        marker="*",
        edgecolors="black",
        linewidth=1.5,
        zorder=10,
    )
    ax_manifold.text(1.1, 0.1, 1.1, "I", fontsize=10, fontweight="bold")

    ax_manifold.set_xlim(0, 4)
    ax_manifold.set_ylim(-2, 2)
    ax_manifold.set_zlim(0, 4)
    ax_manifold.view_init(elev=20, azim=45 + frame * 0.3)

    # Input ellipses (SPD manifold)
    setup_spd_plot(
        ax_input, xlim=(-3, 3), ylim=(-3.5, 3.5), title="Input: SPD Matrices"
    )

    for i, (inp, center) in enumerate(zip(input_matrices, centers_input)):
        ellipse = create_ellipse_patch(
            inp,
            center,
            alpha=0.6 * (1 - 0.5 * t),
            color=colors[i],
            edgecolor="black",
            linewidth=2,
        )
        ax_input.add_patch(ellipse)

        # Label
        eigvals = eigval_sets[i]
        ax_input.text(
            center[0] + 1.5,
            center[1],
            f"lambda={eigvals}",
            fontsize=8,
            va="center",
            color=colors[i],
        )

    ax_input.text(
        0,
        -3.2,
        "Curved manifold\n(non-Euclidean)",
        ha="center",
        fontsize=9,
        style="italic",
    )

    # Output (tangent space)
    setup_spd_plot(
        ax_output,
        xlim=(-3, 3),
        ylim=(-3.5, 3.5),
        title=r"Output: $U \log(\Lambda) U^T$",
    )

    for i, (inp, out, center) in enumerate(
        zip(input_matrices, output_matrices, centers_output)
    ):
        # Interpolate eigenvalues (not matrices, for visualization)
        inp_eigvals = eigval_sets[i]
        out_eigvals = np.log(inp_eigvals)
        current_eigvals = np.exp((1 - t) * np.log(inp_eigvals) + t * out_eigvals)

        # Create interpolated matrix
        U = rotation_matrices[i]

        # For tangent space, we need to handle negative eigenvalues
        # Use abs for visualization purposes
        vis_eigvals = np.abs(current_eigvals)
        vis_matrix = (
            U @ np.diag(vis_eigvals + 0.1) @ U.T
        )  # Add small offset for visibility

        ellipse = create_ellipse_patch(
            vis_matrix,
            center,
            alpha=0.3 + 0.4 * t,
            color=colors[i],
            edgecolor="black",
            linewidth=2,
        )
        ax_output.add_patch(ellipse)

        # Show log eigenvalues
        if t > 0.5:
            log_eigvals = np.log(inp_eigvals)
            ax_output.text(
                center[0] + 1.5,
                center[1],
                f"log(lambda)=[{log_eigvals[0]:.1f},{log_eigvals[1]:.1f}]",
                fontsize=8,
                va="center",
                color=colors[i],
            )

    ax_output.text(
        0,
        -3.2,
        "Flat tangent space\n(Euclidean)",
        ha="center",
        fontsize=9,
        style="italic",
    )

    fig_anim.suptitle(f"LogEig — {phase}", fontsize=14, fontweight="bold")

    return []


def init():
    """Initialize animation."""
    return []


# Create the animation - must be assigned to a variable that persists
total_frames = 2 * pause_frames + n_frames
anim = animation.FuncAnimation(
    fig_anim, animate, init_func=init, frames=total_frames, interval=50, blit=False
)

plt.tight_layout()
plt.show()
