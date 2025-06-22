import matplotlib.pyplot as plt
import numpy as np


def visualize_dolp(dolp):
    """
    Visualize Degree of Linear Polarization (DoLP).

    Args:
        dolp (np.ndarray): DoLP image.
    """
    plt.figure(figsize=(8, 4))
    plt.imshow(dolp, cmap='hot')
    plt.colorbar()
    plt.title('Degree of Linear Polarization (DoLP)')
    plt.axis('off')
    plt.show()


def visualize_aolp(aolp):
    """
    Visualize Angle of Linear Polarization (AoLP) with a labeled angle legend.

    Args:
        aolp (np.ndarray): AoLP image in radians.
    """
    plt.figure(figsize=(8, 4))
    im = plt.imshow(aolp, cmap='hsv', vmin=0, vmax=np.pi)
    cbar = plt.colorbar(im, ticks=[0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi])
    cbar.ax.set_yticklabels(['0°', '45°', '90°', '135°', '180°'])
    cbar.set_label('AoLP (degrees)')
    plt.title('Angle of Linear Polarization (AoLP)')
    plt.axis('off')
    plt.show()


def visualize_normals_rgb(normals):
    """
    Visualize surface normals using RGB color mapping.

    Args:
        normals (np.ndarray): Surface normal vectors.
    """
    norms = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals_unit = normals / (norms + 1e-8)
    normals_rgb = (normals_unit + 1) / 2

    plt.figure(figsize=(8, 8))
    plt.imshow(normals_rgb)
    plt.title("Surface Normals (RGB)")
    plt.axis("off")
    plt.show()

def visualize_normals_2d(N, step=10, scale=1.0):
    """
    Visualize 2D surface normal vectors using arrows.

    Parameters:
    - N: numpy array of shape (H, W, 3), the normal vectors.
    - step: int, sampling step to reduce the number of arrows for clarity.
    - scale: float, scaling factor for arrow lengths.
    """
    H, W, _ = N.shape
    Y, X = np.mgrid[0:H:step, 0:W:step]
    U = N[::step, ::step, 0] * scale
    V = N[::step, ::step, 1] * scale

    plt.figure(figsize=(8, 6))
    plt.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1, color='red')
    plt.gca().invert_yaxis()
    plt.axis('equal')
    plt.title('2D Surface Normal Vectors')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.grid(True)
    plt.show()

def show_reference_normal_colors():
    """
    Shows colors corresponding to positive and negative directions of X, Y, Z
    using the standard X→R, Y→G, Z→B color mapping.
    """
    directions = {
        '+X': [1, 0, 0],
        '-X': [-1, 0, 0],
        '+Y': [0, 1, 0],
        '-Y': [0, -1, 0],
        '+Z': [0, 0, 1],
        '-Z': [0, 0, -1],
    }

    vectors = np.array([directions[k] for k in directions])
    vectors = vectors.reshape((1, 6, 3))

    normals_unit = vectors / (np.linalg.norm(vectors, axis=-1, keepdims=True) + 1e-8)
    normals_rgb = (normals_unit + 1) / 2

    plt.imshow(normals_rgb, extent=[0, 6, 0, 1])
    plt.xticks(np.arange(6) + 0.5, list(directions.keys()))
    plt.title("Surface Normal Reference Colors (X→R, Y→G, Z→B)")
    plt.axis("off")
    plt.show()

