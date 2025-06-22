#!/usr/bin/env python3
import numpy as np
import imageio
import matplotlib.pyplot as plt
from polarisationImage import polarisation_image
from DCT import dct_poisson
from plot_utils import visualize_normals_2d, visualize_normals_rgb
from surfaceNormal import surface_normal_from_zenith_azimuth
from scipy.signal import medfilt2d
from mpl_toolkits.mplot3d import Axes3D

def demo_depth_from_polarization():
    # Load the captured image of four polarization angles
    # in_path = '../data/milk_can2_bad.tiff'
    im0_path = '../data/beanball_0.jpg'
    im45_path = '../data/beanball_45.jpg'
    im90_path = '../data/beanball_90.jpg'
    im135_path = '../data/beanball_135.jpg'
    # im = imageio.imread(in_path)
    
    # Stack the image of four polarization angles into a 4-channel image
    # nrows, ncols = im.shape
    # h2, w2 = nrows // 2, ncols // 2
    # im0   = im[0:h2,   0:w2]
    # im45  = im[0:h2,   w2:ncols]
    # im90  = im[h2:nrows, 0:w2]
    # im135 = im[h2:nrows, w2:ncols]
    im0   = imageio.imread(im0_path)
    im45  = imageio.imread(im45_path)
    im90  = imageio.imread(im90_path)
    im135 = imageio.imread(im135_path)
    
    images = np.stack([
        im0.astype(float),
        im45.astype(float),
        im90.astype(float),
        im135.astype(float)
    ], axis=2)
    
    # downsample for speed
    nskips = 4
    images = images[::nskips, ::nskips, :]
    
    # Assign polarization angles (radians)
    angles = np.deg2rad([0, 45, 90, 135])
    
    # Optional foreground mask by intensity threshold
    use_fg_threshold = True
    mask = np.ones(images.shape[:2], dtype=bool)
    if use_fg_threshold:
        image_avg = images.mean(axis=2)
        fg_threshold = 10
        mask = image_avg >= fg_threshold
    
    # plt.figure()
    # plt.imshow(mask, cmap='gray')
    # plt.title('Foreground Mask')
    # plt.show()
    
    # Compute polarization attributes (requires port of PolarisationImage)
    dolp_est, aolp_est, intensity_est = polarisation_image(
        images, angles, mask, method='linear'
    )
    
    # filter out very low DoLP
    mask[dolp_est < 0.005] = False
    
    plt.figure()
    plt.imshow(aolp_est, cmap='rainbow')
    plt.colorbar()
    plt.title('AoLP')
    plt.show()
    
    plt.figure()
    plt.imshow(dolp_est)
    plt.colorbar()
    plt.title('DoLP')
    plt.show()
    
    # plt.figure()
    # plt.imshow(intensity_est, cmap='gray')
    # plt.title('Intensity')
    # plt.show()
    
    # Estimate surface normals (port of lookup_aolp_cylinder)
    N = surface_normal_from_zenith_azimuth(dolp_est, aolp_est)
    
    # optional median filtering to reduce noise
    for i in range(3):
        N[:, :, i] = medfilt2d(N[:, :, i], kernel_size=5)
    
    # Depth reconstruction via Poisson integration (port of DCT_Poisson)
    # avoid divide-by-zero
    nz = np.abs(N[:, :, 2]) < 1e-5
    N[nz, 2] = np.nan
    
    P = -N[:, :, 0] / N[:, :, 2]
    Q = -N[:, :, 1] / N[:, :, 2]
    P = np.nan_to_num(P)
    Q = np.nan_to_num(Q)
    
    height = dct_poisson(P, Q)
    
    plt.figure()
    plt.imshow(height)
    plt.colorbar()
    plt.title('Reconstructed Height')
    plt.show()
    
    # mask out background again
    height[~mask] = np.nan
    
    # 3D surface plot
    ys, xs = np.indices(height.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(
        xs, ys, height,
        cmap='cool',  # or any other colormap
        edgecolor='none',
        linewidth=0,
        antialiased=False,
        shade=True
    )    
    ax.set_box_aspect([1,1,1])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    plt.title('Depth Surface')
    plt.show()

    print("NORMALS SHAPE\n", N.shape)
    visualize_normals_rgb(N)

    visualize_normals_2d(N, step=10, scale=500.0)

if __name__ == '__main__':
    demo_depth_from_polarization()