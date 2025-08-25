import matplotlib.pyplot as plt
import numpy as np

def visualize_overlap(
    cluster_labels: np.ndarray,
    spot_coords: np.ndarray,
    unet_mask: np.ndarray
):

    H, W = unet_mask.shape[:2]

    plt.figure(figsize=(8, 8))

    plt.imshow(unet_mask, cmap='gray', extent=(0, W, H, 0))

    plt.scatter(
        spot_coords[:, 0],  #x-coords
        spot_coords[:, 1],  #y-coords
        c=cluster_labels,
        s=8,     
        alpha=0.6,
        linewidths=0,
        cmap='viridis'   
    )

    plt.title("Overlap of Clusters on UNet Mask")
    plt.axis("off")
    plt.show()

visualize_overlap(
        cluster_labels=best['labels_post'],
        spot_coords=coords_px,
        unet_mask=unetm_aligned
)
