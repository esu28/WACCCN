import cv2 

def classify(
    cluster_labels: np.ndarray,
    spot_coords: np.ndarray,
    unet_mask: np.ndarray,
    tumor_threshold: float,
) -> np.ndarray:

    if unet_mask.ndim == 3:
        processed_mask = cv2.cvtColor(unet_mask, cv2.COLOR_BGR2GRAY)
    else:
        processed_mask = unet_mask

    _, binary_tumor_mask = cv2.threshold(processed_mask, 200, 1, cv2.THRESH_BINARY)

    unique_clusters = np.unique(cluster_labels)
    cluster_is_tumor = {}

    for cluster_id in unique_clusters:
        indices = np.where(cluster_labels == cluster_id)[0]
        if len(indices) == 0: continue

        current_coords = spot_coords[indices].astype(int)
        y_coords, x_coords = current_coords[:, 1], current_coords[:, 0]

        spot_tumor_values = binary_tumor_mask[y_coords, x_coords]
        tumor_score = np.mean(spot_tumor_values)
        cluster_is_tumor[cluster_id] = tumor_threshold <= tumor_score

    new_labels_1d = np.zeros_like(cluster_labels, dtype=np.uint8)
    for i, original_label in enumerate(cluster_labels):
        if cluster_is_tumor.get(original_label, False):
            new_labels_1d[i] = 2  # tumor clutser
        else:
            new_labels_1d[i] = 1  #non tumorcluster

    return new_labels_1d

final_spot_labels = classify(
        cluster_labels=labels_filtered,
        spot_coords=coords_px_filtered,
        unet_mask=unetm_aligned,
        tumor_threshold=.56,
        tumor_threshold1=.6
    )

    plt.figure(figsize=(10, 10))

    scatter = plt.scatter(
        coords_px_filtered[:, 0],
        coords_px_filtered[:, 1],
        c=final_spot_labels,
        s=10,
        cmap=plt.cm.get_cmap('coolwarm', 2) 
    )

plt.title(f'classification (Threshold = {TUMOR_THRESHOLD})')

handles, _ = scatter.legend_elements()
legend_labels = ['Non-tmuor cluster', 'Tumor cluster']
plt.legend(handles, legend_labels, title="Classification")

plt.axis('off')
plt.show()
