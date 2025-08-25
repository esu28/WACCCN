res_df = pd.DataFrame(results)
res_df = res_df.sort_values("ari_post", ascending=False).reset_index(drop=True)
display_cols = ["ari_post", "ari_pre", "method", "leiden_res", "graph_k", "umap_neighbors", "umap_min_dist", "k_smooth", "min_frac"]
print("\n=== TOP 15 CONFIGS (by ARI post-smoothing) ===")
print(res_df[display_cols].head(15).to_string(index=False))

best_emb = best["embedding"]
best_pre = best["labels_pre"]
best_post = best["labels_post"]

labels_grid_pre  = np.full((H, W), -1, dtype=int); labels_grid_pre[yy, xx]  = best_pre
labels_grid_post = np.full((H, W), -1, dtype=int); labels_grid_post[yy, xx] = best_post

hist_img = None

oords_px = np.stack([xx, yy], axis=1)
plt.figure(figsize=(6,6))
if hist_img is not None:
    plt.imshow(hist_img)
else:
    plt.imshow(np.ones((H, W, 3), dtype=float))
plt.scatter(coords_px[:,0], coords_px[:,1], c=best_post, s=10, alpha=0.9, linewidths=0)
plt.title("Clusters over spatial positions (best, post-smooth)")
plt.axis("off")
plt.show()
