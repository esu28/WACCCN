import numpy as np
import pandas as pd

def map_to_nearest(spot_coord_x,spot_coord_y,coords_orig):
  x_rows = coords_orig.loc[coords_orig["x"] == spot_coord_x]
  closest_value_y = min(x_rows["y"], key=lambda x: abs(x - spot_coord_y))
  if abs(closest_value_y - spot_coord_y >= 5):
    print(f"warning: barcode with coordinates {spot_coord_x} and {spot_coord_y} might be off:")
    print(x_rows)
  return closest_value_y

spot_labels = pd.DataFrame(np.load("/content/drive/MyDrive/Project/Copy of spot labels.npy"))
spot_coords = pd.DataFrame(np.load("/content/drive/MyDrive/Project/Copy of spot coords.npy"))
coords_orig = pd.read_csv("/content/drive/MyDrive/Project/OSCC/Data1/coordinates_img.csv")
spot_coords.columns = ["x", "y"]
spot_coords["y"] = spot_coords["y"] / 1.154

for i, spot_row in spot_coords.iterrows():
  spot_coords.at[i, "y"] = int(map_to_nearest(spot_row["x"], spot_row["y"],coords_orig))

coords_merged = pd.merge(spot_coords, coords_orig, on=['x', 'y'], how='inner')
print(coords_merged.head(20))
print(coords_merged.shape)


concat = pd.concat([spot_coords, spot_labels], axis=1)
concat.columns = ["x", "y", "label"]
print(concat.head())
print(concat.shape)

final_labels = pd.merge(coords_merged, concat, on=['x', 'y'], how='inner')
final_labels["label"] = final_labels["label"]-1

final_labels.drop("x", inplace=True, axis=1)
final_labels.drop("y", inplace=True, axis=1)
print(final_labels.head())
print(final_labels.shape)
final_labels.to_csv("/content/drive/MyDrive/Project/OSCC/DSAG/DSAG_inputs/labels_dsag_input_nocoords.csv")
