#%% 
# imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from ResampleMatrix import getBounds
from PIL import ImageDraw
import os
# %%
def imgAsMat(path):
    img = Image.open(path)
    matrix = np.asarray(img)

    # separate channels
    red = pd.DataFrame(matrix[:,:,0])
    green = pd.DataFrame(matrix[:,:,1])
    blue = pd.DataFrame(matrix[:,:,2])
    return red.to_numpy(), green.to_numpy(), blue.to_numpy() 

def matToImg(red, green, blue):
    matrix = np.dstack((red, green, blue))
    return Image.fromarray(matrix)

def showChannels(red, green, blue): #probably not very useful but fun to look at
    matrix = np.vstack((red, green, blue))
    return Image.fromarray(matrix)

def normalizeChannels(red, green, blue):
    nRed = cv2.normalize(red, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    nGreen = cv2.normalize(green, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    nBlue = cv2.normalize(blue, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    return red, green, blue

#%%
#Z score normalization for RGB images per channel
def normalizeImages(input_dir, output_dir):
    for fname in os.listdir(input_dir):
        if fname.lower().endswith('.png'):
            path = os.path.join(input_dir, fname)
            img = Image.open(path).convert('RGB')  # need to make sure it's RGB
            img_np = np.array(img).astype(np.float32)  # shape: (H, W, 3)

            z_norm = np.zeros_like(img_np)

            for c in range(3):  # For each channel: R, G, B
                channel = img_np[:, :, c]
                mean = np.mean(channel)
                std = np.std(channel)

                if std == 0:
                    print(f"Skipping {fname} channel {c}: std=0")
                    continue

                z_norm[:, :, c] = (channel - mean) / std

            save_path = os.path.join(output_dir, os.path.splitext(fname)[0] + ".npy")
            np.save(save_path, z_norm)

            print(f"Normalized RGB and saved: {save_path}")

    print("All RGB images Z-score normalized :)")

#%%
def cropImage(img, S, scaleFactor):
    S = S.copy()
    def swapAxes(spatial):
        spatial['x'], spatial['y'] = spatial['y'], spatial['x']
        return spatial

    def toPixels(spot):
        pixel = int(spot * scaleFactor)
        return pixel
    
    S = swapAxes(S)
    S['x'] = S['x'].apply(toPixels)
    S['y'] = S['y'].apply(toPixels)
    xMin, xMax, yMin, yMax = getBounds(S)

    shape = [xMin, yMin, xMax, yMax]
    cropped = img

    cropped = cropped.crop(shape)
    
    return cropped
#%%
def overlayST(img, STImg, origin = (0,0)):
    img = img.copy()
    STImg = STImg.copy()
    img = img.convert("RGBA")
    STImg.putalpha(150)
    img.paste(STImg, origin, STImg)
    return img