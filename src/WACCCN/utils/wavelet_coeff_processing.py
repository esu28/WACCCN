# %% imports

import os
import numpy as np
import pandas as pd
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from ResampleMatrix import getBounds, partitionMatrix, resampleMatrix, plotResampledMatrix, upsampleToImage

#%%
def readWaveletCoeffs(path):
    coeffs_dict = {}
    for file in os.listdir(path):
        if file.endswith('.npz'):
            npzFile = np.load(os.path.join(path, file))
            coeffs_dict[os.path.splitext(file)[0]] = npzFile[npzFile.files[0]]
        elif file.endswith('.npy'):
            npyFile = np.load(os.path.join(path, file))
            coeffs_dict[os.path.splitext(file)[0]] = npyFile
        elif file.endswith('.csv'):
            df = pd.read_csv(os.path.join(path, file))
            coeffs_dict[os.path.splitext(file)[0]] = np.asarray(df)
    return coeffs_dict
#%%
def plotWavelets(coeffs, geneName):
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    # Gather all non-zero values for global vmin/vmax (since we only plot non-zero values)
    all_values = []
    for key in coeffs:
        mat = np.asarray(coeffs[key])
        nonzero_vals = mat[np.nonzero(mat)]
        if len(nonzero_vals) > 0:
            all_values.append(nonzero_vals)
            print(f"{key}: min={np.min(nonzero_vals):.3f}, max={np.max(nonzero_vals):.3f}")
    all_values = np.concatenate(all_values)
    vmin, vmax = np.min(all_values), np.max(all_values)
    #print(f"Global vmin: {vmin:.3f}, vmax: {vmax:.3f}")

    #different dot sizes for different levels
    s1 = 9
    s2 = 1

    fig = plt.figure(figsize=(13, 13))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])

    # Top-left: nested 2x2 for level 2
    gs2 = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=gs[0, 0])
    ax_ll2 = plt.subplot(gs2[0, 0])
    ax_lh2 = plt.subplot(gs2[0, 1])
    ax_hl2 = plt.subplot(gs2[1, 0])
    ax_hh2 = plt.subplot(gs2[1, 1])

    # Plot level 2 coefficients
    sc_ll2 = plotResampledMatrix(coeffs['L2_B00'], geneName, ax=ax_ll2, title='LL2', vmin=vmin, vmax=vmax, s=s2, show_colorbar=False)
    plotResampledMatrix(coeffs['L2_B01'], geneName, ax=ax_lh2, title='LH2', vmin=vmin, vmax=vmax, s=s2, show_colorbar=False)
    plotResampledMatrix(coeffs['L2_B10'], geneName, ax=ax_hl2, title='HL2', vmin=vmin, vmax=vmax, s=s2, show_colorbar=False)
    plotResampledMatrix(coeffs['L2_B11'], geneName, ax=ax_hh2, title='HH2', vmin=vmin, vmax=vmax, s=s2, show_colorbar=False)

    # Top-right: LH
    ax_lh = plt.subplot(gs[0, 1])
    plotResampledMatrix(coeffs['L1_B00'], geneName, ax=ax_lh, title='LH', vmin=vmin, vmax=vmax, s=s1, show_colorbar=False)

    # Bottom-left: HL
    ax_hl = plt.subplot(gs[1, 0])
    plotResampledMatrix(coeffs['L1_B10'], geneName, ax=ax_hl, title='HL', vmin=vmin, vmax=vmax, s=s1, show_colorbar=False)

    # Bottom-right: HH
    ax_hh = plt.subplot(gs[1, 1])
    plotResampledMatrix(coeffs['L1_B11'], geneName, ax=ax_hh, title='HH', vmin=vmin, vmax=vmax, s=s1, show_colorbar=False)

    # Add a single colorbar for the whole figure
    cbar = fig.colorbar(
        sc_ll2, 
        ax=fig.get_axes(), 
        orientation='vertical', 
        fraction=0.025, 
        pad=1,
        label='Value',
        aspect=30      # height of colorbar
    )
    cbar.set_label('Value')

    plt.subplots_adjust(right=0.85)# leave space on the right for the colorbar
    #plt.tight_layout()  
    plt.show()

#%%
# inflexible so might need to change later
def resampleCoeffs(coeffs, S, D, scaleFactor, stack=False):
    upsampledCoeffs = {}
    L1 = ["L1_B01", "L1_B10", "L1_B11"]
    L2 = ["L2_B00", "L2_B01", "L2_B10", "L2_B11"]
    for key, value in coeffs.items():
        if key in L1:
            upsampled = upsampleToImage(value, S, D, scaleFactor, wv=2)
        elif key in L2:
            upsampled = upsampleToImage(value, S, D, scaleFactor, wv=4)
        else:
            continue
        upsampledCoeffs[key] = upsampled
    if stack:
        sorted_keys = sorted(upsampledCoeffs.keys())
        print(sorted_keys)
        arrays_to_stack = [upsampledCoeffs[key] for key in sorted_keys]
        upsampledCoeffs = np.stack(arrays_to_stack, axis=2)
        return upsampledCoeffs
    else:
        return upsampledCoeffs
#%%
def exportCoeffs(coeffs, path, geneName, stacked=False):
    '''
    Uploads the coefficients to the database
    Args: 
        coeffs: dictionary of coefficients
        path: path to the coefficients
    '''
    if not stacked:
        folder = f'{path}/{geneName}_upsampled_coeffs'
        if not os.path.exists(folder):
            os.makedirs(folder)
        for key, value in coeffs.items():
            df = pd.DataFrame(value)
            df.to_csv(f'{folder}/{geneName}_{key}.csv', index=False)
    # if they're stacked already just add to the folder
    else:
        df = pd.DataFrame(coeffs)
        df.to_csv(f'{path}/{geneName}_upsampled_coeffs.csv', index=False)


# %%
# use the one in main
def processAllCoeffs(path, S, D, scaleFactor, bounds):
    for folder in os.listdir(path):
        geneName = folder.split("_")[0]
        print(geneName)

        #coeffs = readWaveletCoeffs(f"{path}/{folder}")
        #resampled = resampleCoeffs(coeffs, S, D, scaleFactor, bounds)
        #exportCoeffs(resampled, f"{path}/{folder}", geneName)