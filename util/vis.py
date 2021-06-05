import os
from matplotlib import pyplot as plt
import torch
import h5py
def vis_data(pd, pdfs, pd_name, pdfs_name, slice, output_dir):

    os.makedirs(output_dir, exist_ok=True)

    title = pd_name.split('/')[-1] + ' and ' + pdfs_name.split('/')[-1] + ':' + str(slice)

    fig, axs = plt.subplots(1, 2, figsize=(20, 20))
    axs[0].imshow(pd, cmap='gray')
    axs[1].imshow(pdfs, cmap='gray')
    plt.suptitle(title)
    figname = pd_name.split('/')[-1] + '_' + pdfs_name.split('/')[-1] + '_' + str(slice) + '.png'

    figpath = os.path.join(output_dir, figname)

    plt.savefig(figpath)
    plt.close('all')

def vis_img(img, fname, ftype ,output_dir):
    os.makedirs(output_dir, exist_ok=True)
    plt.figure()
    plt.imshow(img, cmap='gray')
    figname = fname + '_' + ftype + '.png'
    figpath = os.path.join(output_dir, figname)
    plt.savefig(figpath)

def save_reconstructions(reconstructions, out_dir):
    """
    Save reconstruction images.

    This function writes to h5 files that are appropriate for submission to the
    leaderboard.

    Args:
        reconstructions (dict[str, np.array]): A dictionary mapping input
            filenames to corresponding reconstructions (of shape num_slices x
            height x width).
        out_dir (pathlib.Path): Path to the output directory where the
            reconstructions should be saved.
    """
    os.makedirs(str(out_dir), exist_ok=True)
    print(out_dir)
    for fname in reconstructions.keys():
        f_output = torch.stack([v for _, v in reconstructions[fname].items()])

        basename = fname.split('/')[-1]
        with h5py.File(str(out_dir) + '/' + str(basename) + '.hdf5', "w") as f:
            print(fname)
            f.create_dataset("reconstruction", data=f_output.cpu())