import numpy as np
import cv2

import os
import pyprojroot
from pyprojroot import here
root = pyprojroot.find_root(pyprojroot.has_dir(".git"))

import matplotlib.pyplot as plt

def num_foci(mask: np.ndarray):
    return np.unique(mask).shape[0] - 2

def foci_counts(masks):
    return [num_foci(mask) for mask in masks]

def main():
    # test visualization
    # requires masks to be saved as text files in the directory data/masks
    fnames = [
        "P242_73665006707-A6_009_007_proj",
        "P244_73665165741-C7_027_003_proj",
        "P242_73665006707-A8_040_007_proj",
        "P243_73665098237-G4_016_026_proj",
        "P242_73665006707-A6_003_013_proj",
        "P242_73665006707-C8_012_008_proj"
    ]

    for fname in fnames:
        ipath = os.path.join(root, "data", "raw", f"{fname}.tif")
        im = cv2.imread(ipath, cv2.IMREAD_ANYDEPTH)

        mpath = os.path.join(root, "data", "masks", f"{fname}.txt")
        mask = np.loadtxt(mpath)
        nfoci = num_foci(mask)
        f, axarr = plt.subplots(1,2)
        axarr[0].imshow(im)
        axarr[1].imshow(mask)
        plt.title(f"# foci: {nfoci}")
        plt.show()

if __name__ == "__main__":
    main()