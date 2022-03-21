import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation
import matplotlib.cm as cm
from utils.cube_around_inpainted_ndl import plot_inpainting_quality_control

SUBSET = 1
path_inpainted_parent = '/content/drive/MyDrive/Datasets/LUNA16/inpainted_cubes_for_synthesis/'
path_inser = f'{path_inpainted_parent}subset{SUBSET}/inpainted_inserted/'
path_orig = f'{path_inpainted_parent}subset{SUBSET}/original/'
path_mask = f'{path_inpainted_parent}subset{SUBSET}/mask/'
files = os.listdir(path_inser)
files= np.sort(files)
len(files)

text = 'Arch\n[128]*5\n[128]*5\n[128]*5'
SKIP = 36
plot_inpainting_quality_control(files[SKIP:], path_inser, path_orig, path_mask, text=text, subset=SUBSET, zoom=True, skip=SKIP, save=True)