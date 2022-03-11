import os
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import ndimage
from copy import copy, deepcopy
import pandas as pd
import SimpleITK as sitk
import pylidc as pl

from utils.get_ndls_from_inpain import (normalizePatches, load_resampled_image,
                                        load_inpainted_images, get_smaller_versions,
                                        get_the_block_from_the_resampled_image,
                                        load_itk_image, compare_lidc_coords,
                                        qualitative_evaluation_image2, denormalizePatches,
                                        put_inpainted_in_resampled_image,
                                        get_cubes_for_gan)
#%%
path_seg = '/content/drive/MyDrive/Datasets/LUNA16/seg-lungs-LUNA16/'
path_anotations = '/content/drive/MyDrive/Datasets/LUNA16/annotations/annotations.csv'
path_inpainting_done = '/content/drive/MyDrive/Datasets/LUNA16/inpainted_nodules/'
path_last = f'{path_inpainting_done}last/'
path_orig = f'{path_inpainting_done}orig/'
path_masks = f'{path_inpainting_done}masks/'
path_mask_nodules = f'{path_inpainting_done}masks nodules/'
path_boxes_coords = f'{path_inpainting_done[:-7]}box_coords/'
path_dest = '/content/drive/MyDrive/Datasets/LUNA16/inpainted_cubes_for_synthesis/'
# path_qualitative_evaluation = f'{path_dest}versions2D/qualitative assessment/'
ff = os.listdir(path_last)
ff = np.sort(ff)
print(len(ff))
df = pd.read_csv(path_anotations)
df.shape
#%%
name_previous = 'no.one'
for idf, f in enumerate(ff):
#     iii=0
#     if idf < iii: continue

    last, orig, masks, mask_nodules, inserted, ndls_centers_block = load_inpainted_images(f, path_last, path_orig, path_masks, path_mask_nodules)
    inserted = normalizePatches(inserted)
    lungs, ndls_centers, mask = load_resampled_image(f)
    lungs_small, mask_small, z_min, y_min, x_min = get_smaller_versions(lungs, mask)
    boxes_coords = np.load(f'{path_boxes_coords}{f[:-3]}npy') # get the coords of the block (obtained in inpaint_luna.py)
    block_from_resampled, mask_from_resampled = get_the_block_from_the_resampled_image(ndls_centers, z_min, y_min, x_min, lungs_small, mask_small, boxes_coords, mask_nodules)
    lungs_inserted, z_small_plus_boxfound, y_small_plus_boxfound, x_small_plus_boxfound = put_inpainted_in_resampled_image(lungs, inserted, z_min, y_min, x_min, boxes_coords, last)
    cubes_gan_orig, cubes_gan_inpain, mask_ndls, zzs, yys, xxs, zz_cube_resampled, yy_cube_resampled, xx_cube_resampled, paddings = get_cubes_for_gan(mask_nodules, lungs, lungs_inserted, mask, z_small_plus_boxfound, y_small_plus_boxfound, x_small_plus_boxfound)
    
    # the ifs clauses before and after the next *for loop* are to avoid repeating cubes:
    # a nodule might be captured by more than one block but we can identify this by 
    # checking the coordinates with respect to resampled (in file name)
    name_scan = f.split('_')[0]
    _, _, spacing = load_itk_image(f'{path_seg}{name_scan}.mhd') # used in compare_lidc_coords
    if name_scan != name_previous:  
        name_coords_in_scan_all = []
        name_previous = name_scan
    for idx_cube, (cube_orig, cube_inpain, mask_ndl, z,y,x) in enumerate(zip(cubes_gan_orig, cubes_gan_inpain, mask_ndls, zzs, yys, xxs)):
        name_coords_in_scan = f'z{z}y{y}x{x}'
        if name_coords_in_scan in name_coords_in_scan_all: continue # nodule contained in another cube
        else: name_coords_in_scan_all.append(name_coords_in_scan)
        
        coords_obtained = [z,y,x]
        df_lidc_ndl = compare_lidc_coords(name_scan, coords_obtained, spacing, df)
        
        f_name = f'{f[:-4]}_{name_coords_in_scan}.raw'
        assert np.shape(cube_orig) == (64, 64, 64) and np.shape(cube_inpain) == (64, 64, 64) and np.shape(mask_ndl) == (64, 64, 64)
        # qualitative evaluation
        # qualitative_evaluation_image2(cube_orig[31], cube_inpain[31], mask_ndl[31], path_qualitative_evaluation, f_name)
        # denormalize to save in more convenient format 
        cube_orig = denormalizePatches(cube_orig)
        cube_inpain = denormalizePatches(cube_inpain)
        print(idf, f_name)
        # save files
        df_lidc_ndl.to_csv(f'{path_dest}lidc info/{f_name[:-3]}csv', index=False)
        cube_orig.tofile(f'{path_dest}original/{f_name}')
        cube_inpain.tofile(f'{path_dest}inpainted inserted/{f_name}')
        mask_ndl.tofile(f'{path_dest}mask/{f_name}')