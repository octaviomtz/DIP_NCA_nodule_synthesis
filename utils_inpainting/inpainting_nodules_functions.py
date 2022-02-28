'''
Author: Octavio Martinez Manzanera
Helper functions to produce inpainting lung nodules
This functions are tested in the notebooks:
inpainting nodules of all patients v17 (from v15) - 2D conv on 3D vol
inpainting nodules of all patients v15 (from v11)
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import ndimage
from tqdm import tqdm
from numba import jit
import scipy.sparse as sparse
from scipy.ndimage.morphology import binary_erosion, binary_dilation
from matplotlib import rcParams
from itertools import groupby, count
from copy import copy
import logging
logging.basicConfig(filename='example.log',level=logging.DEBUG)

def set_all_rcParams(true_or_false):
    rcParams['ytick.left']=true_or_false
    rcParams['xtick.bottom']=true_or_false
    rcParams['ytick.labelleft'] = true_or_false
    rcParams['xtick.labelbottom'] = true_or_false

def plot_for_gif(image_to_save,num_iter, i):
    fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':[8, 1]}, figsize=(14,10))
    ax[0].imshow(image_to_save, cmap='viridis')
    ax[0].axis('off')
    ax[1].axvline(x=.5, c='k')
    ax[1].scatter(.5, i, c='k')
    ax[1].set_ylim([num_iter, 0])
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    # ax[1].xticks([], [])
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    plt.subplots_adjust(wspace=.04, hspace=0)
    plt.savefig(f'{path_img_dest}images before gifs/iter {i:5d}.jpeg',
                bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)
    
def save_original(image_to_save, id_name, name_extension, error_final=-1):
    name_extension = str(name_extension)
    fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':[8, 1]}, figsize=(14,10))
    ax[0].imshow(image_to_save, cmap='viridis')
    ax[0].axis('off')
    ax[1].axvline(x=.5, c='k')
    ax[1].set_ylim([num_iter, 0])
    ax[1].yaxis.tick_right()
    ax[1].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False) 
    ax[1].spines["top"].set_visible(False)
    ax[1].spines["bottom"].set_visible(False)
    ax[1].spines["left"].set_visible(False)
    ax[1].spines["right"].set_visible(False)
    plt.subplots_adjust(wspace=.04, hspace=0)
    if error_final==-1: # for original
        fig.savefig(f'{path_img_dest}gifs/dip {id_name} {name_extension}.jpeg',
                    bbox_inches = 'tight',pad_inches = 0)
    else:
        fig.savefig(f'{path_img_dest}gifs/dip {id_name} {name_extension} {error_final:05d}.jpeg',
                    bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)

def plot_3d(image, threshold=-300, alpha=.70, fig_size=10):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    
    verts, faces, x,y = measure.marching_cubes_lewiner(p, threshold)

    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def plot_3d_2(image, image2, threshold=-300, threshold2=-300, alpha=.70, fig_size=10):
    
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(2,1,0)
    verts, faces, x,y = measure.marching_cubes_lewiner(p, threshold)
    
    p2 = image2.transpose(2,1,0)
    verts2, faces2, x2,y2 = measure.marching_cubes_lewiner(p2, threshold2)

    fig = plt.figure(figsize=(fig_size*2, fig_size))
    ax = fig.add_subplot(121, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=alpha)
    face_color = [0.45, 0.45, 0.75]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)
    
    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    
    ax = fig.add_subplot(122, projection='3d')
    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts2[faces2], alpha=alpha)
    face_color = [0.75, 0.25, 0.25]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])

    plt.show()

def read_slices(new_name):
    """Read slices of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    idname = new_name.split('_')[0]
    file_lung = np.load(f'{path_data}lungs/{new_name}')
    file_mask = np.load(f'{path_data}masks/{new_name}')
    file_nodule = np.load(f'{path_data}nodule to focus on/{new_name}')
    file_outside  = np.load(f'{path_data}outside lungs mask/{new_name}')
    lungs_slice = file_lung.f.arr_0
    mask_slice = file_mask.f.arr_0
    nodule = file_nodule.f.arr_0
    outside_lungs = file_outside.f.arr_0
    return lungs_slice, mask_slice, nodule, outside_lungs

def make3d_from_sparse(path):
    slices_all = os.listdir(path)
    slices_all = np.sort(slices_all)
    for idx, i in tqdm(enumerate(slices_all), desc='reading slices', total=len(slices_all)):
        sparse_matrix = sparse.load_npz(f'{path}{i}')
        array2d = np.asarray(sparse_matrix.todense())
        if idx == 0: 
            scan3d = array2d
            continue
        scan3d = np.dstack([scan3d,array2d])
    return scan3d

def make_images_right_size3D(lungs_slice, mask_slice, mask_maxvol_and_lungs_small, outside_lungs):
    """Make the images the right size 
    The encoder-decoder has five blocks (the one initially evalluated), 
    therefore, each side has to be divisible by a factor of 32 (2^5)"""
    print('formating shape')
    factor = 32
    pad_dim_0 = factor - np.shape(lungs_slice)[0] % factor
    pad_dim_1 = factor - np.shape(lungs_slice)[1] % factor
    pad_dim_2 = factor - np.shape(lungs_slice)[2] % factor

    #mask_slice = 1 - mask_slice

    lungs_slice = np.pad(lungs_slice, ((0,pad_dim_0), (0,pad_dim_1), (0, pad_dim_2)), mode='constant')
    mask_slice = np.pad(mask_slice, ((0,pad_dim_0), (0,pad_dim_1), (0, pad_dim_2)), mode='constant')
    mask_max =  np.pad(mask_maxvol_and_lungs_small, ((0,pad_dim_0), (0,pad_dim_1), (0, pad_dim_2)), mode='constant')
    outside_lungs = np.pad(outside_lungs, ((0,pad_dim_0), (0,pad_dim_1), (0, pad_dim_2)), mode='constant', constant_values=0)

    # Normalize
    lungs_slice = (lungs_slice - np.min(lungs_slice))/(np.max(lungs_slice)-np.min(lungs_slice))
    
    # Add dimensions
    lungs_slice = np.expand_dims(lungs_slice, 0)
    mask_slice = np.expand_dims(mask_slice, 0)
    outside_lungs = np.expand_dims(outside_lungs, 0)
    mask_max = np.expand_dims(mask_max, 0)
    

    img_np = lungs_slice
    img_mask_np = mask_max
    return img_np, img_mask_np, outside_lungs

def read_slices3D(path_data, ii_ids):
    """Read VOLUMES of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    #ii_ids = f'LIDC-IDRI-{idnumber:04d}'
    print(f'reading scan {ii_ids}')
    vol = make3d_from_sparse(f'{path_data}{ii_ids}/scans/')
    mask = make3d_from_sparse(f'{path_data}{ii_ids}/consensus_masks/')
    mask_maxvol = make3d_from_sparse(f'{path_data}{ii_ids}/maxvol_masks/')
    mask_lungs = make3d_from_sparse(f'{path_data}{ii_ids}/lung_masks/')  
    # rearrange axes to slices first
    vol = np.swapaxes(vol,1,2)
    vol = np.swapaxes(vol,0,1)
    mask = np.swapaxes(mask,1,2)
    mask = np.swapaxes(mask,0,1)
    mask_maxvol = np.swapaxes(mask_maxvol,1,2)
    mask_maxvol = np.swapaxes(mask_maxvol,0,1)
    mask_lungs = np.swapaxes(mask_lungs,1,2)
    mask_lungs = np.swapaxes(mask_lungs,0,1)
    # Find the minimum box that contain the lungs 
    min_box = np.where(vol!=0)
    min_box_c = min_box[0]
    min_box_x = min_box[1]
    min_box_y = min_box[2]
    # Apply the minimum box to the vol and masks
    vol_small = vol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_small = mask[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_maxvol_small = mask_maxvol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_lungs_small = mask_lungs[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)] 
    # Get the mask_maxvol_small and the mask_lungs_small together
    mask_maxvol_and_lungs = mask_lungs_small - mask_maxvol_small
    return vol_small, mask_maxvol_small, mask_maxvol_and_lungs, mask_lungs_small

def read_slices3D_v2(path_data, ii_ids):
    """Read VOLUMES of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    #ii_ids = f'LIDC-IDRI-{idnumber:04d}'
    print(f'reading scan {ii_ids}')
    vol = make3d_from_sparse(f'{path_data}{ii_ids}/scans/')
    mask = make3d_from_sparse(f'{path_data}{ii_ids}/consensus_masks/')
    mask_maxvol = make3d_from_sparse(f'{path_data}{ii_ids}/maxvol_masks/')
    mask_lungs = make3d_from_sparse(f'{path_data}{ii_ids}/lung_masks/')  
    # rearrange axes to slices first
    vol = np.swapaxes(vol,1,2)
    vol = np.swapaxes(vol,0,1)
    mask = np.swapaxes(mask,1,2)
    mask = np.swapaxes(mask,0,1)
    mask_maxvol = np.swapaxes(mask_maxvol,1,2)
    mask_maxvol = np.swapaxes(mask_maxvol,0,1)
    mask_lungs = np.swapaxes(mask_lungs,1,2)
    mask_lungs = np.swapaxes(mask_lungs,0,1)
    # Find the minimum box that contain the lungs 
    min_box = np.where(vol!=0)
    min_box_c = min_box[0]
    min_box_x = min_box[1]
    min_box_y = min_box[2]
    # Apply the minimum box to the vol and masks
    vol_small = vol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_small = mask[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_maxvol_small = mask_maxvol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_lungs_small = mask_lungs[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)] 
    # Get the mask_maxvol_small and the mask_lungs_small together
    mask_maxvol_and_lungs = 1- ((1-mask_lungs_small) | mask_maxvol_small)
    mask_lungs_small2 = mask_lungs_small | mask_maxvol_small
    return vol_small, mask_maxvol_small, mask_maxvol_and_lungs, mask_lungs_small2

def read_slices3D_v3(path_data, ii_ids):
    """Read VOLUMES of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    #ii_ids = f'LIDC-IDRI-{idnumber:04d}'
    print(f'reading scan {ii_ids}')
    vol = make3d_from_sparse(f'{path_data}{ii_ids}/scans/')
    mask = make3d_from_sparse(f'{path_data}{ii_ids}/consensus_masks/')
    mask_maxvol = make3d_from_sparse(f'{path_data}{ii_ids}/maxvol_masks/')
    mask_lungs = make3d_from_sparse(f'{path_data}{ii_ids}/lung_masks/')  
    # rearrange axes to slices first
    vol = np.swapaxes(vol,1,2)
    vol = np.swapaxes(vol,0,1)
    mask = np.swapaxes(mask,1,2)
    mask = np.swapaxes(mask,0,1)
    mask_maxvol = np.swapaxes(mask_maxvol,1,2)
    mask_maxvol = np.swapaxes(mask_maxvol,0,1)
    mask_lungs = np.swapaxes(mask_lungs,1,2)
    mask_lungs = np.swapaxes(mask_lungs,0,1)
    # Find the minimum box that contain the lungs 
    min_box = np.where(vol!=0)
    min_box_c = min_box[0]
    min_box_x = min_box[1]
    min_box_y = min_box[2]
    # Apply the minimum box to the vol and masks
    vol_small = vol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_small = mask[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_maxvol_small = mask_maxvol[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)]
    mask_lungs_small = mask_lungs[np.min(min_box_c):np.max(min_box_c),np.min(min_box_x):np.max(min_box_x),np.min(min_box_y):np.max(min_box_y)] 
    # Get the mask_maxvol_small and the mask_lungs_small together
    mask_maxvol_and_lungs = 1- ((1-mask_lungs_small) | mask_maxvol_small)
    mask_lungs_small2 = mask_lungs_small | mask_maxvol_small
    return vol_small, mask_maxvol_small, mask_maxvol_and_lungs, mask_lungs_small2, min_box_c, min_box_x, min_box_y

def erode_and_split_mask(mask_lungs, slice_middle):
    '''We return the center of each lung (from the middle slice). We erode the center slice of the
    lungs mask to have the lungs separated'''
    # Erode mask
    mask_center_slice = mask_lungs[slice_middle,:,:]
    mask_slice_eroded = binary_erosion(mask_center_slice, iterations=12)
    # Rectangle for lung 1
    labeled, nr_objects = ndimage.label(mask_slice_eroded) 
    blank = np.zeros_like(labeled)
    x, y = np.where(labeled==2)
    blank[x,y] = 2
    ymed_1 = np.median(y); xmed_1 = np.median(x)
    #coords_i_1, coords_j_1, coords_k_1 = find_best_vol(mask_lungs, xmed_1, ymed_1, side1, side2, side3)
    # Rectangle for lung 2
    labeled, nr_objects = ndimage.label(mask_slice_eroded) 
    blank = np.zeros_like(labeled)
    x, y = np.where(labeled==1)
    blank[x,y] = 1
    ymed_2 = np.median(y); xmed_2 = np.median(x)
    # Make sure that number 1 is the lung in the left
    if ymed_1 > ymed_2:
        ymed_temp = ymed_1
        xmed_temp = xmed_1
        ymed_1 = ymed_2
        xmed_1 = xmed_2
        ymed_2 = ymed_temp
        xmed_2 = xmed_temp
    return xmed_1, ymed_1, xmed_2, ymed_2

def box_coords_contain_masks_right_size(coord_max_sideX, coord_min_sideX):
    
    # Max and min coord of nodules for each axis
    z_max_sideX = np.max(np.array(coord_max_sideX)[:,0])
    z_min_sideX = np.min(np.array(coord_min_sideX)[:,0])
    x_max_sideX = np.max(np.array(coord_max_sideX)[:,1])
    x_min_sideX = np.min(np.array(coord_min_sideX)[:,1])
    y_max_sideX = np.max(np.array(coord_max_sideX)[:,2])
    y_min_sideX = np.min(np.array(coord_min_sideX)[:,2])

    # find out the length required to contain all masks per axis
    z_dist_required = z_max_sideX - z_min_sideX
    x_dist_required = x_max_sideX - x_min_sideX
    y_dist_required = y_max_sideX - y_min_sideX
    
    # Fixed distance
    z_dist_adjusted = 96
    x_dist_adjusted = 160
    y_dist_adjusted = 96

    # Add half of the required length to min, and then, get the new max using the required length 
    #add_one_side_z = (factor - z_dist_required % factor)//2
    add_one_side_z = (z_dist_adjusted - z_dist_required)//2
    z_min_sideX  = int(z_min_sideX - add_one_side_z)
    z_min_sideX = np.max([z_min_sideX, 0]) # check it's not smaller than 0
    z_max_sideX_temp = z_min_sideX + z_dist_adjusted
    if z_max_sideX_temp > np.shape(mask_lungs_small)[0]: # if max is outside the scan
        z_min_sideX = z_max_sideX - z_dist_adjusted
    else:
        z_max_sideX = z_max_sideX_temp
    
    #add_one_side_x = (factor - x_dist_required % factor)//2
    add_one_side_x = (x_dist_adjusted - x_dist_required)//2
    x_min_sideX  = int(x_min_sideX - add_one_side_x)
    x_min_sideX = np.max([x_min_sideX, 0])
    x_max_sideX_temp = x_min_sideX + x_dist_adjusted
    if x_max_sideX_temp > np.shape(mask_lungs_small)[1]: # if max is outside the scan
        x_min_sideX = x_max_sideX - x_dist_adjusted
    else:
        x_max_sideX = x_max_sideX_temp

    #add_one_side_y = (factor - y_dist_required % factor)//2
    add_one_side_y = (y_dist_adjusted - y_dist_required)//2
    y_min_sideX  = int(y_min_sideX - add_one_side_y)
    y_min_sideX = np.max([y_min_sideX, 0])
    y_max_sideX_temp = y_min_sideX + y_dist_adjusted
    if y_max_sideX_temp > np.shape(mask_lungs_small)[2]: # if max is outside the scan
        y_min_sideX = y_max_sideX - y_dist_adjusted
    else:
        y_max_sideX = y_max_sideX_temp

    return z_min_sideX, z_max_sideX, x_min_sideX, x_max_sideX, y_min_sideX, y_max_sideX

def box_coords_contain_masks_right_size_search(coord_max_sideX, coord_min_sideX, side, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, dist1 = 96, dist2 = 160, dist3 = 96):
    # new shapes are defined with distances on each axes
    length1 = dist1//2
    length2 = dist2//2
    length3 = dist3//2
    # limits of the nodules masks
    if len(coord_max_sideX) > 0:
        coord_ = [i[0] for i in coord_max_sideX]
        z_max_sideX = np.max(coord_)
        coord_ = [i[0] for i in coord_min_sideX]
        z_min_sideX = np.min(coord_)
        coord_ = [i[1] for i in coord_max_sideX]
        x_max_sideX = np.max(coord_)
        coord_ = [i[1] for i in coord_min_sideX]
        x_min_sideX = np.min(coord_)
        coord_ = [i[2] for i in coord_max_sideX]
        y_max_sideX = np.max(coord_)
        coord_ = [i[2] for i in coord_min_sideX]
        y_min_sideX = np.min(coord_)

    # find if the coords are closer to the center of the right or left lung
    if side == 1:
        xmed_X = xmed_1
        ymed_X = ymed_1
    elif side == 2:
        xmed_X = xmed_2
        ymed_X = ymed_2
    box_found = False  
    
    # find where the vol_cut get more info voxels
    max_sum = 0
    for i in range(30):
        ii = i * 4 - 58
        for j in range(19):
            jj = j * 3 - 27
            for k in range(19):
                kk = k * 4 - 36
            
                
                #if ii == 0 and jj == 0 and kk == 0: pdb.set_trace()
                #zmin = int(slice_middle-length1+ii); zmax = int(slice_middle+length1+ii)
                #xmin = int(xmed_X-length2+jj); xmax = int(xmed_X+length2+jj)
                #ymin = int(ymed_X-length3+kk); ymax = int(ymed_X+length3+kk)
                
                # limits of the current box
                zmin = int(slice_middle-(dist1//2)+ii)
                zmin = np.max([zmin, 0]); zmax = int(zmin + dist1)
                
                xmin = int(xmed_X-(dist2//2)+jj); 
                xmin = np.max([xmin, 0]); xmax = int(xmin + dist2)
                
                ymin = int(ymed_X-(dist3//2)+kk); 
                ymin = np.max([ymin, 0]); ymax = int(ymin + dist3)
            
                #max_cut = mask_maxvol_small[zmin:zmax, xmin:xmax, zmin:zmax]
            
                #if there is a nodule
                if len(coord_max_sideX) > 0:
                    #if the current box contains the masks
                    if zmin < z_min_sideX and zmax > z_max_sideX and xmin < x_min_sideX and xmax > x_max_sideX and ymin < y_min_sideX and ymax > y_max_sideX:
                        #if the current box is inside the scan (small) limits
                        if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                            vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                            # the box contains as many info voxels as possible
                            this_sum = np.sum(vol_cut)
                            if this_sum > max_sum:
                                max_sum = this_sum
                                coords_i = ii; coords_j=jj; coords_k=kk
                                box_found = True
                                z_min_sideX_found = zmin
                                z_max_sideX_found = zmax
                                x_min_sideX_found = xmin
                                x_max_sideX_found = xmax
                                y_min_sideX_found = ymin                        
                                y_max_sideX_found = ymax 
                else: # if it doesn't contain the masks just look for max value of info voxels
                    vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                    #if the current box is inside the scan (small) limits
                    if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                        # the box contains as many info voxels as possible
                        this_sum = np.sum(vol_cut)
                        if this_sum >= max_sum:
                            max_sum = this_sum
                            coords_i = ii; coords_j=jj; coords_k=kk
                            box_found = True
                            z_min_sideX_found = zmin
                            z_max_sideX_found = zmax
                            x_min_sideX_found = xmin
                            x_max_sideX_found = xmax
                            y_min_sideX_found = ymin                        
                            y_max_sideX_found = ymax 
            #print(int(zmin < z_min_sideX) , int(zmax > z_max_sideX) , int(xmin < x_min_sideX) , int(xmax > x_max_sideX) , int(ymin < y_min_sideX) , int(ymax > y_max_sideX))
    if box_found == True:
        return z_min_sideX_found, z_max_sideX_found, x_min_sideX_found, x_max_sideX_found, y_min_sideX_found, y_max_sideX_found
    else:
        return -1, -1, -1, -1, -1, -1

def box_coords_contain_masks_right_size_search_v2(coord_max_sideX, coord_min_sideX, side, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, version, dist1 = 96, dist2 = 160, dist3 = 96):
    # new shapes are defined with distances on each axes
    length1 = dist1//2
    length2 = dist2//2
    length3 = dist3//2
    # limits of the nodules masks
    if version ==1:
        if len(coord_max_sideX) > 0:
            coord_ = [i[0] for i in coord_max_sideX]
            z_max_sideX = np.max(coord_)
            coord_ = [i[0] for i in coord_min_sideX]
            z_min_sideX = np.min(coord_)
            coord_ = [i[1] for i in coord_max_sideX]
            x_max_sideX = np.max(coord_)
            coord_ = [i[1] for i in coord_min_sideX]
            x_min_sideX = np.min(coord_)
            coord_ = [i[2] for i in coord_max_sideX]
            y_max_sideX = np.max(coord_)
            coord_ = [i[2] for i in coord_min_sideX]
            y_min_sideX = np.min(coord_)
    elif version == 2:
        z_max_sideX = coord_max_sideX[0]
        z_min_sideX = coord_min_sideX[0]
        x_max_sideX = coord_max_sideX[1]
        x_min_sideX = coord_min_sideX[1]
        y_max_sideX = coord_max_sideX[2]
        y_min_sideX = coord_min_sideX[2]
    
    # find if the coords are closer to the center of the right or left lung
    if side == 1:
        xmed_X = xmed_1
        ymed_X = ymed_1
    elif side == 2:
        xmed_X = xmed_2
        ymed_X = ymed_2
    box_found = False  

    # find where the vol_cut get more info voxels
    max_sum = 0
    for i in range(24*2):
        ii = i * 10 - (54*2)
        for j in range(24*2):
            jj = j * 10 - (54*2)
            for k in range(24*2):
                kk = k * 10 - (54*2)
                
                # limits of the current box
                zmin = int(slice_middle-(dist1//2)+ii)
                zmin = np.max([zmin, 0]); zmax = int(zmin + dist1)
                if zmax >= np.shape(mask_lungs_small)[0]: continue
                
                xmin = int(xmed_X-(dist2//2)+jj); 
                xmin = np.max([xmin, 0]); xmax = int(xmin + dist2)
                if xmax >= np.shape(mask_lungs_small)[1]: continue
                
                ymin = int(ymed_X-(dist3//2)+kk); 
                ymin = np.max([ymin, 0]); ymax = int(ymin + dist3)
                if ymax >= np.shape(mask_lungs_small)[2]: continue
                
                #print(zmin, zmax, xmin, xmax, ymin, ymax)
            
                #max_cut = mask_maxvol_small[zmin:zmax, xmin:xmax, zmin:zmax]
            
                #if there is a nodule
                if len(coord_max_sideX) > 0:
                    #if the current box contains the masks
                    if zmin < z_min_sideX and zmax > z_max_sideX and xmin < x_min_sideX and xmax > x_max_sideX and ymin < y_min_sideX and ymax > y_max_sideX:
                        #print('FOUND: current box contains the masks')
                        #if the current box is inside the scan (small) limits
                        if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                            vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                            # the box contains as many info voxels as possible
                            this_sum = np.sum(vol_cut)
                            if this_sum > max_sum:
                                max_sum = this_sum
                                coords_i = ii; coords_j=jj; coords_k=kk
                                box_found = True
                                z_min_sideX_found = zmin
                                z_max_sideX_found = zmax
                                x_min_sideX_found = xmin
                                x_max_sideX_found = xmax
                                y_min_sideX_found = ymin                        
                                y_max_sideX_found = ymax 
                else: # if it doesn't contain the masks just look for max value of info voxels
                    vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                    #if the current box is inside the scan (small) limits
                    if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                        # the box contains as many info voxels as possible
                        this_sum = np.sum(vol_cut)
                        if this_sum >= max_sum:
                            max_sum = this_sum
                            coords_i = ii; coords_j=jj; coords_k=kk
                            box_found = True
                            z_min_sideX_found = zmin
                            z_max_sideX_found = zmax
                            x_min_sideX_found = xmin
                            x_max_sideX_found = xmax
                            y_min_sideX_found = ymin                        
                            y_max_sideX_found = ymax 
            #print(int(zmin < z_min_sideX) , int(zmax > z_max_sideX) , int(xmin < x_min_sideX) , int(xmax > x_max_sideX) , int(ymin < y_min_sideX) , int(ymax > y_max_sideX))
    if box_found == True:
        return z_min_sideX_found, z_max_sideX_found, x_min_sideX_found, x_max_sideX_found, y_min_sideX_found, y_max_sideX_found
    else:
        return -1, -1, -1, -1, -1, -1

def nodule_right_or_left_lung(mask_maxvol_smallX, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2):
    '''For each nodule determine if its closer to the right or left cube center.
    Then return, for each side, the min and max coordianates of each nodule'''
    labeled, nr_objects = ndimage.label(mask_maxvol_smallX) 
    masks_ndl = [np.where(labeled==i) for i in range(nr_objects+1) if i>0]   # masks for individual nodules masks
    masks_ndl_centers = [np.median(i,1) for i in masks_ndl] # centers individual nodules masks
    masks_ndl_max = [np.max(i,1) for i in masks_ndl] # centers individual nodules masks
    masks_ndl_min = [np.min(i,1) for i in masks_ndl] # centers individual nodules masks
    
    # For each nodule determine if its closer to the right or left cube center
    nodule_in_side = np.ones((len(masks_ndl_centers)))
    center1 = (slice_middle,xmed_1,ymed_1)
    center2 = (slice_middle,xmed_2,ymed_2)
    for idx, i in enumerate(masks_ndl_centers):
        dist1 = np.linalg.norm(center1-i)
        dist2 = np.linalg.norm(center2-i)
        if dist2 < dist1:
            nodule_in_side[idx]=2
            
    coord_center_side1_X, coord_max_side1_X, coord_min_side1_X = [], [], []
    coord_center_side2_X, coord_max_side2_X, coord_min_side2_X = [], [], []
    for coords, coords_max, coords_min, side in zip(masks_ndl_centers, masks_ndl_max, masks_ndl_min, nodule_in_side):
        if side ==1:
            coord_center_side1_X.append(coords)
            coord_max_side1_X.append(coords_max)
            coord_min_side1_X.append(coords_min)
        if side == 2:
            coord_center_side2_X.append(coords)
            coord_max_side2_X.append(coords_max)
            coord_min_side2_X.append(coords_min)
    return coord_min_side1_X, coord_max_side1_X, coord_min_side2_X, coord_max_side2_X

# https://stackoverflow.com/questions/49515085/python-garbage-collection-sometimes-not-working-in-jupyter-notebook
def my_reset(*varnames):
    """
    varnames are what you want to keep
    """
    globals_ = globals()
    to_save = {v: globals_[v] for v in varnames}
    to_save['my_reset'] = my_reset  # lets keep this function by default
    del globals_
    get_ipython().magic("reset -f")
    globals().update(to_save)

def get_block_if_ndl(block1, block2, block1_mask, block2_mask, block1_mask_maxvol_and_lungs, block2_mask_maxvol_and_lungs, block1_mask_lungs, block2_mask_lungs):
    '''If there are nodules in both blocks put them in a list to be processed on be one in a loop.
    Also include their mask and their names for identification'''
    blocks_ndl, blocks_ndl_mask, blocks_ndl_lungs_mask, block_mask_maxvol_and_lungs = [], [], [], []
    blocks_ndl_names = []
    z,x,y = np.where(block1_mask==1)
    if len(z)>1:
        slice1 = int(np.median(z))
        blocks_ndl.append(block1)
        blocks_ndl_mask.append(block1_mask)
        blocks_ndl_lungs_mask.append(block1_mask_lungs)
        block_mask_maxvol_and_lungs.append(block1_mask_maxvol_and_lungs)
        blocks_ndl_names.append('block1')
    else:
        slice1 = np.shape(block1_mask)[0]//2

    z,x,y = np.where(block2_mask==1)
    if len(z)>1:
        slice2 = int(np.median(z))
        blocks_ndl.append(block2)
        blocks_ndl_mask.append(block2_mask)
        blocks_ndl_lungs_mask.append(block2_mask_lungs)
        block_mask_maxvol_and_lungs.append(block2_mask_maxvol_and_lungs)
        blocks_ndl_names.append('block2')
    else:
        slice2 = np.shape(block2_mask)[0]//2
    return blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs_mask, blocks_ndl_names, slice1, slice2

def get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zmin, c_zmax, c_xmin, c_xmax, c_ymin, c_ymax):
    '''Based on the limits found in "box_coords_contain_masks_right_size_search"
    get the block for the lung scan, the block for the mask with the maxvol segmentation,
    the block for the mask with the maxvol segmentation and the lungs and the block
    for the mask with the mask of the lungs'''
    block = vol_small[c_zmin:c_zmax, c_xmin:c_xmax, c_ymin:c_ymax]
    block_mask = mask_maxvol_small[c_zmin:c_zmax, c_xmin:c_xmax, c_ymin:c_ymax]
    block_mask_maxvol_and_lungs = mask_maxvol_and_lungs_small[c_zmin:c_zmax, c_xmin:c_xmax, c_ymin:c_ymax]
    block_mask_lungs = mask_lungs_small[c_zmin:c_zmax, c_xmin:c_xmax, c_ymin:c_ymax]
    return block, block_mask, block_mask_maxvol_and_lungs, block_mask_lungs

def one_cycle_lr(mse_error_lr, epochs, LRs, main_peak_ratio=0.8):
    '''Find the LR values to apply one cycle. The function returns the learning rates and momentum
    values to be use in one_cycle policy'''
    loss_going_down = np.where(np.diff(mse_error_lr) < -5e-4) # indices that go down (negative diff)
    loss_going_down = list(loss_going_down[0] + 1) # for each pair of indices with neg diff take the 2nd one and convert to list
    c = count()
    val = max((list(g) for _, g in groupby(loss_going_down, lambda x: x-next(c))), key=len) # longest sequence of negative diff
    val = list(val)
    slope_diff = np.diff(mse_error_lr[val])
    largest_diff = np.where(slope_diff == np.min(slope_diff))[0]
    LR = LRs[val[largest_diff[0]]]
    # One_cycle learning rate values. They form a main positive peak followed by a small tail at the end
    epochs_last = epochs * (1-main_peak_ratio)
    epochs_adj = epochs - epochs_last
    lr_min = LRs[val[0]]
    lr_max = LRs[val[-1]]
    one_cycle_up = np.arange(lr_min, lr_max, (lr_max-lr_min)/(epochs_adj/2))
    one_cycle_down = np.arange(lr_max, lr_min, -(lr_max-lr_min)/(epochs_adj/2))
    one_cycle = np.append(one_cycle_up, one_cycle_down)
    one_cycle_last = np.linspace(one_cycle[-1], one_cycle[-1]*.01, epochs_last)
    one_cycle = np.append(one_cycle, one_cycle_last)
    # One_cycle momentum values. They form a main negative peak followed by a small tail at the end
    one_cycle_m_up = np.linspace(0.95, 0.80, len(one_cycle_up))
    one_cycle_m_down = np.linspace(0.80, 0.95, len(one_cycle_down))
    one_cycle_m = np.append(one_cycle_m_up, one_cycle_m_down)
    one_cycle_m_last = np.ones_like(one_cycle_last) * 0.95
    one_cycle_m = np.append(one_cycle_m, one_cycle_m_last)
    return one_cycle, one_cycle_m, val[largest_diff[0]], val[0], val[-1]

def one_cycle_constant_range(epochs, main_peak_ratio=0.8, lr_min=0.0010699, lr_max=0.03047718):
    # One_cycle learning rate values. They form a main positive peak followed by a small tail at the end
    epochs_last = epochs * (1-main_peak_ratio)
    epochs_adj = epochs - epochs_last
    one_cycle_up = np.arange(lr_min, lr_max, (lr_max-lr_min)/(epochs_adj/2))
    one_cycle_down = np.arange(lr_max, lr_min, -(lr_max-lr_min)/(epochs_adj/2))
    one_cycle = np.append(one_cycle_up, one_cycle_down)
    one_cycle_last = np.linspace(one_cycle[-1], one_cycle[-1]*.01, epochs_last)
    one_cycle = np.append(one_cycle, one_cycle_last)
    # One_cycle momentum values. They form a main negative peak followed by a small tail at the end
    one_cycle_m_up = np.linspace(0.95, 0.80, len(one_cycle_up))
    one_cycle_m_down = np.linspace(0.80, 0.95, len(one_cycle_down))
    one_cycle_m = np.append(one_cycle_m_up, one_cycle_m_down)
    one_cycle_m_last = np.ones_like(one_cycle_last) * 0.95
    one_cycle_m = np.append(one_cycle_m, one_cycle_m_last)
    return one_cycle, one_cycle_m

def merged_if_close3(cluster_names, c_min, c_max, BLOCK_SIZE  = [96,160,96]):
    '''
    If two nodules are close remove their names from the original
    Variable with all names and merge them in a new variable.
    By comparing all nodules twice (A != B and B != A) we make sure that
    the min of A is close to the max of B and that
    the min of B is close to the max of A. '''
    clus_names_ = copy(cluster_names)
    sets_pairs_close_nodules = []
    merged = []
    for name_x, i in zip(clus_names_, c_min):
        for name_j, j in zip(clus_names_, c_max):
            if name_j!=name_x:
                comparison = np.sum(np.abs(np.asarray(i)-np.asarray(j))>BLOCK_SIZE)
                # If nodules are closer than threshold
                if comparison == 0:
                    set_pass1 = [name_x,name_j]
                    set_pass2 = [name_j,name_x]
                    sets_pairs_close_nodules.append(set_pass1)
                    # if both maxs are close to the others mins
                    if set_pass2 in sets_pairs_close_nodules:
                        merged=list([name_x, name_j])
                        clus_names_.remove(name_x)
                        clus_names_.remove(name_j)
                        return(clus_names_, merged)
    return(clus_names_, merged)

def new_min_max2(clusters_names, clus_names, mer, coords_min, coords_max):
    '''Merge the names of the nodules that were close (mer). Find the coords
    of those nodules and get their min and max. Merge them and remove their
    old single versions'''
    coords_min_ = copy(coords_min)
    coords_max_ = copy(coords_max)
    clusters_names_ = copy(clusters_names)
    clus_names_ = copy(clus_names)
    # Merge the cluster names
    new_cluster = [str(i) for i in mer]
    new_cluster = ''.join(new_cluster)
    clus_names_.insert(0,new_cluster)
    # Indices of merged nodules
    clus_index1 = clusters_names.index(mer[0])
    clus_index2 = clusters_names.index(mer[1])
    # Get the new min and max
    new_min = []
    new_max = []
    for i, j in zip(coords_min_[clus_index1], coords_min_[clus_index2]):
        new_min.append(np.min([i,j]))
    for i, j in zip(coords_max_[clus_index1], coords_max_[clus_index2]):
        new_max.append(np.max([i,j]))
    # Remove the merged coords from their coords_min and coords_max
    remove_first = np.max([clus_index1, clus_index2])
    remove_second = np.min([clus_index1, clus_index2])
    del coords_min_[remove_first]
    del coords_min_[remove_second]
    del coords_max_[remove_first]
    del coords_max_[remove_second]
    # Add the new min and max
    coords_min_.insert(0,new_min) 
    coords_max_.insert(0,new_max)
    return coords_min_, coords_max_, clus_names_
    print(new_min, new_max)

def merge_nodules_in_clusters(coords_min, coords_max, block_number):
    '''Merge nodules that are close together iteratively'''
    finished = False
    first_iteration = True
    while finished == False:
        if first_iteration == True:
            # Assign original names
            clusters_names = list(np.arange(0,len(coords_min)))
            clusters_names = [str(i) for  i in clusters_names]
        else:
            clusters_names = copy(clus_names)

        clus_names, mer = merged_if_close3(clusters_names, coords_min, coords_max)
        if len(mer)>1:
            coords_min, coords_max, clus_names = new_min_max2(clusters_names, clus_names, mer, coords_min, coords_max)
            mer = []
        else:
            finished = True
        first_iteration = False
        #f'{path_img_dest}
    return clus_names, coords_min, coords_max

def box_coords_contain_masks_right_size_search_v3(coord_max_sideX, coord_min_sideX, side, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, version, dist1 = 96, dist2 = 160, dist3 = 96):
    # new shapes are defined with distances on each axes
    length1 = dist1//2
    length2 = dist2//2
    length3 = dist3//2
    # limits of the nodules masks
    if version ==1:
        if len(coord_max_sideX) > 0:
            coord_ = [i[0] for i in coord_max_sideX]
            z_max_sideX = np.max(coord_)
            coord_ = [i[0] for i in coord_min_sideX]
            z_min_sideX = np.min(coord_)
            coord_ = [i[1] for i in coord_max_sideX]
            x_max_sideX = np.max(coord_)
            coord_ = [i[1] for i in coord_min_sideX]
            x_min_sideX = np.min(coord_)
            coord_ = [i[2] for i in coord_max_sideX]
            y_max_sideX = np.max(coord_)
            coord_ = [i[2] for i in coord_min_sideX]
            y_min_sideX = np.min(coord_)
    elif version == 2:
        z_max_sideX = coord_max_sideX[0]
        z_min_sideX = coord_min_sideX[0]
        x_max_sideX = coord_max_sideX[1]
        x_min_sideX = coord_min_sideX[1]
        y_max_sideX = coord_max_sideX[2]
        y_min_sideX = coord_min_sideX[2]

    # find if the coords are closer to the center of the right or left lung
    if side == 1:
        xmed_X = xmed_1
        ymed_X = ymed_1
    elif side == 2:
        xmed_X = xmed_2
        ymed_X = ymed_2
    box_found = False  
    
    # find where the vol_cut get more info voxels
    max_sum = 0
    for i in range(80*2):
        ii = i * 2 - (160)
        for j in range(80*2):
            jj = j * 2 - (160)
            for k in range(80*2):
                kk = k * 2 - (160)
                
                # limits of the current box
                zmin = int(slice_middle-(dist1//2)+ii)
                #zmin = np.max([zmin, 0]); 
                zmax = int(zmin + dist1)
                if zmin < 0: continue
                if zmax > np.shape(mask_lungs_small)[0]: continue
                try:
                    xmin = int(xmed_X-(dist2//2)+jj); 
                except ValueError:
                    logging.info(xmed_X, dist2, jj)
                #xmin = np.max([xmin, 0]); 
                xmax = int(xmin + dist2)
                if xmin < 0: continue
                if xmax > np.shape(mask_lungs_small)[1]: continue
                
                ymin = int(ymed_X-(dist3//2)+kk); 
                #ymin = np.max([ymin, 0]); 
                ymax = int(ymin + dist3)
                if ymin < 0: continue
                if ymax > np.shape(mask_lungs_small)[2]: continue
                
                #print(zmin, zmax, xmin, xmax, ymin, ymax)
            
                #max_cut = mask_maxvol_small[zmin:zmax, xmin:xmax, zmin:zmax]
                
                #if there is a nodule
                if len(coord_max_sideX) > 0:
                    #if the current box contains the masks
                    if zmin < z_min_sideX and zmax > z_max_sideX and xmin < x_min_sideX and xmax > x_max_sideX and ymin < y_min_sideX and ymax > y_max_sideX:
                        #print('FOUND: current box contains the masks')
                        #if the current box is inside the scan (small) limits
                        #if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                        vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                        # the box contains as many info voxels as possible
                        this_sum = np.sum(vol_cut)
                        if this_sum > max_sum:
                            max_sum = this_sum
                            coords_i = ii; coords_j=jj; coords_k=kk
                            box_found = True
                            z_min_sideX_found = zmin
                            z_max_sideX_found = zmax
                            x_min_sideX_found = xmin
                            x_max_sideX_found = xmax
                            y_min_sideX_found = ymin                        
                            y_max_sideX_found = ymax 
                else: # if it doesn't contain the masks just look for max value of info voxels
                    vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                    #if the current box is inside the scan (small) limits
                    #if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                    # the box contains as many info voxels as possible
                    this_sum = np.sum(vol_cut)
                    if this_sum >= max_sum:
                        max_sum = this_sum
                        coords_i = ii; coords_j=jj; coords_k=kk
                        box_found = True
                        z_min_sideX_found = zmin
                        z_max_sideX_found = zmax
                        x_min_sideX_found = xmin
                        x_max_sideX_found = xmax
                        y_min_sideX_found = ymin                        
                        y_max_sideX_found = ymax 
            #print(int(zmin < z_min_sideX) , int(zmax > z_max_sideX) , int(xmin < x_min_sideX) , int(xmax > x_max_sideX) , int(ymin < y_min_sideX) , int(ymax > y_max_sideX))
    print(f'box_found = {box_found}')
    if box_found == True:
        return z_min_sideX_found, z_max_sideX_found, x_min_sideX_found, x_max_sideX_found, y_min_sideX_found, y_max_sideX_found
    else:
        return -1, -1, -1, -1, -1, -1

def get_box_coords_per_block(coord_min_sideX, coord_max_sideX, block_number, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, normalize=True):
    '''
    Combines in one function the new (v17v2) "box_coords_contain_masks_right_size_search_v2"
    "get_four_blocks" and "normalize_clip_and_mask" and applied them iteratively to the
    list of merged nodules.
    '''
    coords_minX = [list(i) for i in coord_min_sideX]
    coords_maxX = [list(i) for i in coord_max_sideX]
    clus_namesX, coords_minX, coords_maxX = merge_nodules_in_clusters(coords_minX, coords_maxX, block_number)
    blockX_list, blockX_mask_list, blockX_mask_maxvol_and_lungs_list, blockX_mask_lungs_list, box_coords_list = [], [], [], [], []
    for merged_idx, (merged_names, merged_min, merged_max) in enumerate(zip(clus_namesX, coords_minX, coords_maxX)):
        try:
            c_zminX, c_zmaxX, c_xminX, c_xmaxX, c_yminX, c_ymaxX = box_coords_contain_masks_right_size_search_v3_numba(merged_max, merged_min, block_number,  slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, 2)
            # c_zminX, c_zmaxX, c_xminX, c_xmaxX, c_yminX, c_ymaxX = box_coords_contain_masks_right_size_search_v3(merged_max, merged_min, block_number,  slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, 2)
        except ValueError: continue
        blockX, blockX_mask, blockX_mask_maxvol_and_lungs, blockX_mask_lungs = get_four_blocks(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, c_zminX, c_zmaxX, c_xminX, c_xmaxX, c_yminX, c_ymaxX)
        if normalize:
            blockX = normalize_clip_and_mask(blockX, blockX_mask_lungs)
        blockX_list.append(blockX)
        blockX_mask_list.append(blockX_mask)
        blockX_mask_maxvol_and_lungs_list.append(blockX_mask_maxvol_and_lungs)
        blockX_mask_lungs_list.append(blockX_mask_lungs)
        box_coords_list.append([c_zminX, c_zmaxX, c_xminX, c_xmaxX, c_yminX, c_ymaxX])
    return blockX_list, blockX_mask_list, blockX_mask_maxvol_and_lungs_list, blockX_mask_lungs_list, clus_namesX, box_coords_list

def get_block_if_ndl_list(block1_list, block2_list, block1_mask_list, block2_mask_list, block1_mask_maxvol_and_lungs_list, block2_mask_maxvol_and_lungs_list, block1_mask_lungs_list, block2_mask_lungs_list, clus_names1, clus_names2, box_coords1, box_coords2):
    '''If there are nodules in both blocks put them in a list to be processed on be one in a loop.
    Also include their mask and their names for identification'''
    blocks_ndl_, blocks_ndl_mask_, blocks_ndl_lungs_mask_, block_mask_maxvol_and_lungs_, blocks_ndl_names_, box_coords_ = [], [], [], [], [], []
    
    if len(block1_list)>0:
        for idx, (b_, b_mask_, b_mask_maxvol_and_lungs_, b_mask_lungs_, clus_names_, coords_) in enumerate(zip(block1_list, block1_mask_list, block1_mask_maxvol_and_lungs_list, block1_mask_lungs_list, clus_names1, box_coords1)):
            blocks_ndl_.append(b_)
            blocks_ndl_mask_.append(b_mask_)
            blocks_ndl_lungs_mask_.append(b_mask_lungs_)
            block_mask_maxvol_and_lungs_.append(b_mask_maxvol_and_lungs_)
            box_coords_.append(coords_)
            blocks_ndl_names_.append(f'block1_{clus_names_}')
    
    if len(block2_list)>0:
        for idx, (b_, b_mask_, b_mask_maxvol_and_lungs_, b_mask_lungs_, clus_names_, coords_) in enumerate(zip(block2_list, block2_mask_list, block2_mask_maxvol_and_lungs_list, block2_mask_lungs_list, clus_names2, box_coords2)):
            blocks_ndl_.append(b_)
            blocks_ndl_mask_.append(b_mask_)
            blocks_ndl_lungs_mask_.append(b_mask_lungs_)
            block_mask_maxvol_and_lungs_.append(b_mask_maxvol_and_lungs_)
            box_coords_.append(coords_)
            blocks_ndl_names_.append(f'block2_{clus_names_}')

    return blocks_ndl_, blocks_ndl_mask_, block_mask_maxvol_and_lungs_, blocks_ndl_lungs_mask_, blocks_ndl_names_, box_coords_

def normalize_clip_and_mask(block_, block_mask_):
    block_ = (block_ - (-1018.0))/(1171.0-(-1018.0)) 
    block_ = np.clip(block_,0,1)
    block_ = block_*block_mask_
    return block_

def pad_if_vol_too_small(vol_small_, mask_maxvol_small_, mask_maxvol_and_lungs_small_, mask_lungs_small_, box_shape=[97,161,97]):
    '''padd the volumes if they are smaller than the box shape.
    This can happen specially for the slices because this happens before we resample across slices'''
    vol_is_too_small = np.asarray(box_shape) - np.shape(vol_small_)
    padd_0, padd_1, padd_2 = np.maximum(vol_is_too_small, 0)
    vol_small_ = np.pad(vol_small_, ((0,padd_0), (0,padd_1), (0, padd_2)), mode='constant', constant_values=0)
    mask_maxvol_small_ = np.pad(mask_maxvol_small_, ((0,padd_0), (0,padd_1), (0, padd_2)), mode='constant', constant_values=0)
    mask_maxvol_and_lungs_small_ = np.pad(mask_maxvol_and_lungs_small_, ((0,padd_0), (0,padd_1), (0, padd_2)), mode='constant', constant_values=0)
    mask_lungs_small_ = np.pad(mask_lungs_small_, ((0,padd_0), (0,padd_1), (0, padd_2)), mode='constant', constant_values=0)
    return vol_small_, mask_maxvol_small_, mask_maxvol_and_lungs_small_, mask_lungs_small_

def box_coords_contain_masks_right_size_search_v3_numba(coord_max_sideX, coord_min_sideX, side, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, mask_lungs_small, version, dist1 = 96, dist2 = 160, dist3 = 96):
    # new shapes are defined with distances on each axes
    length1 = dist1//2
    length2 = dist2//2
    length3 = dist3//2
    # limits of the nodules masks
    if version ==1:
        if len(coord_max_sideX) > 0:
            coord_ = [i[0] for i in coord_max_sideX]
            z_max_sideX = np.max(coord_)
            coord_ = [i[0] for i in coord_min_sideX]
            z_min_sideX = np.min(coord_)
            coord_ = [i[1] for i in coord_max_sideX]
            x_max_sideX = np.max(coord_)
            coord_ = [i[1] for i in coord_min_sideX]
            x_min_sideX = np.min(coord_)
            coord_ = [i[2] for i in coord_max_sideX]
            y_max_sideX = np.max(coord_)
            coord_ = [i[2] for i in coord_min_sideX]
            y_min_sideX = np.min(coord_)
    elif version == 2:
        z_max_sideX = coord_max_sideX[0]
        z_min_sideX = coord_min_sideX[0]
        x_max_sideX = coord_max_sideX[1]
        x_min_sideX = coord_min_sideX[1]
        y_max_sideX = coord_max_sideX[2]
        y_min_sideX = coord_min_sideX[2]

    # find if the coords are closer to the center of the right or left lung
    if side == 1:
        xmed_X = xmed_1
        ymed_X = ymed_1
    elif side == 2:
        xmed_X = xmed_2
        ymed_X = ymed_2
      
    
    # find where the vol_cut get more info voxels
    coords_sideX = z_min_sideX, z_max_sideX, x_min_sideX, x_max_sideX, y_min_sideX, y_max_sideX
    z_min_sideX_found, z_max_sideX_found, x_min_sideX_found, x_max_sideX_found, y_min_sideX_found, y_max_sideX_found, box_found = nodules_in_box_loop(slice_middle, dist1, dist2, dist3, xmed_X, ymed_X, mask_lungs_small, coord_max_sideX, coords_sideX)

            #print(int(zmin < z_min_sideX) , int(zmax > z_max_sideX) , int(xmin < x_min_sideX) , int(xmax > x_max_sideX) , int(ymin < y_min_sideX) , int(ymax > y_max_sideX))
    print(f'box_found = {box_found}')
    if box_found == True:
        return z_min_sideX_found, z_max_sideX_found, x_min_sideX_found, x_max_sideX_found, y_min_sideX_found, y_max_sideX_found
    else:
        return -1, -1, -1, -1, -1, -1

@jit(nopython=True)
def nodules_in_box_loop(slice_middle, dist1, dist2, dist3, xmed_X, ymed_X, mask_lungs_small, coord_max_sideX, coords_sideX):
    max_sum = 0
    box_found = False
    print('using numba')
    z_min_sideX, z_max_sideX, x_min_sideX, x_max_sideX, y_min_sideX, y_max_sideX = coords_sideX
    for i in range(80*2):
        ii = i * 2 - (160)
        for j in range(80*2):
            jj = j * 2 - (160)
            for k in range(80*2):
                kk = k * 2 - (160)
                
                # limits of the current box
                zmin = int(slice_middle-(dist1//2)+ii)
                #zmin = np.max([zmin, 0]); 
                zmax = int(zmin + dist1)
                if zmin < 0: continue
                if zmax > np.shape(mask_lungs_small)[0]: continue
                
                xmin = int(xmed_X-(dist2//2)+jj); 
                #xmin = np.max([xmin, 0]); 
                xmax = int(xmin + dist2)
                if xmin < 0: continue
                if xmax > np.shape(mask_lungs_small)[1]: continue
                
                ymin = int(ymed_X-(dist3//2)+kk); 
                #ymin = np.max([ymin, 0]); 
                ymax = int(ymin + dist3)
                if ymin < 0: continue
                if ymax > np.shape(mask_lungs_small)[2]: continue
                
                #print(zmin, zmax, xmin, xmax, ymin, ymax)
            
                #max_cut = mask_maxvol_small[zmin:zmax, xmin:xmax, zmin:zmax]
                
                #if there is a nodule
                if len(coord_max_sideX) > 0:
                    #if the current box contains the masks
                    if zmin < z_min_sideX and zmax > z_max_sideX and xmin < x_min_sideX and xmax > x_max_sideX and ymin < y_min_sideX and ymax > y_max_sideX:
                        #print('FOUND: current box contains the masks')
                        #if the current box is inside the scan (small) limits
                        #if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                        vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                        # the box contains as many info voxels as possible
                        this_sum = np.sum(vol_cut)
                        if this_sum > max_sum:
                            max_sum = this_sum
                            coords_i = ii; coords_j=jj; coords_k=kk
                            box_found = True
                            z_min_sideX_found = zmin
                            z_max_sideX_found = zmax
                            x_min_sideX_found = xmin
                            x_max_sideX_found = xmax
                            y_min_sideX_found = ymin                        
                            y_max_sideX_found = ymax 
                else: # if it doesn't contain the masks just look for max value of info voxels
                    vol_cut=mask_lungs_small[zmin:zmax,xmin:xmax,ymin:ymax]
                    #if the current box is inside the scan (small) limits
                    #if zmin >= 0 and zmax <= np.shape(mask_lungs_small)[0] and xmin >= 0 and xmax <= np.shape(mask_lungs_small)[1] and ymin >= 0 and ymax <= np.shape(mask_lungs_small)[2]:
                    # the box contains as many info voxels as possible
                    this_sum = np.sum(vol_cut)
                    if this_sum >= max_sum:
                        max_sum = this_sum
                        coords_i = ii; coords_j=jj; coords_k=kk
                        box_found = True
                        z_min_sideX_found = zmin
                        z_max_sideX_found = zmax
                        x_min_sideX_found = xmin
                        x_max_sideX_found = xmax
                        y_min_sideX_found = ymin                        
                        y_max_sideX_found = ymax 
    return z_min_sideX_found, z_max_sideX_found, x_min_sideX_found, x_max_sideX_found, y_min_sideX_found, y_max_sideX_found, box_found
