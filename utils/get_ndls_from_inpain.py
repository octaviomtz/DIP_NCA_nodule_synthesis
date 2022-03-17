import os
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from scipy import ndimage
from copy import copy, deepcopy
import pandas as pd
import SimpleITK as sitk
import pylidc as pl
from scipy.ndimage import binary_dilation
import matplotlib.cm as cm

def median_center(segmentation, label = 1):
    z,y,x = np.where(segmentation == label)
    zz = int(np.median(z))
    yy = int(np.median(y))
    xx = int(np.median(x))
    return zz,yy,xx

def make3d_from_sparse_v2(path):
    slices_all = os.listdir(path)
    slices_all = np.sort(slices_all)
    for idx, i in enumerate(slices_all):
        sparse_matrix = sparse.load_npz(f'{path}{i}')
        array2d = np.asarray(sparse_matrix.todense())
        if idx == 0: 
            scan3d = array2d
            continue
        scan3d = np.dstack([scan3d,array2d])
    scan3d = np.swapaxes(scan3d,2,1)
    scan3d = np.swapaxes(scan3d,1,0)
    return scan3d

def load_itk_image(filename):
    itkimage = sitk.ReadImage(filename)
    numpyImage = sitk.GetArrayFromImage(itkimage)
     
    numpyOrigin = np.array(list(reversed(itkimage.GetOrigin())))
    numpySpacing = np.array(list(reversed(itkimage.GetSpacing())))
     
    return numpyImage, numpyOrigin, numpySpacing

def resample_scan_sitk(image,spacing, original_shape, new_spacing=[1,1,1], resampling_method=sitk.sitkLinear):
    # reorder sizes as sitk expects them
    spacing_sitk = [spacing[1],spacing[2],spacing[0]]
    new_spacing_sitk = [new_spacing[1],new_spacing[2],new_spacing[0]]
    
    # set up the input image as at SITK image
    img = sitk.GetImageFromArray(image)
    img.SetSpacing(spacing_sitk)                
            
    # set up an identity transform to apply
    affine = sitk.AffineTransform(3)
    affine.SetMatrix(np.eye(3,3).ravel())
    affine.SetCenter(img.GetOrigin())
    
    # make the reference image grid, original_shape, with new spacing
    refImg = sitk.GetImageFromArray(np.zeros(original_shape,dtype=image.dtype))
    refImg.SetSpacing(new_spacing_sitk)
    refImg.SetOrigin(img.GetOrigin())
    
    imgNew = sitk.Resample(img, refImg, affine ,resampling_method, 0)
    
    imOut = sitk.GetArrayFromImage(imgNew).copy()
    
    return imOut

# def load_inpainted_images(file_name_):
#     '''load the inpainting results. These are obtained from the blocks (~[96,160,96])'''
#     last = np.load(f'{path_last}{file_name_}')
#     last = np.squeeze(last)
#     orig = np.load(f'{path_orig}{file_name_}')
#     masks = np.load(f'{path_masks}{file_name_[:-1]}z')
#     masks = masks.f.arr_0
#     mask_nodules = np.load(f'{path_mask_nodules}{file_name_[:-1]}z')
#     mask_nodules = mask_nodules.f.arr_0
#     inserted = mask_nodules*last + (-mask_nodules+1)*orig
#     # find nodule centers
#     labeled, nr = ndimage.label(mask_nodules)
#     ndls_centers = []
#     for i in np.arange(1, nr+1):
#         zz, yy, xx = median_center(labeled, i)
#         ndls_centers.append([zz,yy,xx])
#     return last, orig, masks, mask_nodules, inserted, ndls_centers

def load_inpainted_images(file_name_, path_last, path_orig, path_masks, path_mask_nodules):
    '''load the inpainting results. These are obtained from the blocks (~[96,160,96])'''
    last = np.fromfile(f'{path_last}{file_name_}',dtype='int16').astype('float32').reshape((96,160,96))
    #last = np.squeeze(last)
    orig = np.fromfile(f'{path_orig}{file_name_}',dtype='int16').astype('float32').reshape((96,160,96))
    masks = np.load(f'{path_masks}{file_name_[:-3]}npz')
    masks = masks.f.arr_0
    mask_nodules = np.load(f'{path_mask_nodules}{file_name_[:-3]}npz')
    mask_nodules = mask_nodules.f.arr_0
    inserted = mask_nodules*last + (-mask_nodules+1)*orig
    # find nodule centers
    labeled, nr = ndimage.label(mask_nodules)
    ndls_centers = []
    for i in np.arange(1, nr+1):
        zz, yy, xx = median_center(labeled, i)
        ndls_centers.append([zz,yy,xx])
    return last, orig, masks, mask_nodules, inserted, ndls_centers

def load_resampled_image(file_name_, path_data):
    '''load resampled scan and masks of the nodules '''
    file_name_ = file_name_.split('_')[0]
    lungs = np.load(f'{path_data}{file_name_}/lungs_segmented/lungs_segmented.npz')
    lungs = lungs.f.arr_0
    mask = make3d_from_sparse_v2(f'{path_data}{file_name_}/maxvol_masks/')
    labeled, nr = ndimage.label(mask)
    ndls_centers = []
    for i in np.arange(1, nr+1):
        zz, yy, xx = median_center(labeled, i)
        ndls_centers.append([zz,yy,xx])
    return lungs, ndls_centers, mask

def get_smaller_versions(lungs, mask):
    '''Use the mask of the lungs to create the smaller versions like the ones used in preprocessing.
    Return the smaller versions and the coords (min). These are needed to find the nodules on the small versions'''
    z,y,x = np.where(lungs>0)
    x_max = np.max(x); x_min = np.min(x);
    y_max = np.max(y); y_min = np.min(y);
    z_max = np.max(z); z_min = np.min(z);
    lungs_small = lungs[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_small = mask[z_min:z_max, y_min:y_max, x_min:x_max]
    return lungs_small, mask_small, z_min, y_min, x_min

def get_the_block_from_the_resampled_image(ndls_centers, z_min, y_min, x_min, lungs_small, mask_small, boxes_coords, mask_nodules):
    '''make sure that the block is from the right position. We confirm this by comparing the nodules' masks''' 
    ndls_centers_small = [np.asarray(i)- np.asarray([z_min, y_min, x_min]) for i in ndls_centers]
    block_from_resampled = lungs_small[boxes_coords[0]:boxes_coords[1], boxes_coords[2]:boxes_coords[3], boxes_coords[4]:boxes_coords[5]]
    mask_from_resampled = mask_small[boxes_coords[0]:boxes_coords[1], boxes_coords[2]:boxes_coords[3], boxes_coords[4]:boxes_coords[5]]
    # make sure mask from inpainting results == block from resampled
    # we might need to reshape because mask from inpainting results might be smaller (really close to the border)
    sz1,sy1,sx1 = np.shape(mask_nodules)
    sz2,sy2,sx2 = np.shape(mask_from_resampled)
    sz,sy,sx = np.min([sz1, sz2]), np.min([sy1, sy2]), np.min([sx1, sx2])
    assert (mask_from_resampled[0:sz,0:sy,0:sx] == mask_nodules[0:sz,0:sy,0:sx]).all() 
    return block_from_resampled, mask_from_resampled

def put_inpainted_in_resampled_image(lungs, inserted, z_min, y_min, x_min, boxes_coords, last):
    '''We insert the inpainted nodule (from the inserted block) into the resampled image'''
    z_small_plus_boxfound = z_min + boxes_coords[0]
    y_small_plus_boxfound = y_min + boxes_coords[2]
    x_small_plus_boxfound = x_min + boxes_coords[4]
    z_small_plus_boxfound, y_small_plus_boxfound, x_small_plus_boxfound
    shape_block = np.shape(last)
    lungs_inserted = deepcopy(lungs)
    lungs_inserted[z_small_plus_boxfound:z_small_plus_boxfound+shape_block[0], y_small_plus_boxfound:y_small_plus_boxfound+shape_block[1], x_small_plus_boxfound:x_small_plus_boxfound+shape_block[2]] = inserted
    return lungs_inserted, z_small_plus_boxfound, y_small_plus_boxfound, x_small_plus_boxfound

def padd_if_mask_close_to_edge(zz, lungs, cube_half, axis):
    padd_z_low = 0
    padd_z_up = 0
    if (zz - cube_half) < 0:
        padd_z_low = cube_half - zz
    if (zz + cube_half) > np.shape(lungs)[axis]:
        zz = np.shape(lungs)[0]
        padd_z_up = cube_half - (np.shape(lungs)[0] - zz)
    return padd_z_low, padd_z_up

def get_cubes_for_gan(mask_nodules, lungs, lungs_inserted, mask_big, z_small_plus_boxfound, y_small_plus_boxfound, x_small_plus_boxfound):
    '''we get a cube around each nodule of the original and the inserted resampled images.
       To do so, we count the mask_nodules of the block and find their centers. Then we find the corresponding
       coords in the resampled image by adding _small_plus_boxfound to these coords.
       we are suppossed to get a cube of size _cube_size_ but if the coord is too close to the edge we take a 
       smaller portion (until the edge) and pad later with 0s'''
    cube_size = 64
    cube_half = cube_size // 2
    cubes_for_gan_inpain = []
    cubes_for_gan_orig = []
    mask_ndls = []
    zzs,yys,xxs = [], [], []
    labeled, nr = ndimage.label(mask_nodules)
    for i in np.arange(1, nr + 1):
        zz,yy,xx = median_center(labeled, i)
        # find the corresponding coords in the resampled image by adding _small_plus_boxfound to these coords
        zz = zz + z_small_plus_boxfound
        yy = yy + y_small_plus_boxfound
        xx = xx + x_small_plus_boxfound
        # we are suppossed to get a cube of size _cube_size_ but if the coord is too close to the edge we take a 
        # smaller portion and pad later
        padd_z_low, padd_z_up = padd_if_mask_close_to_edge(zz, lungs, cube_half, 0)
        padd_y_low, padd_y_up = padd_if_mask_close_to_edge(yy, lungs, cube_half, 1)
        padd_x_low, padd_x_up = padd_if_mask_close_to_edge(xx, lungs, cube_half, 2)
        # get the cube
        cubes_for_gan_orig_temp = lungs[zz-(cube_half-padd_z_low):zz+(cube_half-padd_z_up),yy-(cube_half-padd_y_low):yy+(cube_half-padd_y_up),xx-(cube_half-padd_x_low):xx+(cube_half-padd_x_up)]
        cubes_for_gan_inpain_temp = lungs_inserted[zz-(cube_half-padd_z_low):zz+(cube_half-padd_z_up),yy-(cube_half-padd_y_low):yy+(cube_half-padd_y_up),xx-(cube_half-padd_x_low):xx+(cube_half-padd_x_up)]
        mask_ndl_from_big_temp = mask_big[zz-(cube_half-padd_z_low):zz+(cube_half-padd_z_up),yy-(cube_half-padd_y_low):yy+(cube_half-padd_y_up),xx-(cube_half-padd_x_low):xx+(cube_half-padd_x_up)]
        
        # pad if needed
        cubes_for_gan_orig_temp = np.pad(cubes_for_gan_orig_temp, ((padd_z_low,padd_z_up), (padd_y_low,padd_y_up), (padd_x_low, padd_x_up)), 'constant', constant_values=0)
        cubes_for_gan_inpain_temp = np.pad(cubes_for_gan_inpain_temp, ((padd_z_low,padd_z_up), (padd_y_low,padd_y_up), (padd_x_low, padd_x_up)), 'constant', constant_values=0)
        mask_ndl_from_big_temp = np.pad(mask_ndl_from_big_temp, ((padd_z_low,padd_z_up), (padd_y_low,padd_y_up), (padd_x_low, padd_x_up)), 'constant', constant_values=0)
        
        # coords of the cube_for_gan with respect of the resampled image
        zz_cube_resampled = zz + padd_z_low - cube_half
        yy_cube_resampled = yy + padd_y_low - cube_half
        xx_cube_resampled = xx + padd_x_low - cube_half
        paddings = [padd_z_low, padd_z_up, padd_y_low, padd_y_up, padd_x_low, padd_x_up]
        
        # Append
        cubes_for_gan_orig.append(cubes_for_gan_orig_temp)
        cubes_for_gan_inpain.append(cubes_for_gan_inpain_temp)
        mask_ndls.append(mask_ndl_from_big_temp)
        zzs.append(zz)
        yys.append(yy)
        xxs.append(xx)
    return cubes_for_gan_orig, cubes_for_gan_inpain, mask_ndls, zzs, yys, xxs, zz_cube_resampled, yy_cube_resampled, xx_cube_resampled, paddings

def normalizePatches(npzarray):
    npzarray = npzarray
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray - minHU) / (maxHU - minHU)
    npzarray[npzarray>1] = 1.
    npzarray[npzarray<0] = 0.
    return npzarray

def denormalizePatches(npzarray):
    
    maxHU = 400.
    minHU = -1000.
 
    npzarray = (npzarray * (maxHU - minHU)) + minHU
    npzarray = (npzarray).astype('int16')
    return npzarray

def qualitative_evaluation_image(orig, inpain, mask, path_save_images, name):
    plt.style.use('dark_background');
    fig, ax = plt.subplots(3,1,figsize=(5,15));
    ax[0].imshow(orig, vmin=0, vmax=1)
    ax[1].imshow(inpain, vmin=0, vmax=1)
    # ax[2].imshow(np.abs(orig-inpain))
    ax[2].imshow(mask)
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()
    fig.savefig(f'{path_save_images}{name[:-4]}.jpg');
    plt.close()
    plt.style.use('default')

def qualitative_evaluation_image2(orig, inpain, mask, path_save_images, name):
    plt.style.use('dark_background');
    fig, ax = plt.subplots(3,2,figsize=(9,19.5));
    ax[0,0].imshow(orig, vmin=0, vmax=1)
    ax[0,1].imshow(orig, vmin=0, vmax=1)
    ax[0,1].imshow(mask, alpha = .3)
    ax[1,0].imshow(inpain, vmin=0, vmax=1)
    ax[1,1].imshow(inpain, vmin=0, vmax=1)
    ax[1,1].imshow(mask, alpha = .3)
    ax[2,0].imshow(mask)
    ax[2,1].imshow(np.abs(orig-inpain))
    for axx in ax.ravel(): axx.axis('off')
    fig.tight_layout()
    fig.savefig(f'{path_save_images}{name[:-4]}.jpg');
    plt.close()
    plt.style.use('default')

def compare_lidc_coords(name_scan, coords_obtained, numpySpacing, df):
    '''compare_lidc_coords_against_coords_obtained_in the script and return a DF 
    with the corresponding LIDC information'''
    df_to_return = pd.DataFrame()
    df_annotations_scan = df.loc[df['seriesuid'] == name_scan]
    nodules_unique = np.unique(df_annotations_scan['cluster_id'].values)
    for i_nodules_unique in nodules_unique:
        df_annotations_scan_ndl = df_annotations_scan.loc[df_annotations_scan['cluster_id'] == i_nodules_unique]
        coordZ = int(np.mean(df_annotations_scan_ndl['lidc_coordZ'].values))
        coordY = int(np.mean(df_annotations_scan_ndl['lidc_coordY'].values))
        coordX = int(np.mean(df_annotations_scan_ndl['lidc_coordX'].values))

        coordZ_resampled = int(coordZ * numpySpacing[0])
        coordY_resampled = int(coordY * numpySpacing[1])
        coordX_resampled = int(coordX * numpySpacing[2])
        
        # Compute differences (for some reason we need to compare X-Y)
        diff_Z = coords_obtained[0] - coordZ_resampled       
        diff_Y = coords_obtained[1] - coordX_resampled       
        diff_X = coords_obtained[2] - coordY_resampled
        
        # If the coords are close
        if diff_Z <= 2 and diff_Y <= 2 and diff_X <= 2:
            df_to_return = df_annotations_scan_ndl
            break
    return df_to_return

def plot_inpainting_quality_control(files, path_inser, path_orig, path_mask, text='', subset=-1, zoom = False, skip=0, save=False, my_cmap = cm.Wistia):
    my_cmap.set_under('k', alpha=0)
     
    fig, ax = plt.subplots(6,12, figsize=(32,16))
    for i in range(35):
        inser = np.fromfile(f'{path_inser}{files[i]}',dtype='int16').astype('float32').reshape((64,64,64))
        orig = np.fromfile(f'{path_orig}{files[i]}',dtype='int16').astype('float32').reshape((64,64,64))
        mask = np.fromfile(f'{path_mask}{files[i]}',dtype='int16').astype('float32').reshape((64,64,64))
        SLICE, SH0, SH1 = np.shape(orig)
        SLICE= SLICE//2 - 1
        mask2 = binary_dilation(mask[SLICE]) - mask[SLICE]
        mask2 = np.ma.masked_where(mask2<.5, mask2)
        
        Z = SLICE//2 if zoom == True else 0
        val_max = np.max(orig[SLICE][Z:SH0-Z,Z:SH1-Z])
        val_min = np.min(orig[SLICE][Z:SH0-Z,Z:SH1-Z])
        ax[(i//12)*2, i%12].imshow(orig[SLICE][Z:SH0-Z,Z:SH1-Z])
        im = ax[(i//12)*2, i%12].imshow(mask2[Z:SH0-Z,Z:SH1-Z], cmap=my_cmap, 
            interpolation='none', clim=[0.9, 1])
        ax[(i//12)*2, i%12].text(4,4,i+skip, c='#FFA500', fontsize=14)
        ax[(i//12)*2+1, i%12].imshow(inser[SLICE][Z:SH0-Z,Z:SH1-Z], vmin=val_min, vmax = val_max)
        
        ax[(i//12)*2, i%12].axis('off')
        ax[(i//12)*2+1, i%12].axis('off')
    
    i+=1
    ax[(i//12)*2, i%12].imshow(np.zeros_like(orig[SLICE][Z:SH0-Z,Z:SH1-Z]))
    ax[(i//12)*2, i%12].text(3,15,text, c='#FFA500', fontsize=14)
    ax[(i//12)*2+1, i%12].imshow(np.zeros_like(orig[SLICE][Z:SH0-Z,Z:SH1-Z]))
    ax[(i//12)*2+1, i%12].text(3,15,f'subset: {subset}', c='#FFA500', fontsize=14)
    ax[(i//12)*2, i%12].axis('off')
    ax[(i//12)*2+1, i%12].axis('off')
    fig.tight_layout()

    if save:
        plt.savefig(f'inpain_qc_subset{subset}_from_{skip}_temp.png')