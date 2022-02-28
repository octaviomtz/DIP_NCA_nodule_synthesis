import os
import numpy as np
import SimpleITK as sitk

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

def make3d_from_sparse_v2(path):
    slices_all = os.listdir(path)
    slices_all = np.sort(slices_all)
    for idx, i in enumerate(slices_all):
#         print(idx, i)
        sparse_matrix = sparse.load_npz(f'{path}{i}')
        array2d = np.asarray(sparse_matrix.todense())
        if idx == 0: 
            scan3d = array2d
            continue
        scan3d = np.dstack([scan3d,array2d])
    scan3d = np.swapaxes(scan3d,2,1)
    scan3d = np.swapaxes(scan3d,1,0)
    return scan3d

def median_center(segmentation, label = 1):
    z,y,x = np.where(segmentation == label)
    zz = int(np.median(z))
    yy = int(np.median(y))
    xx = int(np.median(x))
    return zz,yy,xx

def read_slices3D_v4(path_data_, path_seg_, ii_ids):
    """Read VOLUMES of lung, mask outside lungs and nodule, mask nodule, mask outside"""
    #ii_ids = f'LIDC-IDRI-{idnumber:04d}'
    print(f'reading scan {ii_ids}')
    vol = np.load(f'{path_data_}{ii_ids}/lungs_segmented/lungs_segmented.npz')
    vol = vol.f.arr_0
    mask = make3d_from_sparse(f'{path_data_}{ii_ids}/consensus_masks/')
    mask_maxvol = make3d_from_sparse(f'{path_data_}{ii_ids}/maxvol_masks/')
    # read segmentations from luna
    numpyImage, numpyOrigin, numpySpacing = load_itk_image(f'{path_seg_}{ii_ids}.mhd')
    new_spacing = [1,1,1]
    numpyImage_shape = ((np.shape(numpyImage) * numpySpacing) / np.asarray(new_spacing)).astype(int)
    mask_lungs = resample_scan_sitk(numpyImage, numpySpacing, numpyImage_shape, new_spacing=new_spacing)
    np.shape(mask_lungs)
    ##rearrange axes to slices first
    # vol = np.swapaxes(vol,1,2)
    # vol = np.swapaxes(vol,0,1)
    mask = np.swapaxes(mask,1,2)
    mask = np.swapaxes(mask,0,1)
    mask_maxvol = np.swapaxes(mask_maxvol,1,2)
    mask_maxvol = np.swapaxes(mask_maxvol,0,1)
    mask_consensus = mask
    # use only two labels for mask
    mask_lungs = (mask_lungs>0).astype(int)
    return vol, mask_maxvol, mask_consensus, mask_lungs

def small_versions(vol_, mask_maxvol_, mask_consensus_, mask_lungs_):
    z,y,x = np.where(mask_lungs==1)
    z_min = np.min(z); z_max = np.max(z)
    y_min = np.min(y); y_max = np.max(y)
    x_min = np.min(x); x_max = np.max(x)
    vol_small = vol_[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_maxvol_small = mask_maxvol_[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_consensus_small = mask_consensus_[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_lungs_small = mask_lungs_[z_min:z_max, y_min:y_max, x_min:x_max]
    small_boundaries = [z_min, z_max, y_min, y_max, x_min, x_max]
    return vol_small, mask_maxvol_small, mask_consensus_small, mask_lungs_small, small_boundaries

def denormalizePatches(npzarray):
    maxHU = 400.
    minHU = -1000.
    npzarray = (npzarray * (maxHU - minHU)) + minHU
    npzarray = (npzarray).astype('int16')
    return npzarray