import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy
from skimage.morphology import ball, dilation
from time import time

import SimpleITK as sitk
import pylidc as pl 
from pylidc.utils import consensus

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from utils.preprocess import (load_itk_image, resample_scan_sitk, 
                            normalizePatches, worldToVoxelCoord)

@hydra.main(config_path="config", config_name="config_preprocess.yaml")
def main(cfg: DictConfig):
    # HYDRA
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    path_orig = hydra.utils.get_original_cwd()
    # PATHS
    path_seg = f'{cfg.path_drive}seg-lungs-LUNA16/'
    data_dir = f'{path_orig}/{cfg.data_dir}'
    # out_path = f'{path_orig}/{cfg.out_path}'
    # ff = os.listdir(path_scans)

    # df = pd.read_csv(f'{path_drive}annotations.csv')
    df = pd.read_csv(f'{cfg.path_drive}annotations/annotations.csv')
    annotations_df = pd.read_csv(f'{cfg.path_drive}annotations/annotations.csv')
    cands_df  = pd.read_csv(f'{cfg.path_drive}candidates_V2.csv')
    print(df.shape, cands_df.shape, annotations_df.shape)
    # df.head()

    for each_subset in range(10):
        if each_subset not in cfg.SUBSET: continue
        path_scans = f'{cfg.path_drive}/subsets/subset{each_subset}/'
        out_path_subset = cfg.out_path + f'subset{each_subset}/'
        log.info(f'preprocessing subset{each_subset}')
        log.info(f'output in: {out_path_subset}')
        if not os.path.exists(out_path_subset): 
            os.makedirs(out_path_subset)

        subset_files = os.listdir(path_scans)
        subset_series_ids = np.unique(np.asarray([subset_files[ll][:-4:] for ll in range(len(subset_files))]))
        subset_series_ids = np.sort(subset_series_ids)

        out_df = pd.DataFrame(columns=['seriesuid','coordX','coordY','coordZ','class','diameter_mm','filename'])
        columns_plus_new_coords = np.append(df.columns.values,['coordZ_cands_class1_resampled', 'coordY_cands_class1_resampled', 'coordX_cands_class1_resampled'])
        columns_plus_new_coords_class1 = np.append(cands_df.columns.values,['coordX_resamp', 'coordY_resamp', 'coordZ_resamp'])

        for series_id_idx, series_id in tqdm(enumerate(subset_series_ids), total=len(subset_series_ids)):
            # skip already proprocessed files
            if series_id_idx < cfg.SKIP_IDX: continue
            log.info(f'preprocessing: {series_id_idx}, {series_id}')
            start = time()

            df_lidc_new = pd.DataFrame(columns=columns_plus_new_coords)
            df_cands_class1 = pd.DataFrame(columns=columns_plus_new_coords_class1)
            image_file = path_scans + series_id + '.mhd'
            numpyImage, numpyOrigin, numpySpacing = load_itk_image(image_file)
            curr_cands = cands_df.loc[cands_df['seriesuid'] == series_id].reset_index(drop=True)

            # resample original image
            new_spacing = [1,1,1]
            numpyImage_shape = ((np.shape(numpyImage) * numpySpacing) / np.asarray(new_spacing)).astype(int)
            numpyImage_resampled = resample_scan_sitk(numpyImage, numpySpacing, numpyImage_shape, new_spacing=new_spacing)
            
            # get the correspponding lung segmentation
            path_segment_lungs = f'{path_seg}{series_id}.mhd'
            segment_lungs, seglung_orig, seglung_spacing = load_itk_image(path_segment_lungs)
            assert (numpyOrigin - seglung_orig < .1).all()
            assert (numpySpacing - seglung_spacing < .1).all()
            segment_lungs_resampled = resample_scan_sitk(segment_lungs, numpySpacing, numpyImage_shape, new_spacing=new_spacing)

            # normalize
            numpyImage_normalized = normalizePatches(numpyImage_resampled)

            # save the segmented lungs
            numpyImage_segmented = numpyImage_normalized * (segment_lungs_resampled>0)
            if not os.path.exists(f'{out_path_subset}{series_id}/lungs_segmented'): os.makedirs(f'{out_path_subset}{series_id}/lungs_segmented')
            np.savez_compressed(f'{out_path_subset}{series_id}/lungs_segmented/lungs_segmented.npz',numpyImage_segmented)

            # go through all candidates that are in this image
            # sort to make sure we have all the trues (for prototyping only)
            curr_cands = curr_cands.sort_values('class',ascending=False).reset_index(drop=True)

            # Added in v2
            one_segmentation_consensus = np.zeros_like(numpyImage)
            one_segmentation_maxvol = np.zeros_like(numpyImage)
            labelledNods = np.zeros_like(numpyImage)

            # query the LIDC images HERE WE JUST USE THE FIRST ONE!!
            idx_scan = 0 
            scan = pl.query(pl.Scan).filter(pl.Scan.series_instance_uid == series_id)[idx_scan] 
            nods = scan.cluster_annotations() # get the annotations for all nodules in this scan
            #print(np.shape(nods))

            #Get all the nodules (class==1)
            curr_cands_class1 = curr_cands.loc[curr_cands['class']==1]
            
            for idx_row, curr_cand in curr_cands_class1.iterrows():
                # print(curr_cand)
                # first need to find the corresponding column in the annotations csv (assuming its the closest 
                # nodule to  the current candidate)
                # extract the annotations for the scan id of our current candidate
                annotations_scan_df = annotations_df.loc[annotations_df['seriesuid'] == curr_cand['seriesuid']]

                # Get the right coordinates        
                worldCoord = [curr_cand['coordZ'], curr_cand['coordY'], curr_cand['coordX']]
                voxelCoord = worldToVoxelCoord(worldCoord, numpyOrigin, numpySpacing)
                voxelCoord_resampled = ((voxelCoord * numpySpacing) / np.asarray(new_spacing)).astype(int)

                # Create new DF with all curr_cands['class']==1
                curr_cand['coordZ_resamp'] = voxelCoord_resampled[0]
                curr_cand['coordY_resamp'] = voxelCoord_resampled[1]
                curr_cand['coordX_resamp'] = voxelCoord_resampled[2]
                df_cands_class1 = df_cands_class1.append(curr_cand)

                df_nodule = df.loc[df['seriesuid'] == curr_cand['seriesuid']]

                # SAVE DATAFRAMES (the pylidc DF from the nodules in LUNA)
                threshold_coord_found  = 4
                # seriesuid might include several nodules, if its coords are close to the coords in annotations then save
                df_series_all_nodules = df.loc[df['seriesuid']==series_id]
                df_series_number_nodules = np.unique(df_series_all_nodules['cluster_id'].values)

                for idxx, i_number_nodule in enumerate(df_series_number_nodules):
                    df_series_one_nodule = df_series_all_nodules.loc[df_series_all_nodules['cluster_id']==i_number_nodule]
                    lidc_coordZ = np.mean(df_series_one_nodule['lidc_coordZ'].values)
                    lidc_coordY = np.mean(df_series_one_nodule['lidc_coordY'].values)
                    lidc_coordX = np.mean(df_series_one_nodule['lidc_coordX'].values)
                        # print(f'lidc_coords = {lidc_coordZ, lidc_coordX, lidc_coordY}')
                        # print(f'voxel_coords = {voxelCoord, voxelCoord_resampled}')
                        # print(np.sum(np.abs(np.asarray([lidc_coordZ, lidc_coordX, lidc_coordY]) - voxelCoord)))
                    # WARNING: now the comparison is done Z-Z, X-Y, Y-X instead of Z-Z, Y-Y, X-X
                    if np.sum(np.abs(np.asarray([lidc_coordZ, lidc_coordX, lidc_coordY]) - voxelCoord)) < threshold_coord_found:
                            # print(f'save')
                        df_series_one_nodule_save = df_series_one_nodule
                        df_series_one_nodule_save['coordZ_cands_class1_resampled'] = voxelCoord_resampled[0]
                        df_series_one_nodule_save['coordY_cands_class1_resampled'] = voxelCoord_resampled[1]
                        df_series_one_nodule_save['coordX_cands_class1_resampled'] = voxelCoord_resampled[2]
                        df_lidc_new = df_lidc_new.append(df_series_one_nodule_save)

            if os.path.isfile(f'{data_dir}dataframes/df_cands_class1.csv'):
                df_cands_class1.to_csv(f'{data_dir}dataframes/df_cands_class1.csv', index=False, mode='a', header=False)
                df_lidc_new.to_csv(f'{data_dir}dataframes/df_lidc_new.csv', index=False, mode='a', header=False)

            else:
                df_cands_class1.to_csv(f'{data_dir}dataframes/df_cands_class1.csv', index=False, mode='w', header=True)
                df_lidc_new.to_csv(f'{data_dir}dataframes/df_lidc_new.csv', index=False, mode='w', header=True)

            # SAVE SEGMENTATIONS    TO DO!!!! ADD FOLDER NAMES FOR CASES WHEN VARIOUS SEGMENTATIONS ARE SAVED
            for idx_anns, anns in enumerate(nods): 
                # if idx_anns==1:break # WARNING !!!!!!!
                #print(idx_anns)
                cmask,cbbox,masks = consensus(anns, clevel=0.5, pad=[(0,0), (0,0), (0,0)])

                # we want to save the consensus AND the mask of all segmented voxels in all annotations
                one_mask_consensus = cmask
                one_mask_maxvol = np.zeros_like(cmask)
                for mask in masks:
                    one_mask_maxvol = (one_mask_maxvol > 0) | (mask > 0)    

                # pylidc loads in a different order to our custom 3D dicom reader, so need to swap dims
                one_mask_consensus = np.swapaxes(one_mask_consensus,1,2);one_mask_consensus = np.swapaxes(one_mask_consensus,0,1)
                one_mask_maxvol = np.swapaxes(one_mask_maxvol,1,2);one_mask_maxvol = np.swapaxes(one_mask_maxvol,0,1)

                # Dilate the mask
                one_mask_maxvol = dilation(one_mask_maxvol)

                # fill the consensus bounding box with the mask to get a nodule segmentation in original image space (presumably the cbbox is big enough for all the individual masks)
                one_segmentation_consensus[cbbox[2].start:cbbox[2].stop,cbbox[0].start:cbbox[0].stop,cbbox[1].start:cbbox[1].stop] = one_mask_consensus
                one_segmentation_maxvol[cbbox[2].start:cbbox[2].stop,cbbox[0].start:cbbox[0].stop,cbbox[1].start:cbbox[1].stop] = one_mask_maxvol
                labelledNods[cbbox[2].start:cbbox[2].stop,cbbox[0].start:cbbox[0].stop,cbbox[1].start:cbbox[1].stop] = one_mask_maxvol * (idx_anns + 1) # label each nodule with its 'cluster_id'

            # one_mask_maxvol.shape

            one_segmentation_consensus_resampled = resample_scan_sitk(one_segmentation_consensus, numpySpacing, numpyImage_shape, new_spacing, sitk.sitkNearestNeighbor)
            one_segmentation_maxvol_resampled = resample_scan_sitk(one_segmentation_maxvol, numpySpacing, numpyImage_shape, new_spacing, sitk.sitkNearestNeighbor)
            labelledNods_resampled = resample_scan_sitk(labelledNods, numpySpacing, numpyImage_shape, new_spacing, sitk.sitkNearestNeighbor)

            if not os.path.exists(f'{out_path_subset}{series_id}/consensus_masks'): os.makedirs(f'{out_path_subset}{series_id}/consensus_masks')
            if not os.path.exists(f'{out_path_subset}{series_id}/maxvol_masks'): os.makedirs(f'{out_path_subset}{series_id}/maxvol_masks')
            if not os.path.exists(f'{out_path_subset}{series_id}/cluster_id_images'): os.makedirs(f'{out_path_subset}{series_id}/cluster_id_images')

            for i_sparse, (one_seg_consen, one_seg_max, labelNods) in enumerate(zip(one_segmentation_consensus_resampled, one_segmentation_maxvol_resampled, labelledNods_resampled)):
                sparse_matrix_one_segmentation_consensus = scipy.sparse.csc_matrix(one_seg_consen)
                sparse_matrix_one_segmentation_maxvol = scipy.sparse.csc_matrix(one_seg_max)
                sparse_matrix_labelledNods = scipy.sparse.csc_matrix(labelNods)

                scipy.sparse.save_npz(f'{out_path_subset}{series_id}/consensus_masks/slice_{i_sparse:04d}.npz', sparse_matrix_one_segmentation_consensus, compressed=True)
                scipy.sparse.save_npz(f'{out_path_subset}{series_id}/maxvol_masks/slice_m_{i_sparse:04d}.npz', sparse_matrix_one_segmentation_maxvol, compressed=True)
                scipy.sparse.save_npz(f'{out_path_subset}{series_id}/cluster_id_images/slice_m_{i_sparse:04d}.npz', sparse_matrix_labelledNods, compressed=True)
            
            stop = time()
            log.info(f'completed in {(stop-start)/60:.2f} mins')

if __name__ == '__main__':
    main()