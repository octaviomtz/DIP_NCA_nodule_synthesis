import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
from copy import copy, deepcopy
from time import time
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import pandas as pd
from skimage import measure, morphology
from itertools import groupby, count
import matplotlib.patches as patches
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from scipy import ndimage
import SimpleITK as sitk
import gc

import torch
import torch.optim
from torch.autograd import Variable

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from models.resnet import ResNet
from models.unet import UNet
from models.skip import skip 
from utils_inpainting.utils_main import load_itk_image, resample_scan_sitk, read_slices3D_v4, small_versions, denormalizePatches
from utils_inpainting.inpainting_nodules_functions import (pad_if_vol_too_small, erode_and_split_mask, 
                                                            nodule_right_or_left_lung, get_box_coords_per_block,
                                                            get_block_if_ndl_list)
from utils_inpainting.common_utils import optimize_ndls, get_noise, get_params, np_to_torch                                                          

dtype = torch.cuda.FloatTensor
torch.cuda.empty_cache()


@hydra.main(config_path="config", config_name="config_inpainting.yaml")
def main(cfg: DictConfig):
    # HYDRA
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    path_orig = hydra.utils.get_original_cwd()
    # PATHS
    path_data = f'{cfg.path_data}subset{cfg.SUBSET}/'
    path_out = f'{cfg.path_img_dest}subset{cfg.SUBSET}/'
    log.info(f'inpainting subset{cfg.SUBSET}')
    log.info(f'output in: {path_out}')
    ids = os.listdir(path_data)
    ids = np.sort(ids)
    if not os.path.exists(f'{path_out}arrays/last/'): os.makedirs(f'{path_out}arrays/last/')
    if not os.path.exists(f'{path_out}arrays/orig/'): os.makedirs(f'{path_out}arrays/orig/')
    if not os.path.exists(f'{path_out}arrays/masks/'): os.makedirs(f'{path_out}arrays/masks/')
    if not os.path.exists(f'{path_out}arrays/masks_nodules/'): os.makedirs(f'{path_out}arrays/masks_nodules/')
    if not os.path.exists(f'{path_out}arrays/masks_lungs/'): os.makedirs(f'{path_out}arrays/masks_lungs/')
    if not os.path.exists(f'{path_out}mse_error_curves_inpainting/'): os.makedirs(f'{path_out}mse_error_curves_inpainting/')
    if not os.path.exists(f'{path_out}inpainting_times/'): os.makedirs(f'{path_out}inpainting_times/')
    if not os.path.exists(f'{path_out}box_coords/'): os.makedirs(f'{path_out}box_coords/')

    global i
    i = 0

    def closure(i):
        # global i
        images_all = []
        if cfg.param_noise:
            for n in [x for x in net.parameters() if len(x.size()) == 4]:
                n = n + n.detach().clone().normal_() * n.std() / 50
        net_input = net_input_saved
        if cfg.reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * cfg.reg_noise_std)
            
        out = net(net_input)
        total_loss = mse(out * mask_var, img_var * mask_var)
        total_loss.backward()
            
        print ('Iteration %05d    Loss %.12f' % (i, total_loss.item()), '\r', end='')
        if  cfg.PLOT:
            out_np = out.detach().cpu().numpy()[0]
            image_to_save = out_np
            # images_all.append(image_to_save)
        i += 1    
        return total_loss, out_np#, images_all

    for idx_name, name in enumerate(ids):

        if idx_name < cfg.SKIP_IDX: continue
        log.info(f'inpainting: {idx_name}, {name}')
        start = time()

        vol, mask_maxvol, mask_consensus, mask_lungs = read_slices3D_v4(path_data, cfg.path_seg, name)
        maxvol0 = np.where(mask_maxvol==1)
        mask_maxvol_and_lungs = deepcopy(mask_lungs)
        mask_maxvol_and_lungs[maxvol0] = 0
        vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, small_boundaries = small_versions(vol, mask_maxvol, mask_maxvol_and_lungs, mask_lungs)
        vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small = pad_if_vol_too_small(vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small)
        slice_middle = np.shape(vol_small)[0] // 2
        labeled, n_items = ndimage.label(mask_maxvol_small)
        xmed_1, ymed_1, xmed_2, ymed_2 = erode_and_split_mask(mask_lungs_small,slice_middle)
        coord_min_side1, coord_max_side1, coord_min_side2, coord_max_side2 = nodule_right_or_left_lung(mask_maxvol_small, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2)
        block1_list, block1_mask_list, block1_mask_maxvol_and_lungs_list, block1_mask_lungs_list, clus_names1, box_coords1 = get_box_coords_per_block(coord_min_side1, coord_max_side1, 1, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, normalize=False)
        block2_list, block2_mask_list, block2_mask_maxvol_and_lungs_list, block2_mask_lungs_list, clus_names2, box_coords2 = get_box_coords_per_block(coord_min_side2, coord_max_side2, 2, slice_middle, xmed_1, ymed_1, xmed_2, ymed_2, vol_small, mask_maxvol_small, mask_maxvol_and_lungs_small, mask_lungs_small, normalize=False)
        blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names, box_coords = get_block_if_ndl_list(block1_list, block2_list, block1_mask_list, block2_mask_list, block1_mask_maxvol_and_lungs_list, block2_mask_maxvol_and_lungs_list, block1_mask_lungs_list, block2_mask_lungs_list, clus_names1, clus_names2, box_coords1, box_coords2)
        del vol_small, mask_maxvol_small, mask_consensus, mask_lungs_small
        gc.collect()

        for (block, block_mask, block_maxvol_and_lungs, block_lungs, block_name, box_coord) in zip(blocks_ndl, blocks_ndl_mask, block_mask_maxvol_and_lungs, blocks_ndl_lungs, blocks_ndl_names, box_coords): 
            torch.cuda.empty_cache()
            print(block_name)
            if np.shape(block)==(0,0,0): 
                print(f'{block_name} skip empty blocks!!!!!!!!!!!!')
                continue
            # Here we dont add batch channels
            img_np = block
            img_mask_np = block_maxvol_and_lungs
            # img_mask_np = block_mask

            # LR FOUND
            LR = 0.0002

            # INPAINTING
            restart_i = 0
            restart = True

            while restart == True:
                print(f'training initialization {restart_i} with LR = {LR:.8f}')
                restart_i += 1

                #lungs_slice, mask_slice, nodule, outside_lungs = read_slices(new_name)
                #img_np, img_mask_np, outside_lungs = make_images_right_size(lungs_slice, mask_slice, nodule, outside_lungs)

                # Loss
                mse = torch.nn.MSELoss().type(dtype)
                img_var = np_to_torch(img_np).type(dtype)
                mask_var = np_to_torch(img_mask_np).type(dtype)

                net = skip(cfg.INPUT_DEPTH, img_np.shape[0], 
                        num_channels_down = [128]*5,
                        num_channels_up   = [128]*5, 
                        num_channels_skip = [128]*5, 
                        upsample_mode='nearest', filter_skip_size=1, filter_size_up=3, filter_size_down=3, 
                        need_sigmoid=True, need_bias=True, pad='zero', act_fun='LeakyReLU').type(dtype)
                net = net.type(dtype)        
                net_input = get_noise(cfg.INPUT_DEPTH, cfg.INPUT, img_np.shape[1:]).type(dtype)

                #path_trained_model = f'{path_img_dest}models/v6_Unet_init_sample_{idx}.pt'
                #torch.save(net.state_dict(), path_trained_model)

                #mse_error = []
                i = 0
                net_input_saved = net_input.detach().clone()
                noise = net_input.detach().clone()
                p = get_params(cfg.OPT_OVER, net, net_input)
                mse_error, images_generated_all, best_iter, restart = optimize_ndls('adam', p, closure, LR, cfg.EPOCHS, cfg.show_every, path_out, restart, annealing=True, lr_finder_flag=False)
                mse_error = np.squeeze(mse_error)
                # mse_error = [i.detach().cpu().numpy() for i in mse_error]
                # mse_error_all.append(mse_error)
                # mse_error_last = mse_error[-1].detach().cpu().numpy()

                if restart_i % 10 == 0: # reduce lr if the network is not learning with the initializations
                    LR /= 1.2
                if restart_i == 30: # if the network cannot be trained continue (might not act on for loop!!)
                    continue
            #print(np.shape(images_generated_all))
            image_last = images_generated_all[-1] * block_lungs
            image_orig = img_np[0] * block_lungs
            best_iter = f'{best_iter:4d}'

            # convert into ints to occupy less space
            image_last = denormalizePatches(image_last)
            img_np = denormalizePatches(img_np)

            image_last.tofile(f'{path_out}arrays/last/{name}_{block_name}.raw')
            img_np.tofile(f'{path_out}arrays/orig/{name}_{block_name}.raw')
            np.savez_compressed(f'{path_out}arrays/masks/{name}_{block_name}',block_maxvol_and_lungs)
            np.savez_compressed(f'{path_out}arrays/masks_nodules/{name}_{block_name}',block_mask)
            np.savez_compressed(f'{path_out}arrays/masks_lungs/{name}_{block_name}',block_lungs)
            np.save(f'{path_out}mse_error_curves_inpainting/{name}_{block_name}.npy',mse_error)
            # np.save(f'{path_out}inpainting_times/{name}_{block_name}_{int(stop-start)}s.npy',stop-start)
            np.save(f'{path_out}box_coords/{name}_{block_name}.npy', box_coord)
            # torch.save({'epoch': len(mse_error), 'model_state_dict': net.state_dict(),
            #    'LR': LR,'loss': mse_error, 'net_input_saved': net_input_saved}, 
            #    f'{path_out}v17v2_merged_clusters/models/{name}_{block_name}.pt')
            del net, images_generated_all
        
        stop = time()
        log.info(f'completed in {(stop-start)/60:.2f} mins')


if __name__ == '__main__':
    main()