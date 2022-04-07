#%%
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
from copy import copy
from typing import Tuple, List
import os
from time import time

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from utils.preprocess import normalizePatches
from utils.cellular_automata import (
    get_raw_nodule,
    crop_only_nodule,
    get_center_of_volume_from_largest_component_return_center_or_extremes,
    perception
)


class CA(torch.nn.Module):
    def __init__(self, ident, sobel_x, lap, device, chn=16, hidden_n=96):
        super().__init__()
        self.ident = ident
        self.sobel_x = sobel_x
        self.sobel_y = torch.rot90(self.sobel_x, k=1, dims= (0,1))
        self.sobel_z = torch.rot90(self.sobel_x, k=1, dims= (0,2))
        self.lap = lap
        self.device = device
        self.chn = chn

        self.w1 = torch.nn.Conv3d(chn*5, hidden_n, 3, padding='same')
        self.w2 = torch.nn.Conv3d(hidden_n, chn, 1, bias=False)
        self.w2.weight.data.zero_()

    def forward(self, x, update_rate=0.5, noise=None):
        # if noise is not None:
        #     x += torch.randn_like(x)*noise
        y = perception(x, self.device, self.ident, self.sobel_x, self.sobel_y, self.sobel_z, self.lap)
        y = self.w1(y)
        y = self.w2(torch.relu(y))
        b, c, z, h, w = y.shape
        udpate_mask = (torch.rand(b, 1, z, h, w, device=self.device)+update_rate).floor()
        # udpate_mask = udpate_mask.to(device)
        return x + y*udpate_mask

@hydra.main(config_path="config", config_name="config_ca.yaml")
def main(cfg: DictConfig):
    # HYDRA
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    path_orig = hydra.utils.get_original_cwd()
    # PATHS
    path_data = f'{cfg.path_data}subset{cfg.SUBSET}/'
    path_dest = f'{cfg.path_dest}subset{cfg.SUBSET}/'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    files = os.listdir(f'{path_data}original')
    for idx_name, ndl in tqdm(enumerate(files), total=len(files)):
        if idx_name < cfg.SKIP_IDX: continue
        log.info(f'cellular_automata: {idx_name}, {ndl}')
        start = time()

        if idx_name ==50: break
        
        last, orig, mask = get_raw_nodule(path_data, ndl)
        ndl_only, ndl_only_coords = crop_only_nodule(mask, orig)
        center_z, center_y, center_x = get_center_of_volume_from_largest_component_return_center_or_extremes(mask)

        # Minimalistic Neural CA
        ident = torch.zeros((3,3,3), device=device)
        ident[1,1,1] = 1.0
        sobel_x = torch.tensor([[[1,2,1],[2,4,2],[1,2,1]],[[0,0,0],[0,0,0],[0,0,0]],[[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]], device=device)/32.0
        lap = torch.tensor([[[2,3,2],[3,6,3],[2,3,2]],[[3,6,3],[6,-88,6],[3,6,3]],[[2,3,2],[3,6,3],[2,3,2]]], device=device) / 26.0 #https://en.wikipedia.org/wiki/Discrete_Laplace_operator

        SHZ, SHY, SHX = ndl_only.shape[0], ndl_only.shape[1], ndl_only.shape[2]
        pool = torch.zeros((256, 16, SHZ, SHY, SHX))
        seed = np.zeros((1, 16, SHZ, SHY, SHX))
        c_crop_z = center_z - ndl_only_coords[0]
        c_crop_y = center_y - ndl_only_coords[2]
        c_crop_x = center_x - ndl_only_coords[4]
        seed[:,1:, c_crop_z, c_crop_y, c_crop_x] = cfg.SEED_INIT
        seed = torch.from_numpy(seed.astype(np.float32))
        seed = seed.to(device)

        ca = CA(ident, sobel_x, lap, device)
        ca = ca.to(device)
        opt = torch.optim.Adam(ca.parameters(), 1e-3)
        lr_sched = torch.optim.lr_scheduler.MultiStepLR(opt, [2000, 4000, 6000], 0.3)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9999)

        target = torch.from_numpy(ndl_only)
        target_alpha = (target>0)*1.
        target = torch.stack([target, target_alpha], dim=0)[None,...]
        target = target.to(device)

        start = time()
        losses = []
        for ep in tqdm(range(cfg.EPOCHS)):
            opt.zero_grad()
            x = seed
            for i in range(cfg.ITER_CA):
                x = ca(x)
            # x = torch.moveaxis(x, 1, -1)
            loss = F.mse_loss(x[:, :2, ...], target)
            loss.backward()
            losses.append(loss.detach().cpu().numpy())
            for p in ca.parameters():
                p.grad /= (p.grad.norm()+1e-8)
            opt.step()
            scheduler.step()
        stop = time()

        # 5. GET TRAINED IMAGES
        nodule_growing = []
        x = seed
        nodule_growing_losses = []
        for i in range(cfg.ITER_SYNTHESIS):
            x = ca(x)
            x_py = x.detach().cpu().numpy()[0,...]
            nodule_growing.append(x_py)
            target_py = target.detach().cpu().numpy()[0,...]
            nodule_growing_losses.append(np.mean((x_py[:2,...] - target_py)**2))

        nodule_growing = np.stack(nodule_growing, axis=0)
        nodule_growing_losses = np.stack(nodule_growing_losses, axis=0)
        if not os.path.exists(f'{path_dest}{ndl[:-3]}/'): os.makedirs(f'{path_dest}{ndl[:-3]}/')
        np.savez_compressed(f'{path_dest}{ndl[:-3]}/synthesis_ca.npz', nodule_growing)
        np.savez_compressed(f'{path_dest}{ndl[:-3]}/synthesis_ca_loss.npz', nodule_growing_losses)
        time_and_min_loss = f'{(stop-start)/60:.2f}m_{np.min(nodule_growing_losses):.1E}'
        with open(f'{path_dest}{ndl[:-3]}/{time_and_min_loss}.txt', 'w') as f: f.write(time_and_min_loss)
        with open(f'{path_dest}{ndl[:-3]}/zyx_min_max_len.txt', 'w') as f: f.write(f'{ndl_only_coords}, {ndl_only.shape}')

        log.info(f'completed in {(stop-start)/60:.2f} mins')

        # # figures of growing nodule
        # tgt_min, tgt_max = np.min(target_py), np.max(target_py)
        # _, _, sh_z, _, _= nodule_growing.shape
        # fig, ax = plt.subplots(5,10, figsize=(24,12))
        # for i in range(50):
        #     ii = i
        #     ax.flat[i].imshow(nodule_growing[ii,0,sh_z//2,...], vmin = tgt_min, vmax=tgt_max)
        #     ax.flat[i].axis('off')
        # ax.flat[0].imshow(target_py[0, sh_z//2,...], vmin = tgt_min, vmax=tgt_max)
        # plt.tight_layout()

        # # plot reconstruction loss
        # plt.semilogy(nodule_growing_losses)

if __name__ == "__main__":
    main()