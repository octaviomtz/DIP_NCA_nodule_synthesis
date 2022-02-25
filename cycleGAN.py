import argparse
import itertools
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scipy import ndimage, optimize
import pdb 
import cv2
import matplotlib.patches as patches
import multiprocessing
import datetime
from tqdm import tqdm
import os
from itertools import product
# from datasets import ImageDataset
# import visdom

import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
from torchsummary import summary

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from utils_cycleGAN.models import Generator3D
from utils_cycleGAN.models import Discriminator3D
from utils_cycleGAN.models import LIDCGAN_dataready
from utils_cycleGAN.utils import ReplayBuffer
from utils_cycleGAN.utils import LambdaLR
from utils_cycleGAN.utils import Logger
from utils_cycleGAN.utils import weights_init_normal
from utils_cycleGAN.extractRect import *

@hydra.main(config_path="config", config_name="config_cycleGAN.yaml")
def main(cfg: DictConfig):
    #HYDRA
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    path_orig = hydra.utils.get_original_cwd()

    path_dest = f'{path_orig}{cfg.path_dest}{cfg.exclude_this_and_previous}'

    if torch.cuda.is_available():
        cuda = True
        torch.cuda.set_device(0)

    # Networks
    netG_A2B = Generator3D(cfg.input_nc, cfg.output_nc)
    netG_B2A = Generator3D(cfg.output_nc, cfg.input_nc)
    netD_A = Discriminator3D(cfg.input_nc)
    netD_B = Discriminator3D(cfg.output_nc)

    if cuda:
        netG_A2B.cuda()
        netG_B2A.cuda()
        netD_A.cuda()
        netD_B.cuda()

    netG_A2B.apply(weights_init_normal)
    netG_B2A.apply(weights_init_normal)
    netD_A.apply(weights_init_normal)
    netD_B.apply(weights_init_normal)

    #summary(netG_A2B, (cfg.input_nc,32,32,32))

    # Losses
    criterion_GAN = torch.nn.MSELoss()
    criterion_cycle = torch.nn.L1Loss()
    criterion_identity = torch.nn.L1Loss()

    # Optimizers & LR schedulers
    optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                                    lr=cfg.lr, betas=(0.5, 0.999))
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=cfg.lr, betas=(0.5, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=cfg.lr, betas=(0.5, 0.999))

    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(cfg.n_epochs, cfg.epoch, cfg.decay_epoch).step)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(cfg.n_epochs, cfg.epoch, cfg.decay_epoch).step)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(cfg.n_epochs, cfg.epoch, cfg.decay_epoch).step)

    files = os.listdir(f'{cfg.path_source}original')
    files = np.sort(files)
    files = [i for i in files if 'raw' in i]
    print(f'total files = {len(files)}')

    # QUALITATIVE EVALUATION
    files_quality = os.listdir(cfg.path_quality)
    print(len(files_quality))
    files_quality = [f'{i[:-3]}raw' for i in files_quality]
    files_common = list(set(files).intersection(set(files_quality)))
    print(len(files_common), len(files_common)/len(files))

    # GET THE FILES ACCORDING TO 10FOLDS-CV (EXCLUDE VAL AND TEST)
    # Get the indices of the folds that should be included AND excluded

    fold_excluded, fold_included = [], []
    all_fold_exclude = np.arange(0,10)
    fold_exclude = all_fold_exclude[cfg.exclude_this_and_previous]
    fold_excluded.append(fold_exclude)
    all_fold_exclude = np.roll(all_fold_exclude,1) # ROLL
    fold_exclude = all_fold_exclude[cfg.exclude_this_and_previous]
    fold_excluded.append(fold_exclude)
    fold_included = [i for i in all_fold_exclude if i not in fold_excluded]
    fold_excluded, fold_included
    # EXCLUDE AND INCLUDE TEST FOLDS
    files_fold = files_common
    print(f'files after quality approval = {len(files_fold)}')
    # EXCLUDE TEST FOLDS
    for idx,i in enumerate(fold_excluded):
        path_csv_folds = f'/home/om18/Documents/KCL/Nov 5 2019 - Nodule Detection/LUNA16_nodule_detection/FP_reduction_3_5fold_xval/cycleGAN_augmented/results_fold_{i}/test_results_fold{i}.csv'
        csv_folds = pd.read_csv(path_csv_folds)
        files_in_fold = np.unique(csv_folds['seriesuid'].values)
        files_fold = [i for idx, i in enumerate(files_fold) if i.split('_')[0] not in files_in_fold]
        print(f'files to train after excluding {i} fold = {len(files_fold)}')
    # INCLUDE TEST FOLDS 
    files_in_folds =[]
    for i in fold_included:
        path_csv_folds = f'/home/om18/Documents/KCL/Nov 5 2019 - Nodule Detection/LUNA16_nodule_detection/FP_reduction_3_5fold_xval/cycleGAN_augmented/results_fold_{i}/test_results_fold{i}.csv'
        csv_folds = pd.read_csv(path_csv_folds)
        files_in_folds += list(np.unique(csv_folds['seriesuid'].values))
        
    files_fold = [i for idx, i in enumerate(files_fold) if i.split('_')[0] in files_in_folds]
    print(f'files to train after including the {fold_included} = {len(files_fold)}')
    files_common = files_fold

    print(f'fold_excluded = {fold_excluded}')
    print(f'fold_included = {fold_included}')


    # ### main
    fold = 1
    np.random.seed(fold)
    files_train, files_test = train_test_split(files_common, test_size=0.003, random_state=fold)
    files_train = np.sort(files_train)
    files_test = np.sort(files_test)

    len(files_train), len(files_test)

    # Since we are using ReplayBuffer we need to make sure our minibatches have the correct size
    batch_size_remainder_train = len(files_train) % cfg.batchsize
    batch_size_remainder_test = len(files_test) % cfg.batchsize
    if batch_size_remainder_train != 0:
        files_train = files_train[:-batch_size_remainder_train]
    if batch_size_remainder_test != 0:
        files_test = files_test[:-batch_size_remainder_test]
    #files_train = files_train[:-batch_size_remainder_train]
    #files_test = files_test[:-batch_size_remainder_test]
    len(files_train), len(files_test)

    
    dataset_train = LIDCGAN_dataready(files_train, cfg.path_source, transform=False)
    dataloader_train = DataLoader(dataset_train, batch_size=cfg.batchsize, shuffle=False)
    # set_all_rcParams(False) 
    # dataloader_iter = iter(dataloader_train)
    # plot_next_batch_dataloader(dataloader_iter)
    # plot_next_batch_dataloader(dataloader_iter)

    Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor
    input_A = Tensor(cfg.batchsize, cfg.input_nc, 32, 32, 32)
    input_B = Tensor(cfg.batchsize, cfg.input_nc, 32, 32, 32)
    target_real = Variable(Tensor(cfg.batchsize).fill_(1.0), requires_grad=False)
    target_fake = Variable(Tensor(cfg.batchsize).fill_(0.0), requires_grad=False)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    # def save_generated_images(name, files_to_check):
        
    #     if name in files_to_check:        
    #         np.save(f'{path_dest}A/{name}_ep{cfg.epoch:03d}',fake_A)
    #         np.save(f'{path_dest}B/{name}_ep{cfg.epoch:03d}',fake_B)

    path_netG_A2B = f'{path_dest}models/netG_A2B.pth'
    path_netG_B2A = f'{path_dest}models/netG_B2A.pth'
    path_netD_A = f'{path_dest}models/netD_A.pth'
    path_netD_B = f'{path_dest}models/netD_B.pth'

    dataset_test = LIDCGAN_dataready(files_test, transform=False)
    dataloader_test = DataLoader(dataset_test, batch_size=cfg.batchsize, shuffle=False)

    # to continue training
    try:
        cfg.epoch_done = int(np.load(f'{path_dest}models/last_epoch.npy'))
        netG_A2B.load_state_dict(torch.load(path_netG_A2B))
        netG_B2A.load_state_dict(torch.load(path_netG_B2A))
        netD_A.load_state_dict(torch.load(path_netD_A))
        netD_B.load_state_dict(torch.load(path_netD_B))
        loss_G_all = np.load(f'{path_dest}metrics/loss_G_all.npy')
        loss_G_identity_all = np.load(f'{path_dest}metrics/loss_G_identity_all.npy')
        loss_G_GAN_all = np.load(f'{path_dest}metrics/loss_G_GAN_all.npy')
        loss_G_cycle_all = np.load(f'{path_dest}metrics/loss_G_cycle_all.npy')
        loss_D_all = np.load(f'{path_dest}metrics/loss_D_all.npy')
        loss_G_all = np.expand_dims(loss_G_all,-1).tolist()
        loss_G_identity_all = np.expand_dims(loss_G_identity_all,-1).tolist()
        loss_G_GAN_all = np.expand_dims(loss_G_GAN_all,-1).tolist()
        loss_G_cycle_all = np.expand_dims(loss_G_cycle_all,-1).tolist()
        loss_D_all = np.expand_dims(loss_D_all,-1).tolist()
    except FileNotFoundError:
        cfg.epoch_done = -1
        loss_G_all = []
        loss_G_identity_all = []
        loss_G_GAN_all = []
        loss_G_cycle_all = []
        loss_D_all = []
    print(cfg.epoch_done)

    print(f'path_dest = {path_dest}')

    # path_test_output = f'{path_dest}test output/'

    for cfg.epoch in tqdm(range(cfg.n_epochs)):
        if cfg.epoch <=  cfg.epoch_done: continue
        #if cfg.epoch==1:break
        print(f'cfg.epoch = {cfg.epoch}')
        # A = original
        # B = inpainted
        for i, (A, B, mask, name) in tqdm(enumerate(dataloader_train), total= dataloader_train.dataset.__len__() // cfg.batchsize):
            netG_A2B.train()
            netG_B2A.train()
            netD_A.train()
            netD_B.train()
            
            # Set model input
            real_A = Variable(input_A.copy_(A))
            real_B = Variable(input_B.copy_(B))
            #pdb.set_trace()
            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B) # OMM
            loss_identity_B = criterion_identity(same_B, real_B) * 5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A) # OMM
            loss_identity_A = criterion_identity(same_A, real_A) * 5.0

            # GAN loss
            fake_B = netG_A2B(real_A) # OMM # 
            pred_fake = netD_B(fake_B)
            pred_fake.squeeze_() # OMM
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B) # OMM # 
            pred_fake = netD_A(fake_A)
            pred_fake.squeeze_() # OMM
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B) # OMM
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A) * 10.0

            recovered_B = netG_A2B(fake_A) # OMM
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B) * 10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()
            
            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            pred_real.squeeze_() # OMM
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            pred_fake.squeeze_() # OMM
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake) * 0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            pred_real.squeeze_() # OMM
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            pred_fake.squeeze_() # OMM
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake) * 0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################
            
            # Save results
            loss_G_all.append(loss_G.detach().cpu().numpy())
            loss_G_identity_all.append((loss_identity_A.detach().cpu().numpy() + loss_identity_B.detach().cpu().numpy()))
            loss_G_GAN_all.append((loss_GAN_A2B.detach().cpu().numpy() + loss_GAN_B2A.detach().cpu().numpy()))
            loss_G_cycle_all.append((loss_cycle_ABA.detach().cpu().numpy() + loss_cycle_BAB.detach().cpu().numpy()))
            loss_D_all.append((loss_D_A.detach().cpu().numpy() + loss_D_B.detach().cpu().numpy()))

                    # Save generated images (from a few examples)
            if cfg.epoch % 2 == 0:
                
                # save metrics
                loss_G_all_squeezed = np.squeeze(loss_G_all)
                loss_G_identity_all_squeezed = np.squeeze(loss_G_identity_all)
                loss_G_GAN_all_squeezed = np.squeeze(loss_G_GAN_all)
                loss_G_cycle_all_squeezed = np.squeeze(loss_G_cycle_all)
                loss_D_all_squeezed = np.squeeze(loss_D_all)
                np.save(f'{path_dest}metrics/loss_G_all',loss_G_all_squeezed)
                np.save(f'{path_dest}metrics/loss_G_identity_all',loss_G_identity_all_squeezed)
                np.save(f'{path_dest}metrics/loss_G_GAN_all',loss_G_GAN_all_squeezed)
                np.save(f'{path_dest}metrics/loss_G_cycle_all',loss_G_cycle_all_squeezed)
                np.save(f'{path_dest}metrics/loss_D_all',loss_D_all_squeezed)
                
                
                for iA, iB, i_name in zip(fake_A, fake_B, name):
                    
                    iA = np.squeeze(iA.detach().cpu().numpy())
                    iB = np.squeeze(iB.detach().cpu().numpy())
                    i_name = i_name.split('/')[-1]
                    
                    iA.tofile(f'{path_dest}A/{i_name}_ep{cfg.epoch:03d}') # SAVE THE IMAGES!!!!!
                    # iB.tofile(f'{path_dest}B/{i_name}_ep{cfg.epoch:03d}')
                    
                
                
                
                # new_name = name[0].split('/')[-1][:-4]
                ## if new_name in files_check: # WE NOW SAVE ALL ITERATIONS
                # np.save(f'{path_dest}A/{new_name}_ep{cfg.epoch:03d}',fake_A)
                # np.save(f'{path_dest}B/{new_name}_ep{cfg.epoch:03d}',fake_B)
            
        ##== Test sets 
        # netG_A2B.eval()
        # netG_B2A.eval()
                # input_B_test = Tensor(cfg.batchsize, cfg.output_nc, 64, 64)
        
        # for i, (A, B, mask, name) in enumerate(dataloader_test):
            # # Set model input
            # real_A = Variable(input_A_test.copy_(A))
            # real_B = Variable(input_B_test.copy_(B))

            # # Generate output # OMM the original equation was modified
            # fake_B = (netG_A2B(real_A).data) # 0.5*(netG_A2B(real_A).data + 1.0) <-original equation
            # fake_A = (netG_B2A(real_B).data)
            # new_name = name[0].split('/')[-1][:-4]
            # # Save output
            # np.save(f'{path_test_output}A/{new_name}_ep{cfg.epoch:03d}', fake_A)
            # np.save(f'{path_test_output}B/{new_name}_ep{cfg.epoch:03d}', fake_B)
        

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()
        
        

        # Save models checkpoints
        torch.save(netG_A2B.state_dict(), path_netG_A2B)
        torch.save(netG_B2A.state_dict(), path_netG_B2A)
        torch.save(netD_A.state_dict(), path_netD_A)
        torch.save(netD_B.state_dict(), path_netD_B)
        np.save(f'{path_dest}models/last_epoch.npy', cfg.epoch)

if __name__ == '__main__':
    main()