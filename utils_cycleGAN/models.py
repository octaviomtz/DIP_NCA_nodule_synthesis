import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch import Tensor
import matplotlib.pyplot as plt

class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm2d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(256, 512, 4, padding=1),
                    nn.InstanceNorm2d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv2d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock3D, self).__init__()

        conv_block = [  nn.ConstantPad3d(1, value=0), # instead of ReflectionPad3d
                        nn.Conv3d(in_features, in_features, 3),
                        nn.InstanceNorm3d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ConstantPad3d(1, value=0), # instead of ReflectionPad3d
                        nn.Conv3d(in_features, in_features, 3),
                        nn.InstanceNorm3d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)

class Generator3D(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator3D, self).__init__()

        # Initial convolution block       
        model = [   nn.ConstantPad3d(3, value=0), # instead of ReflectionPad3d
                    nn.Conv3d(input_nc, 64, 7),
                    nn.InstanceNorm3d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv3d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock3D(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose3d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm3d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ConstantPad3d(3, value=0), # instead of ReflectionPad3d
                    nn.Conv3d(64, output_nc, 7),
                    nn.Tanh() ]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

class Discriminator3D(nn.Module):
    def __init__(self, input_nc):
        super(Discriminator3D, self).__init__()

        # A bunch of convolutions one after another
        model = [   nn.Conv3d(input_nc, 64, 4, stride=2, padding=1),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(64, 128, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(128), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(128, 256, 4, stride=2, padding=1),
                    nn.InstanceNorm3d(256), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv3d(256, 512, 4, padding=1),
                    nn.InstanceNorm3d(512), 
                    nn.LeakyReLU(0.2, inplace=True) ]

        # FCN classification layer
        model += [nn.Conv3d(512, 1, 4, padding=1)]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        # Average pooling and flatten
        return F.avg_pool3d(x, x.size()[2:]).view(x.size()[0], -1)


class LIDCGAN_dataready(Dataset):
        def __init__(self, files_train_test, path_source, transform = False):
            self.A = [f'{path_source}original/{i}' for i in files_train_test]
            self.B = [f"{path_source}inpainted inserted/{i}" for i in files_train_test]
            self.mask = [f"{path_source}mask/{i}" for i in files_train_test]
            self.transform = transform
            
        def __len__(self):
            return len(self.A)
        
        def normalizePatches(self, npzarray):
            maxHU = 400.
            minHU = -1000.

            npzarray = (npzarray - minHU) / (maxHU - minHU)
            npzarray[npzarray>1] = 1.
            npzarray[npzarray<0] = 0.
            return npzarray
        
        def rotate_and_flip(self, image1, image2, image3):
            rand_int = np.random.randint(1,4)
            if np.random.rand() > .3:
                image1 = np.rot90(image1,rand_int).copy()
                image2 = np.rot90(image2,rand_int).copy()
                image3 = np.rot90(image3,rand_int).copy()
            if np.random.rand() > .5:
                image1 = np.flip(image1,0).copy()
                image2 = np.flip(image2,0).copy()
                image3 = np.flip(image3,0).copy()
            if np.random.rand() > .5:
                image1 = np.flip(image1,1).copy()
                image2 = np.flip(image2,1).copy()
                image3 = np.flip(image3,1).copy()
            return image1, image2, image3
                                
        def __getitem__(self, idx):
            imgA = np.fromfile(self.A[idx],dtype='int16').astype('float32').reshape((64,64,64))
            imgB = np.fromfile(self.B[idx],dtype='int16').astype('float32').reshape((64,64,64))
            mask = np.fromfile(self.mask[idx],dtype='int16').astype('float32').reshape((64,64,64))
            
            # v1: ONLY USE THE INSIDE OF THE CUBE (32x32x32 size)
            imgA = imgA[16:-16,16:-16,16:-16]
            imgB = imgB[16:-16,16:-16,16:-16]
            mask = mask[16:-16,16:-16,16:-16]
            
            # normalize
            imgA = self.normalizePatches(imgA)
            imgB = self.normalizePatches(imgB) 
            
            # Flips
            if self.transform:
                imgA, imgB, mask = self.rotate_and_flip(imgA, imgB, mask)
                        
            # Add channels dimension
            imgA = np.expand_dims(imgA,0)
            imgB = np.expand_dims(imgB,0)
            mask = np.expand_dims(mask,0)
            
            # Pytorch
            imgA = Tensor(imgA)
            imgB = Tensor(imgB)
            mask = Tensor(mask)
                    
            # Get name to save the data                      
    #         name = self.A[idx].split('orig/')[-1]
            name = self.A[idx]
        
            return imgA, imgB, mask, name

def plot_next_batch_dataloader(dataloader_iterX, slice_middle = 15):
    a, b, mask, name = next(dataloader_iterX)
    print(np.shape(a), np.shape(b), np.shape(mask))
    for i,j,k in zip(a,b,mask):
        fig, ax = plt.subplots(1,4,figsize=(10,5))
        i = i[0].detach().cpu().numpy()
        j = j[0].detach().cpu().numpy()
        k = k[0].detach().cpu().numpy()
        print(np.shape(i), np.shape(j), np.shape(k))
        new_name = name[0].split('/')[-1][:-4]
        ax[0].imshow(i[slice_middle], vmin=0, vmax=1)
        ax[1].imshow(j[slice_middle], vmin=0, vmax=1)
        ax[2].imshow(np.abs(i[slice_middle]-j[slice_middle]), vmin=0, vmax=.1)
        ax[3].imshow(k[slice_middle])
        fig.tight_layout()

def shape_next_batch_dataloader(dataloader_iterX):
    a, b, mask, name = next(dataloader_iterX)
    for i,j,k in zip(a,b,mask):
        print(i.shape)
        print(j.shape)