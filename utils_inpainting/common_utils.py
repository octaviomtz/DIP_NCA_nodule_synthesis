import torch
import torch.nn as nn
import torchvision
import sys

import numpy as np
from PIL import Image
import PIL
import numpy as np

import matplotlib.pyplot as plt
import pdb
# from utils_omm.clr import CyclicLR 

def crop_image(img, d=32):
    '''Make dimensions divisible by `d`'''

    new_size = (img.size[0] - img.size[0] % d, 
                img.size[1] - img.size[1] % d)

    bbox = [
            int((img.size[0] - new_size[0])/2), 
            int((img.size[1] - new_size[1])/2),
            int((img.size[0] + new_size[0])/2),
            int((img.size[1] + new_size[1])/2),
    ]

    img_cropped = img.crop(bbox)
    return img_cropped

def get_params(opt_over, net, net_input, downsampler=None):
    '''Returns parameters that we want to optimize over.
    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    
    for opt in opt_over_list:
    
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='down':
            assert downsampler is not None
            params = [x for x in downsampler.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'
            
    return params

def get_image_grid(images_np, nrow=8):
    '''Creates a grid from a list of images by concatenating them.'''
    images_torch = [(torch.from_numpy(x)).double() for x in images_np] # OMM added ().double()
    torch_grid = torchvision.utils.make_grid(images_torch, nrow)
    
    return torch_grid.numpy()

def plot_image_grid(images_np, nrow =8, factor=1, interpolation=None): # OMM 'lanczos' -> None
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    n_channels = max(x.shape[0] for x in images_np)
    assert (n_channels == 3) or (n_channels == 1), "images should have 1 or 3 channels"
    
    images_np = [x if (x.shape[0] == n_channels) else np.concatenate([x, x, x], axis=0) for x in images_np]

    grid = get_image_grid(images_np, nrow)
    
    plt.figure(figsize=(len(images_np) + factor, 12 + factor))
    
    if images_np[0].shape[0] == 1:
        plt.imshow(grid[0], cmap='viridis', interpolation=interpolation) #OMM 'gray' -> 'viridis'
    else:
        plt.imshow(grid.transpose(1, 2, 0), interpolation=interpolation)
    
    plt.show()
    
    return grid

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size. 
    Args: 
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np



def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_() 
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)
        
        fill_noise(net_input, noise_type)
        net_input *= var            
    elif method == 'meshgrid': 
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False
        
    return net_input

def get_noise2(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
        #net_input += .7 #WARNING ADDED FOR MELANOMA
        print('using get_noise2')
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False

    return net_input


def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.
    
    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / 255.

def np_to_pil(img_np): 
    '''Converts image in np.array format to PIL image.
    
    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)
    
    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def optimize(optimizer_type, parameters, closure, LR, num_iter, show_every):
    """Runs optimization loop.
    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            #if j ==0: DEL 
                #loss_best=1000 #DEL
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure()
            total_loss.append(total_loss_temp)
            #if j < 50 and j % 10 == 0:
                #images_generated.append(image_generated_temp)
            if j % show_every == 0:
            #if total_loss_temp < loss_best:
                #loss_best = total_loss_temp
                images_generated.append(image_generated_temp)
            #print(f'total_loss_temp:{total_loss_temp}')
            optimizer.step()
    else:
        assert False
    return total_loss, images_generated


def optimize2(optimizer_type, parameters, closure, LR, num_iter, show_every, path_img_dest, annealing=False):
    """Runs optimization loop.
    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []
    if optimizer_type == 'LBFGS':
        # Do several steps with adam first
        optimizer = torch.optim.Adam(parameters, lr=0.001)
        for j in range(100):
            optimizer.zero_grad()
            closure()
            optimizer.step()

        print('Starting optimization with LBFGS')        
        def closure2():
            optimizer.zero_grad()
            return closure()
        optimizer = torch.optim.LBFGS(parameters, max_iter=num_iter, lr=LR, tolerance_grad=-1, tolerance_change=-1)
        optimizer.step(closure2)

    elif optimizer_type == 'adam':
        print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            if j == 0:
                loss_best = 1000
                j_best = 0
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure()
            total_loss.append(total_loss_temp)
            
            if total_loss_temp < loss_best:
                loss_best = total_loss_temp
                iterations_without_improvement = 0
                j_best = j
                if j > 500: # the first 500 iterations are blurry
                    images_generated.append(image_generated_temp)
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            if annealing == True and iterations_without_improvement == 300:
                LR = adjust_learning_rate(optimizer, j, LR)
                print(f'LR reduced to: {LR}')
            #print(f'total_loss_temp:{total_loss_temp}')
            optimizer.step()
            iterations_without_improvement += 1
            if iterations_without_improvement == 500:
                print(f'training stopped at it {j} after 500 iterations without improvement')
                break

    elif optimizer_type == 'clr':
        #print('Starting optimization with Cyclic Learning Rate')
        optimizer = torch.optim.SGD(parameters, lr=LR, momentum=0.9)
        scheduler = CyclicLR(optimizer)
        #optimizer = torch.optim.Adam(parameters, lr=LR)
        
        for j in range(num_iter):
            if j == 0:
                loss_best = 1000
                j_best = 0
            scheduler.batch_step()
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure()
            total_loss.append(total_loss_temp)
            
            if total_loss_temp < loss_best:
                loss_best = total_loss_temp
                iterations_without_improvement = 0
                j_best = j
                if j > 500: # the first 500 iterations are blurry
                    images_generated.append(image_generated_temp)
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            #print(f'total_loss_temp:{total_loss_temp}')
            optimizer.step()
            iterations_without_improvement += 1
            if iterations_without_improvement == 500:
                print(f'training stopped at it {j} after 500 iterations without improvement')
                break
    else:
        assert False
    return total_loss, images_generated, j_best

def plot_for_gif(image_to_save,num_iter, i, path_img_dest):
    fig, ax = plt.subplots(1,2, gridspec_kw = {'width_ratios':[8, 1]}, figsize=(14,10))
    ax[0].imshow(image_to_save[0], cmap='viridis')
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
    plt.savefig(f'{path_img_dest}images before gifs/iter {i:05d}.jpeg',
                bbox_inches = 'tight',pad_inches = 0)
    plt.close(fig)

def adjust_learning_rate(optimizer, epoch, LR):
    """Sets the learning rate to the initial LR decayed by 10 every X epochs"""
    #LR = LR * (0.1 ** (epoch // 1000))
    LR = LR * 0.1 
    for param_group in optimizer.param_groups:
        param_group['lr'] = LR
    return LR

def optimize_v18(optimizer_type, parameters, closure, LR, num_iter, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """
    # We stop training when loss reaches 0.0005. We saw some examples that looked ok
    Runs optimization loop.
    Args:
        optimizer_type: 'LBFGS' of 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        num_iter: number of iterations 
    """
    total_loss = []
    images_generated = []
    if optimizer_type == 'adam':
        #print('Starting optimization with ADAM')
        optimizer = torch.optim.Adam(parameters, lr=LR)
        
        lr_finder = 1e-7
        lrs_finder = []
        for j in range(num_iter):
            
            # LR finder
            if lr_finder_flag:
                if j == 0: print('Finding LR')
                lr_finder = lr_finder * 1.2
                if lr_finder >= 1e-2: #completed
                    return total_loss, images_generated, j_best, 
                lrs_finder.append(lr_finder)
                optimizer = torch.optim.Adam(parameters, lr=lr_finder)
            
            # Init values for best loss selection
            if j == 0:
                loss_best = 1000
                j_best = 0
                
            optimizer.zero_grad()
            total_loss_temp, image_generated_temp = closure()
            total_loss.append(total_loss_temp)
            
            # Restart if bad initialization
            print(type(total_loss))
            if j == 200 and np.sum(np.diff(total_loss.cpu())==0) > 50:
                """if the network is not learning (bad initialization) Restart"""
                restart = True
                return total_loss, images_generated, j_best, restart
            else:
                restart = False
               
            if total_loss_temp < 0.0005: 
                print('loss reached 0.0005')
                break # v18
            
            if total_loss_temp < loss_best:
                loss_best = total_loss_temp
                iterations_without_improvement = 0
                j_best = j
                if j > 1500: # the first 1500 iterations are blurry
                    images_generated.append(image_generated_temp)
                    #plot_for_gif(image_generated_temp, num_iter, j, path_img_dest)
            if annealing == True and iterations_without_improvement == 200:
                LR = adjust_learning_rate(optimizer, j, LR)
                print(f'LR reduced to: {LR:.5f}')
            #print(f'total_loss_temp:{total_loss_temp}')
            optimizer.step()
            iterations_without_improvement += 1
            if iterations_without_improvement == 500:
                print(f'training stopped at it {j} after 500 iterations without improvement')
                break
    else:
        assert False
    return total_loss, images_generated, j_best, restart

def optimize_ndls(optimizer_type, parameters, closure, LR, EPOCHS, show_every, path_img_dest, restart = True, annealing=False, lr_finder_flag=False):
    """
    # We stop training when loss reaches 0.0005. We saw some examples that looked ok
    Runs optimization loop.
    Args:
        optimizer_type: 'adam'
        parameters: list of Tensors to optimize over
        closure: function, that returns loss variable
        LR: learning rate
        EPOCHS: number of iterations 
    """
    total_loss = []
    images_generated = []
    optimizer = torch.optim.Adam(parameters, lr=LR)
    
    for j in range(EPOCHS):
        
        # Init values for best loss selection
        if j == 0:
            loss_best = 1000
            j_best = 0
            
        optimizer.zero_grad()
        total_loss_temp, image_generated_temp = closure()
        total_loss.append(total_loss_temp.detach().cpu().numpy())
        
        # Restart if bad initialization
        # print(type(total_loss), type(total_loss[0]), (total_loss[:3]), '\n')
        if j == 200 and np.sum(np.diff(total_loss)==0) > 50:
            """if the network is not learning (bad initialization) Restart"""
            restart = True
            return total_loss, images_generated, j_best, restart
        else:
            restart = False
            
        if total_loss_temp < 0.0005: 
            print('\nloss reached 0.0005')
            images_generated.append(image_generated_temp)
            break # v18
        
        if total_loss_temp < loss_best:
            loss_best = total_loss_temp
            iterations_without_improvement = 0
            j_best = j
            if j > 3000 and j % 100 == 0: # skip the first epochs
                images_generated.append(image_generated_temp)
                #plot_for_gif(image_generated_temp, EPOCHS, j, path_img_dest)

        if annealing == True and iterations_without_improvement == 200:
            LR = adjust_learning_rate(optimizer, j, LR)
            print(f'LR reduced to: {LR:.5f}')
        
        optimizer.step()
        iterations_without_improvement += 1
        if iterations_without_improvement == 500:
            print(f'training stopped at it {j} after 500 iterations without improvement')
            break
    return total_loss, images_generated, j_best, restart