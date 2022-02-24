import os
import io
import PIL.Image, PIL.ImageDraw
import base64
import zipfile
import json
import requests
import numpy as np
import matplotlib.pylab as pl
import matplotlib.pyplot as plt
from pathlib import Path
import glob
import time
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv3D
from IPython.display import Image, HTML, clear_output
from tqdm import tqdm
import os
os.environ['FFMPEG_BINARY'] = 'ffmpeg'
import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
clear_output()
from scipy.ndimage import label
import sys

from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from utils.neural_cellular_automata1 import (fig_multiple3D, get_living_mask,
                                            place_seed_in_corners, largest_distance,
                                            to_rgba, SamplePool, export_model)

def export_model(ca, base_fn, CHANNEL_N=16):
  ca.save_weights(base_fn)

  cf = ca.call.get_concrete_function(
      x=tf.TensorSpec([None, None, None, None, CHANNEL_N]),
      fire_rate=tf.constant(0.5),
      angle=tf.constant(0.0),
      step_size=tf.constant(1.0))
  cf = convert_to_constants.convert_variables_to_constants_v2(cf)
  graph_def = cf.graph.as_graph_def()
  graph_json = MessageToDict(graph_def)
  graph_json['versions'] = dict(producer='1.14', minConsumer='1.14')
  model_json = {
      'format': 'graph-model',
      'modelTopology': graph_json,
      'weightsManifest': [],
  }
  with open(base_fn+'.json', 'w') as f:
    json.dump(model_json, f)

@hydra.main(config_path="config", config_name="config.yaml")
def main(cfg: DictConfig):
    #HYDRA
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))
    path_orig = hydra.utils.get_original_cwd()


    path_dest = f'{path_orig}/ca_nodules_generated/'
    nodules_sorted = np.load('/content/drive/MyDrive/KCL/cellular automata v2/nodules_sorted3.npz')
    filenames_sorted = np.load('/content/drive/MyDrive/KCL/cellular automata v2/files_sorted3.npy')
    nodules_sorted = nodules_sorted.f.arr_0
    # For now work only with 32x32x32
    nodules_smaller = nodules_sorted[:,16:-16, 16:-16, 16:-16]
    # only use those images that include only 1 nodule
    nodules_centered, names_centered = [], []
    for i, name in zip(nodules_smaller, filenames_sorted):
        labelled, nr = label(i)
        if nr == 1:
          nodules_centered.append(i)
          names_centered.append(name)
    nodules_smaller = np.expand_dims(nodules_centered,1)
    print(f'nodules_sorted = {np.shape(nodules_sorted)}, nodules_smaller = {np.shape(nodules_smaller)}')

    # 3D versions
    nodules3D = [np.swapaxes(i,0,1) for i in nodules_smaller]
    nodules3D = [np.swapaxes(i,1,2) for i in nodules3D]
    nodules3D = [np.swapaxes(i,2,3) for i in nodules3D]
    print(np.shape(nodules3D))

    fig_multiple3D(nodules3D[1610:],r=4,c=4,name='nodules 1400')

    class CAModel3D(tf.keras.Model):

      def __init__(self, channel_n = cfg.CHANNEL_N, fire_rate = cfg.CELL_FIRE_RATE):
        super().__init__()
        self.channel_n = channel_n
        self.fire_rate = fire_rate

        self.dmodel = tf.keras.Sequential([
              Conv3D(128, 1, activation=tf.nn.relu),
              Conv3D(self.channel_n, 1, activation=None,
                  kernel_initializer=tf.zeros_initializer),
        ])

        self(tf.zeros([1, 3, 3, 3, channel_n]))  # dummy call to build the model

      @tf.function
      def perceive(self, x, angle=0.0):
        identify = np.float32(np.zeros((3,3,3)))
        identify[1,1,1] = 1
        dx = np.float32(np.asarray([[[1,2,1],[2,4,2],[1,2,1]],[[0,0,0],[0,0,0],[0,0,0]],[[-1,-2,-1],[-2,-4,-2],[-1,-2,-1]]]) / 32.0) # Engel. Real-Time Volume Graphics (p.112)
        dy = dx.transpose((1,0,2))
        dz = dx.transpose((2,1,0))
        c, s = tf.cos(0.), tf.sin(0.)
        kernel = tf.stack([identify, dx, dy, dz], -1)[:, :, :, None, :] # we removed the sin cos used for rotations
        kernel = tf.repeat(kernel, self.channel_n, 3) # OMM WARNING maybe the last param is 2
        # kernel = tf.repeat(kernel, self.channel_n, 2)
        # function to replace tf.nn.depthwise_conv3d
        # https://github.com/alexandrosstergiou/keras-DepthwiseConv3D/blob/master/DepthwiseConv3D.py
        input_dim = x.shape[-1]
        groups = input_dim
        y = tf.concat([tf.nn.conv3d(x[:,:,:,:,i:i+input_dim//groups], kernel[:,:,:,i:i+input_dim//groups,:],
                        strides=[1, 1, 1, 1, 1],
                        padding= 'SAME') 
        for i in range(0,input_dim,input_dim//groups)], axis=-1)
        
        return y

      @tf.function
      def call(self, x, fire_rate=None, angle=0.0, step_size=1.0):
        pre_life_mask = get_living_mask(x)

        y = self.perceive(x, angle)
        dx = self.dmodel(y)*step_size
        if fire_rate is None:
          fire_rate = self.fire_rate
        update_mask = tf.random.uniform(tf.shape(x[:, :, :, :, :1])) <= fire_rate
        x += dx * tf.cast(update_mask, tf.float32)

        post_life_mask = get_living_mask(x)
        life_mask = pre_life_mask & post_life_mask
        return x * tf.cast(life_mask, tf.float32)

    CAModel3D().dmodel.summary()

    lr_sched = tf.keras.optimizers.schedules.PiecewiseConstantDecay([2000], [cfg.lr, cfg.lr*0.1])
    trainer = tf.keras.optimizers.Adam(lr_sched)

    def loss_f(x):
      return tf.reduce_mean(tf.square(to_rgba(x)-target_img), [-2, -3, -4, -1])

    for one_nodule in np.arange(cfg.NDL_OFFSET, cfg.NDL_OFFSET+cfg.NDL_NUM):

      FISH = (32,32,32,2) # final shape
      ndl = nodules3D[one_nodule]
      filename_one_nodule = names_centered[one_nodule]
      nsh = np.shape(ndl)
      ndl_pad_temp = np.pad(np.squeeze(ndl),(((FISH[0] - nsh[0])//2, (FISH[0] - nsh[0])//2), ((FISH[1] - nsh[1])//2, (FISH[1] - nsh[1])//2), ((FISH[2] - nsh[2])//2, (FISH[2] - nsh[2])//2)))
      target_img = np.zeros(FISH)
      target_img[:,:,:,0] = ndl_pad_temp
      target_img[:,:,:,1] = ndl_pad_temp>0 # the second channel should be the active cells
      # 2. GET THE SEED
      h, w, d = target_img.shape[:3]
      seed = np.zeros([h, w, d, cfg.CHANNEL_N], np.float32)
      iq, jq, kq = 16, 16, 16
      # coord1, coord2 = largest_distance(ndl_pad_temp)
      # iq, jq, kq = coord2
      seed[iq, jq, kq , 1:] = 1.0
      x = np.expand_dims(seed,0)

      fig, ax = plt.subplots(1,2,figsize=(4,2))
      ax[0].imshow(target_img[16,...,0])
      ax[1].imshow(x[0,iq,...,1])
      for axx in ax.ravel(): axx.axis('off')
      fig.tight_layout()
      plt.savefig(f'nodule_and_seed_{one_nodule:04d}')

      # 3. MODEL, LOSS AND LEARNER
      tf.compat.v1.reset_default_graph()
      ca = CAModel3D()
      loss_log = []
      loss0 = loss_f(seed).numpy()
      pool = SamplePool(x=np.repeat(seed[None, ...], cfg.POOL_SIZE, 0))
      Path(f'{path_orig}/train_logB').mkdir(parents=True, exist_ok=True)
      try:
          os.remove(f'{path_orig}/train_logB/*')
      except OSError:
          pass
      # !mkdir -p train_logB && rm -f train_logB/*

      # 4. TRAIN
      @tf.function
      def train_step(x):
        iter_n = tf.random.uniform([], 64, 96, tf.int32) # random range
        with tf.GradientTape() as g:
          for i in tf.range(iter_n):
            x = ca(x)
          loss = tf.reduce_mean(loss_f(x))
        grads = g.gradient(loss, ca.weights)
        grads = [g/(tf.norm(g)+1e-8) for g in grads]
        trainer.apply_gradients(zip(grads, ca.weights))
        return x, loss

      for i in tqdm(range(cfg.TRAIN_EPOCHS+1), total=cfg.TRAIN_EPOCHS+1, desc = str(one_nodule), leave=False):
        x0 = np.repeat(seed[None, ...], cfg.BATCH_SIZE, 0)
        x, loss = train_step(x0)
        loss_log.append(loss.numpy())
        if i in [cfg.TRAIN_EPOCHS]:
            export_model(ca, f'{path_orig}/train_logB/{i:04d}')

      # 5. GET TRAINED IMAGES
      grow_iterations = 100
      models = []
      for i in [cfg.TRAIN_EPOCHS]: 
        ca = CAModel3D()
        ca.load_weights(f'{path_orig}/train_logB/%04d'%i)
        models.append(ca)
      nodule_growing = []
      x = np.zeros([len(models), 32, 32, 32, cfg.CHANNEL_N], np.float32)
      x[..., iq, jq, kq, 1:] = 1.0
      #print(f'5A. {one_nodule}')
      for i in range(grow_iterations):
        for ca, xk in zip(models, x):
          temp = ca(xk[None,...])[0]
          xk[:] = temp
          nodule_growing.append(temp.numpy()[...,0]) 

      # 6. SAVE NODULE FROM MODEL TRAINED THE LONGEST
      if len(models)>1:
        nodule_growing = [i for idx, i in enumerate(nodule_growing) if idx%len(models)-1==0] # get the images from the last model
      # nodule_growing = [i[...,0].numpy() for idx, i in enumerate(nodule_growing)] # get the images from the last model
      nodule_growing = np.stack(nodule_growing, axis=0) # convert from list of 3d arrays to 4d array
      np.savez_compressed(f'{path_dest}{filename_one_nodule}.npz', nodule_growing)
      loss_log_py = np.asarray(loss_log)
      np.save(f'{path_dest}weights_and_losses/loss_{filename_one_nodule}.npy', loss_log_py)
      ca.save_weights(f'{path_dest}weights_and_losses/weights_{filename_one_nodule}')

if __name__ == '__main__':
    main()