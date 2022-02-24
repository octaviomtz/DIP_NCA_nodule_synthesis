import tensorflow as tf
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
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from google.protobuf.json_format import MessageToDict
from tensorflow.python.framework import convert_to_constants
from scipy.spatial import distance

def np2pil(a):
  if a.dtype in [np.float32, np.float64]:
    a = np.uint8(np.clip(a, 0, 1)*255)
  return PIL.Image.fromarray(a)

def imwrite(f, a, fmt=None):
  a = np.asarray(a)
  if isinstance(f, str):
    fmt = f.rsplit('.', 1)[-1].lower()
    if fmt == 'jpg':
      fmt = 'jpeg'
    f = open(f, 'wb')
  np2pil(a).save(f, fmt, quality=95)

def imencode(a, fmt='jpeg'):
  a = np.asarray(a)
  if len(a.shape) == 3 and a.shape[-1] == 4:
    fmt = 'png'
  f = io.BytesIO()
  imwrite(f, a, fmt)
  return f.getvalue()

def im2url(a, fmt='jpeg'):
  encoded = imencode(a, fmt)
  base64_byte_string = base64.b64encode(encoded).decode('ascii')
  return 'data:image/' + fmt.upper() + ';base64,' + base64_byte_string

# def imshow(a, fmt='jpeg'):
#   display(Image(data=imencode(a, fmt)))

def tile2d(a, w=None):
  a = np.asarray(a)
  if w is None:
    w = int(np.ceil(np.sqrt(len(a))))
  th, tw = a.shape[1:3]
  pad = (w-len(a))%w
  a = np.pad(a, [(0, pad)]+[(0, 0)]*(a.ndim-1), 'constant')
  h = len(a)//w
  a = a.reshape([h, w]+list(a.shape[1:]))
  a = np.rollaxis(a, 2, 1).reshape([th*h, tw*w]+list(a.shape[4:]))
  return a

def zoom(img, scale=4):
  img = np.repeat(img, scale, 0)
  img = np.repeat(img, scale, 1)
  return img

class VideoWriter:
  def __init__(self, filename, fps=30.0, **kw):
    self.writer = None
    self.params = dict(filename=filename, fps=fps, **kw)

  def add(self, img):
    img = np.asarray(img)
    if self.writer is None:
      h, w = img.shape[:2]
      self.writer = FFMPEG_VideoWriter(size=(w, h), **self.params)
    if img.dtype in [np.float32, np.float64]:
      img = np.uint8(img.clip(0, 1)*255)
    if len(img.shape) == 2:
      img = np.repeat(img[..., None], 3, -1)
    self.writer.write_frame(img)

  def close(self):
    if self.writer:
      self.writer.close()

  def __enter__(self):
    return self

  def __exit__(self, *kw):
    self.close()

def load_image(url, max_size=40):
  r = requests.get(url)
  img = PIL.Image.open(io.BytesIO(r.content))
  img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
  img = np.float32(img)/255.0
  # premultiply RGB by Alpha
  img[..., :3] *= img[..., 3:] # OMM original
  # img[..., :1] *= img[..., 1:] # OMM modified for 1 channel
  return img

def load_emoji(emoji):
  code = hex(ord(emoji))[2:].lower()
  url = 'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u%s.png'%code
  return load_image(url)


def to_rgba(x):
  # return x[..., :4] # OMM modified for 1 channel
  return x[..., :2] # OMM modified for 1 channel

def to_alpha(x):
  # return tf.clip_by_value(x[..., 3:4], 0.0, 1.0) # OMM modified for 1 channel
  return tf.clip_by_value(x[..., 1:2], 0.0, 1.0) # OMM modified for 1 channel

def to_rgb(x):
  # assume rgb premultiplied by alpha
  # rgb, a = x[..., :3], to_alpha(x) # OMM modified for 1 channel
  rgb, a = x[..., :1], to_alpha(x) # OMM modified for 1 channel
  return 1.0-a+rgb

def get_living_mask(x):
  # return tf.nn.max_pool2d
  return tf.nn.max_pool3d(to_alpha(x), 3, [1, 1, 1, 1, 1], 'SAME') > 0.1

def make_seed(size, CHANNEL_N, n=1):
  x = np.zeros([n, size, size, CHANNEL_N], np.float32)
  # x[:, size//2, size//2, 3:] = 1.0 # OMM modified for 1 channel
  x[:, size//2, size//2, 1:] = 1.0 # OMM modified for 1 channel
  return x

def to_RGBA(x):
  '''create an image of 4 channels where the first three channels are a copy of 
  the first channel of x and the last channel (alpha) is the last channel of x'''
  # create empty image
  empty_size = list(np.shape(x[...,0])) + list([4])
  empty = np.zeros(empty_size)
  slice0 = x[...,0]
  for i in range(3):
    empty[:,:,i] = to_rgba(x)[...,0]
  empty[:,:,-1] = x[...,-1]
  return empty
 
#=== train utilities
class SamplePool:
  def __init__(self, *, _parent=None, _parent_idx=None, **slots):
    self._parent = _parent
    self._parent_idx = _parent_idx
    self._slot_names = slots.keys()
    self._size = None
    for k, v in slots.items():
      if self._size is None:
        self._size = len(v)
      assert self._size == len(v)
      setattr(self, k, np.asarray(v))

  def sample(self, n):
    idx = np.random.choice(self._size, n, False)
    batch = {k: getattr(self, k)[idx] for k in self._slot_names}
    batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
    return batch

  def commit(self):
    for k in self._slot_names:
      getattr(self._parent, k)[self._parent_idx] = getattr(self, k)

#=== visualization 1 channel


def generate_pool_figures(pool, step_i, sz=32):
  tiled_pool = tile2d(np.squeeze(to_rgb(pool.x[:15,16]))) # :49 was changed to :15
  fade = np.linspace(1.0, 0.0, sz)
  ones = np.ones(sz) 
  tiled_pool[:, :sz] += (-tiled_pool[:, :sz] + ones[None, :]) * fade[None, :] 
  tiled_pool[:, -sz:] += (-tiled_pool[:, -sz:] + ones[None, :]) * fade[None, ::-1]
  tiled_pool[:sz, :] += (-tiled_pool[:sz, :] + ones[:, None]) * fade[:, None]
  tiled_pool[-sz:, :] += (-tiled_pool[-sz:, :] + ones[:, None]) * fade[::-1, None]
  imwrite('train_logB/%04d_pool.jpg'%step_i, tiled_pool)

# def visualize_batch(ca, step_i):
#   print(f'\nx0 = {np.shape(x0)}, x = {np.shape(x)}')
#   print(f'np.squeeze(x0) = {np.shape(np.squeeze(to_rgb(x0).numpy()))}')
#   print(f'np.squeeze(x) = {np.shape(np.squeeze(to_rgb(x).numpy()))}')
#   print(f'np.squeeze(x0)[:,16] = {np.shape(np.squeeze(to_rgb(x0).numpy())[:,16])}')
#   print(f'np.squeeze(x)[:,16] = {np.shape(np.squeeze(to_rgb(x).numpy())[:,16])}')
#   vis0 = np.hstack(np.squeeze(to_rgb(x0).numpy())[:,16])
#   vis1 = np.hstack(np.squeeze(to_rgb(x).numpy())[:,16])
#   print(f'vis0 = {np.shape(vis0)}, vis1 = {np.shape(vis1)}')
#   vis = np.vstack([vis0, vis1])
#   print(f'vis = {np.shape(vis)}')
#   imwrite('train_logB/batches_%04d.jpg'%step_i, vis)
#   export_model(ca, 'train_logB/%04d'%step_i)
#   clear_output()
#   print('batch (before/after):')
#   imshow(vis)
#   pl.figure(figsize=(10, 4))
#   pl.title('Loss history (log10)')
#   pl.plot(np.log10(loss_log), '.', alpha=0.1)
#   pl.show()

def export_model(ca, base_fn):
  ca.save_weights(base_fn)

  cf = ca.call.get_concrete_function(
      x=tf.TensorSpec([None, None, None, None, 16]),
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

#=== custom functions 3D plots
def plot_3d(image, threshold=-300, detail_speed=1, figsize=(6,6)):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(1,2,0)
    p = p.transpose(1,0,2)
    p = p[:,::-1,:]

    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold, step_size=detail_speed)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    face_color = [0.5, 0.5, 1]
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    plt.show()

def subplot_3d_two_figs(image, image2, global_fig, threshold=-300, detail_speed=1, detail_speed2=1, figsize=(6,6), sub=161, color=[.5, .5, 1], color2=[.5, 1, 0.5]):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(1,2,0)
    p = p.transpose(1,0,2)
    p = p[:,::-1,:]

    p2 = image2.transpose(1,2,0)
    p2 = p2.transpose(1,0,2)
    p2 = p2[:,::-1,:]

    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold, step_size=detail_speed)
    verts2, faces2, _, _ = measure.marching_cubes_lewiner(p2, threshold, step_size=detail_speed2)

    ax = global_fig.add_subplot(sub, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.3)
    face_color = color
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    # figure 2
    mesh2 = Poly3DCollection(verts2[faces2], alpha=0.99)
    face_color2 = color2
    mesh2.set_facecolor(face_color2)
    ax.add_collection3d(mesh2)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    return ax

def subplot_3d_one_fig(image, global_fig, threshold=-300, detail_speed=1, figsize=(6,6), sub=161, color=[.5, .5, 1]):
    # Position the scan upright, 
    # so the head of the patient would be at the top facing the camera
    p = image.transpose(1,2,0)
    p = p.transpose(1,0,2)
    p = p[:,::-1,:]

    verts, faces, _, _ = measure.marching_cubes_lewiner(p, threshold, step_size=detail_speed)

    ax = global_fig.add_subplot(sub, projection='3d')

    # Fancy indexing: `verts[faces]` to generate a collection of triangles
    mesh = Poly3DCollection(verts[faces], alpha=0.5)
    face_color = color
    mesh.set_facecolor(face_color)
    ax.add_collection3d(mesh)

    ax.set_xlim(0, p.shape[0])
    ax.set_ylim(0, p.shape[1])
    ax.set_zlim(0, p.shape[2])
    return ax

def call_subplots(imm, imm2=False, threshold=.1, detail_speed=2, color=[.5, .5, 1]):
  global global_fig
  width = int(len(imm)*4)
  global_fig = plt.figure(figsize=(width,6))
  imm_str = [int('1'+str(len(imm))+str(i)) for i in np.arange(1,len(imm)+1)]
  if imm2:
    for img, img2, i in zip(imm, imm2, imm_str):  
      subplot_3d_two_figs(img, img2, global_fig, threshold=threshold, detail_speed=detail_speed, figsize=(6,6), sub=i, color=color, color2=[1, .6, .6])
  else:
    for img, i in zip(imm, imm_str):
      subplot_3d_one_fig(img, global_fig, threshold=threshold, detail_speed=detail_speed, figsize=(6,6), sub=i, color=color)

#=== custom  figures

def figure_4_channels(ff, name='ff'):
  fig, ax = plt.subplots(1,4, figsize=(8,2))
  for idx in range(4):
    ax[idx].imshow(ff[:,:,idx])
  for axx in ax.ravel(): axx.axis('off')
  fig.suptitle(f'{name} {np.shape(ff)}')

def fig_multiple(ff, r=1, c=4, name='ff', vmin=0, vmax=1, figmul=1):
  fig, ax = plt.subplots(r,c, figsize=(c*2*figmul,r*2*figmul))
  for idx in range(r*c):
    if r <= 1:
      ax[idx].imshow(ff[idx])
    else:
      # print(idx,idx//c,idx%c)
      ax[idx//c,idx%c].imshow(ff[idx], vmin=0, vmax=1)
  for axx in ax.ravel(): axx.axis('off')
  fig.suptitle(f'{name}')
  fig.tight_layout()

def fig_multiple3D(ff, r=1, c=4, name='ff', vmin=0, vmax=1):
  '''Plot the middle slice of a 3D array.
  We assume that np.shape(ff) == B'''
  middle_slice = np.shape(np.squeeze(ff[0]))[0]//2 - 1
  fig, ax = plt.subplots(r,c, figsize=(c*2,r*2))
  for idx in range(r*c):
    if r <= 1:
      ax[idx].imshow(np.squeeze(ff[idx])[middle_slice,...])
    else:
      # print(idx,idx//c,idx%c)
      ax[idx//c,idx%c].imshow(np.squeeze(ff[idx])[middle_slice,...], vmin=0, vmax=1)
      # ax[idx//c,idx%c].text(5,10,f'{np.shape(ff[idx])}, {middle_slice}', color='y')
  for axx in ax.ravel(): axx.axis('off')
  # fig.suptitle(f'{name} {np.shape(ff)}')
  fig.tight_layout()

def print_3d_horiz(dx):
  '''https://stackoverflow.com/questions/58365744/python-3d-array-printing-horizontally'''
  j=0
  for i in range(len((dx[j]))):
      for j in range(len(dx)):        
              for k in range(len((dx[j][i]))):  
                  if k == 2: print(f'{dx[j][i][k]:+d}  ',end = ' ')
                  else: print(f'{dx[j][i][k]:+d}',end = ' ')
      print()

#=== place seed in corners
def place_seed_in_corners(ndl_pad_temp, coords = 0):
    '''Find the voxel from the nodule closest to a corner (coords)'''
    zz,yy,xx = np.where(ndl_pad_temp>0)
    dist_max = 100
    for (i,j,k) in zip(zz,yy,xx):
        dist = np.linalg.norm((i-coords,j-coords,k-coords))
        if dist < dist_max:
            dist_max = dist
            iq,jq,kq = i,j,k
    return iq,jq,kq

def largest_distance(ndl):
    aa = np.squeeze(ndl)
    print(np.shape(ndl))
    zz,yy,xx = np.where(aa > 0)
    point_cloud = np.asarray([zz,yy,xx]).T
    Y = distance.cdist(point_cloud,point_cloud)
    max_dist, _ = np.where(Y==np.max(Y))
    max_dist[0], max_dist[1]
    coord1 = point_cloud[max_dist[0]]
    coord2 = point_cloud[max_dist[1]]
    return coord1, coord2 
