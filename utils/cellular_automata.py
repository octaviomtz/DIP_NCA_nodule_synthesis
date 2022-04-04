import PIL
import requests
import io
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
from cv2 import VideoWriter
import PIL
from matplotlib.colors import to_rgba
import requests
import io
import matplotlib.pyplot as plt
import numpy as np

from scipy.ndimage import label
from scipy.ndimage import distance_transform_bf
from scipy.spatial import distance

def to_rgb(x):
    rgb, alpha = x[...,:3], np.clip(x[...,3:4],0,1)
    return 1-alpha+rgb

    import moviepy.editor as mvp
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

def load_image(url, max_size=40):
    r = requests.get(url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    img = np.float32(img)/255.0
    # premultiply RGB by Alpha
    img[..., :3] *= img[..., 3:]
    return img

def load_emoji(emoji):
    code = hex(ord(emoji))[2:].lower()
    url = 'https://github.com/googlefonts/noto-emoji/blob/main/png/128/emoji_u%s.png?raw=true'%code
    return load_image(url)

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

# def get_living_mask(x):
#       x = nn.max_pool(x[...,3:4], (3,3), padding='SAME')
#       return x > 0.1

def imread(url, max_size=None, mode=None):
    if url.startswith(('http:', 'https:')):
        # wikimedia requires a user agent
        headers = {
          "User-Agent": "Requests in Colab/0.0 (https://colab.research.google.com/; no-reply@google.com) requests/0.0"
        }
        r = requests.get(url, headers=headers)
        f = io.BytesIO(r.content)
    else:
        f = url
    img = PIL.Image.open(f)
    if max_size is not None:
        img.thumbnail((max_size, max_size), PIL.Image.ANTIALIAS)
    if mode is not None:
        img = img.convert(mode)
    img = np.array(img, dtype=np.float32)/255.0
    return img

def largest_distance(ndl):
    aa = np.squeeze(ndl)
    zz,yy,xx = np.where(aa > 0)
    point_cloud = np.asarray([zz,yy,xx]).T
    Y = distance.cdist(point_cloud,point_cloud)
    max_dist, _ = np.where(Y==np.max(Y))
    max_dist[0], max_dist[1]
    coord1 = point_cloud[max_dist[0]]
    coord2 = point_cloud[max_dist[1]]
    return coord1, coord2

def get_center_of_volume_from_largest_component_return_center_or_extremes(vol, ndl, extreme=0, coord='coord1'):
    '''get largest component and calculate center of irregular volume using distance transform and
    return those coords'''
    assert (extreme==0 or extreme==1)
    # get largest component
    mask_multiple, cc_num = label(vol > 0)
    sorted_comp = np.bincount(mask_multiple.flat)
    sorted_comp = np.sort(sorted_comp)[::-1]
    comp_largest = (mask_multiple == np.where(np.bincount(mask_multiple.flat) == sorted_comp[1])[0][0])
    if extreme == 0:
        # calculate center of irregular volume using distance transform
        center_volume = distance_transform_bf(comp_largest.astype('float'))
        center_z,center_y,center_x = np.where(center_volume == np.max(center_volume))
        center_z,center_y,center_x = center_z[0],center_y[0],center_x[0]
    if extreme == 1:
        coord1, coord2 = largest_distance(ndl)
        if coord == 'coord2':
            center_z,center_y,center_x = coord2
        else:
            center_z,center_y,center_x = coord1

    return center_z,center_y,center_x

def fig_loss_and_synthesis(imgs_syn, losses, EPOCHS, save=False):
    fig = plt.figure(figsize=(24,8))
    gs = fig.add_gridspec(4,11)
    count=0
    for r in range(4):
        for c in range(8):
            ax_ = fig.add_subplot(gs[r, c+3])
            ax_.imshow(imgs_syn[count])
            ax_.axis('off')
            count += 2
    ax2 = fig.add_subplot(gs[:4, :4])
    ax2.semilogy(losses, label=f'epochs = {EPOCHS}')
    ax2.legend(fontsize=14)
    ax2.set_xlabel('epochs', fontsize=14)
    ax2.set_ylabel('MSE', fontsize=14)
    ax2.tick_params(axis='x', labelsize=14)
    ax2.tick_params(axis='y', labelsize=14)
    plt.show()