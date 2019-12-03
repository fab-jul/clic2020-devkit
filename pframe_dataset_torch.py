import torch
from PIL import Image
import glob
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F
import pframe_dataset_shared
import numpy as np


class YUVFramesDataset(Dataset):
    """
    Yields frames either as tuples (Y, U, V) or, if merge_channels=True, as a single tensor (YUV).
    Dataformat is always torch default, CHW, and dtype is float32, output is in [0, 1]
    """
    def __init__(self, data_root, merge_channels=False):
        self.frame_ps = YUVFramesDataset.get_frames_paths(data_root)
        self.merge_channels = merge_channels
        self.image_to_tensor = lambda pic: image_to_tensor(pic, normalize=True)

    @staticmethod
    def get_frames_paths(data_root):
        """ :return a list of tuples, [(Y, U, V)]"""
        globs = pframe_dataset_shared.get_yuv_globs(data_root)
        ys, us, vs = (sorted(glob.glob(g)) for g in globs)
        return list(zip(ys, us, vs))

    def __len__(self):
        return len(self.frame_ps)

    def __getitem__(self, idx):
        y, u, v = (self.image_to_tensor(Image.open(p)) for p in self.frame_ps[idx])
        if not self.merge_channels:
            return y, u, v
        yuv = yuv_420_to_444(y, u, v)
        return yuv


class FramePairsDataset(Dataset):
    def __init__(self, data_root, merge_channels=False):
        self.yuv_frames_dataset = YUVFramesDataset(data_root, merge_channels)
        if len(self.yuv_frames_dataset) == 0:
            raise ValueError('No frames found in {}'.format(data_root))

    def __getitem__(self, idx):
        frame_1 = self.yuv_frames_dataset[idx]
        frame_2 = self.yuv_frames_dataset[idx + 1]
        return frame_1, frame_2

    def __len__(self):
        # substract one because we always look at tuples, final one is (N-1, N)
        return len(self.yuv_frames_dataset) - 1


def yuv_420_to_444(y, u, v):
    """ Convert Y, U, V, given in 420, to RGB 444. Expects CHW dataformat """
    u, v = map(_upsample_nearest_neighbor, (u, v))  # upsample U, V
    return torch.cat((y, u, v), dim=0)    # merge


def _upsample_nearest_neighbor(t, factor=2):
    """ Upsample tensor `t` by `factor`. """
    return F.interpolate(t.unsqueeze(0), scale_factor=factor, mode='nearest').squeeze(0)


def image_to_tensor(pic, normalize=True):
    """
    Convert a ``PIL Image`` to tensor.
    Copied from torchvision.transforms.functional.to_tensor, adapted
    to only support PIL inputs and normalize flag
    :param pic PIL Image
    :param normalize If False, return uint8, otherwise return float32 in range [0,1]
    """
    # handle PIL Image
    if pic.mode == 'I':
        img = torch.from_numpy(np.array(pic, np.int32, copy=False))
    elif pic.mode == 'I;16':
        img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    elif pic.mode == 'F':
        img = torch.from_numpy(np.array(pic, np.float32, copy=False))
    elif pic.mode == '1':
        img = 255 * torch.from_numpy(np.array(pic, np.uint8, copy=False))
    else:
        img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
    # PIL image mode: L, LA, P, I, F, RGB, YCbCr, RGBA, CMYK
    if pic.mode == 'YCbCr':
        nchannel = 3
    elif pic.mode == 'I;16':
        nchannel = 1
    else:
        nchannel = len(pic.mode)
    img = img.view(pic.size[1], pic.size[0], nchannel)
    # put it from HWC to CHW format
    # yikes, this transpose takes 80% of the loading time/CPU
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor) and normalize:
        return img.float().div(255)
    else:
        return img
