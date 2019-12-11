import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F
import pframe_dataset_shared
import numpy as np


class YUVFramesDataset(Dataset):
    """
    Yields frames either as tuples (Y, U, V) or, if merge_channels=True, as a single tensor (YUV).
    Dataformat is always torch default, CHW, and dtype is float32, output is in [0, 1]
    """
    def __init__(self, data_root, merge_channels=False, num_frames=2):
        """
        :param data_root:
        :param merge_channels:
        """
        self.tuple_ps = pframe_dataset_shared.get_frame_tuple_paths(data_root, num_frames_per_tuple=num_frames)
        self.merge_channels = merge_channels
        self.image_to_tensor = lambda pic: image_to_tensor(pic, normalize=True)

    def __len__(self):
        return len(self.tuple_ps)

    def __getitem__(self, idx):
        # this is a tuple of tuple, e.g.,
        #    ( (f21_y, f21_u, f21_v), (f22_y, f22_u, f22_v) )
        frame_seq = self.tuple_ps[idx]
        return tuple(self.load_frame(y_p, u_p, v_p)
                     for y_p, u_p, v_p in frame_seq)

    def load_frame(self, y_p, u_p, v_p):
        y, u, v = (self.image_to_tensor(Image.open(p)) for p in (y_p, u_p, v_p))
        if not self.merge_channels:
            return y, u, v
        yuv = yuv_420_to_444(y, u, v)
        return yuv


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
