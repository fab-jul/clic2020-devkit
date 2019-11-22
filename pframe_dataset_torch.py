import torch
from PIL import Image
import glob
from torch.utils.data.dataset import Dataset
from torch.nn import functional as F
from torchvision.transforms.functional import to_tensor
import pframe_dataset_shared


class YUVFramesDataset(Dataset):
    """ Yields frames either as tuples (Y, U, V) or, if merge_channels=True, as a single tensor (YUV).
    Dataformat is always torch default, CHW, and dtype is float32, output is in [0, 1] """
    def __init__(self, data_root, merge_channels=False):
        self.frame_ps = YUVFramesDataset.get_frames_paths(data_root)
        self.merge_channels = merge_channels

    @staticmethod
    def get_frames_paths(data_root):
        """ :return a list of tuples, [(Y, U, V)]"""
        globs = pframe_dataset_shared.get_yuv_globs(data_root)
        ys, us, vs = (sorted(glob.glob(g)) for g in globs)
        return list(zip(ys, us, vs))

    def __len__(self):
        return len(self.frame_ps)

    def __getitem__(self, idx):
        y, u, v = (to_tensor(Image.open(p)) for p in self.frame_ps[idx])
        if not self.merge_channels:
            return y, u, v
        yuv = yuv_422_to_444(y, u, v)
        return yuv


class FramePairsDataset(Dataset):
    def __init__(self, data_root, merge_channels=False):
        self.yuv_frames_dataset = YUVFramesDataset(data_root, merge_channels)

    def __getitem__(self, idx):
        frame_1 = self.yuv_frames_dataset[idx]
        frame_2 = self.yuv_frames_dataset[idx + 1]
        return frame_1, frame_2

    def __len__(self):
        # substract one because we always look at tuples, final one is (N-1, N)
        return len(self.yuv_frames_dataset) - 1


def yuv_422_to_444(y, u, v):
    """ Convert Y, U, V, given in 422, to RGB 444. Expects CHW dataformat """
    u, v = map(_upsample_nearest_neighbor, (u, v))  # upsample U, V
    return torch.cat((y, u, v), dim=0)    # merge


def _upsample_nearest_neighbor(t, factor=2):
    """ Upsample tensor `t` by `factor`. """
    return F.interpolate(t.unsqueeze(0), scale_factor=factor, mode='nearest').squeeze(0)

