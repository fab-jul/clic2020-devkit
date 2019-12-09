"""

Simple non-learned baseline.

Input: Frame 1, Frame 2 (F1, F2)

Algorithm:
    Encode
      1. Calculate Residual_Normalized = (F2 - F1) // 2 + 127
         (this is no in range {0, ..., 255}. The idea of the //2 is to have 256 possible values, because
          otherwise we would have 511 values.)
      2. Compress Residual_Normalized with JPG
      -> Bitstream
    Decode, given F1 and Bitstream
      1. get Residual_Normalized from JPG in Bistream
      2. F2' = F1 + ( Residual_normalized - 127 ) * 2
"""
import torch
import argparse
from io import BytesIO
from PIL import Image
from pframe_dataset_torch import FramePairsDataset
import numpy as np
import ms_ssim_np


_MSSSIM_WEIGHTS = (1/1.5, .25/1.5, .25/1.5)


def encoder(frame1, frame2):
    # Convert to long so that the substraction does not overflow
    residual_normalized = (frame2.astype(np.long) - frame1) // 2 + 127
    # print(np.amax(residual_normalized), np.amin(residual_normalized))
    # Convert back to uint8
    residual_normalized = residual_normalized.astype(np.uint8)
    f = BytesIO()
    Image.fromarray(residual_normalized).save(f, format='jpeg')
    return f.getvalue()


def decoder(frame1, frame2_compressed: bytes):
    residual_normalized = np.array(Image.open(BytesIO(frame2_compressed)))
    return frame1 + (residual_normalized - 127) * 2


def torch_image_to_numpy(t):
    """ Convert CHW normalized float32 (image) torch tensor to HWC numpy array """
    return t.mul(255).round().to(torch.uint8).permute(1, 2, 0).detach().cpu().numpy()


def compress_folder(data_dir):
    ds = FramePairsDataset(data_dir) #, merge_channels=True)
    ds.yuv_frames_dataset.image_to_tensor = lambda pic: np.array(pic)
    N = len(ds)
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    metrics = []
    for count, i in enumerate(idxs):
        (y1, u1, v1), (y2, u2, v2) = ds[i]

        num_bytes = 0
        num_pixels = np.prod(y1.shape)
        ms_ssim = 0

        for ms_ssim_weight, c1, c2 in zip(_MSSSIM_WEIGHTS, (y1, u1, v1), (y2, u2, v2)):
            b = encoder(c1, c2)
            num_bytes += len(b)
            c2_decoded = decoder(c1, b)
            ms_ssim_c = ms_ssim_np.MultiScaleSSIM(_batch(c1), _batch(c2_decoded))
            ms_ssim += ms_ssim_weight * ms_ssim_c

        bpp = num_bytes * 8 / num_pixels
        metrics.append((bpp, ms_ssim))
        print('{: 10d}/{}: {:.4f} bpp // {:.4f} weighted MS-SSIM'.format(count, N, bpp, ms_ssim))

    # TODO output metrics

def _batch(c):
    return np.expand_dims(np.expand_dims(c, -1), 0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir')
    flags = p.parse_args()

    compress_folder(flags.data_dir)


if __name__ == '__main__':
    main()
