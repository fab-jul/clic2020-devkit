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

def encoder(frame1, frame2):
    # Convert to long so that the substraction does not overflow
    residual_normalized = (frame2.astype(np.long) - frame1) // 2 + 127
    print(np.amax(residual_normalized), np.amin(residual_normalized))
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
    idxs = np.arange(len(ds))
    np.random.shuffle(idxs)
    for i in idxs:
        (y1, u1, v1), (y2, u2, v2) = ds[i]

        num_bytes = 0
        num_pixels = np.prod(y1.shape)

        for c1, c2 in zip((y1, u1, v1), (y2, u2, v2)):
            b = encoder(c1, c2)
            num_bytes += len(b)
            c2_decoded = decoder(c1, b)
            num_pixels_c = np.prod(c1.shape)
            ms_ssim_weight = 1 # todo check for MS-SSIM code

    # TODO output metrics


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir')
    flags = p.parse_args()

    compress_folder(flags.data_dir)


if __name__ == '__main__':
    main()
