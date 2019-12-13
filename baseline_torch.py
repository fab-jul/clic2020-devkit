#!/usr/bin/env python3

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

# TODO: more like baseline_np.py actually

import argparse
import time
from io import BytesIO
from PIL import Image
import os
import glob
import pframe_dataset_shared

import numpy as np
# import ms_ssim_np


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


def decode(p):
    assert p.endswith('.baseline')
    p2 = os.path.splitext(os.path.basename(p))[0] + '.png'
    p1 = pframe_dataset_shared.get_previous_frame_path(p2)
    assert os.path.isfile(p1)
    with open(p, 'rb') as f_in:
        b = f_in.read()
    f2_reconstructed = decoder(np.array(Image.open(p1)), b)
    Image.fromarray(f2_reconstructed).save(p2)


def compress_folder(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    inputs = os.path.join(data_dir, 'inputs')
    targets = os.path.join(data_dir, 'targets')
    assert os.path.isdir(inputs), inputs
    assert os.path.isdir(targets), targets

    inputs = sorted(glob.glob(os.path.join(inputs, '*.png')))
    targets = sorted(glob.glob(os.path.join(targets, '*.png')))
    assert len(inputs) == len(targets)
    assert len(inputs) > 0

    N = len(inputs)
    start = time.time()
    for count, (p1, p2) in enumerate(zip(inputs, targets)):
        p2_expected = pframe_dataset_shared.get_frame_path(p1, offset=1)
        assert os.path.basename(p2_expected) == os.path.basename(p2), (p1, p2)
        i1, i2 = np.array(Image.open(p1)), np.array(Image.open(p2))
        # (y1, u1, v1), (y2, u2, v2) = ...

        p_out = os.path.join(output_dir, os.path.splitext(os.path.basename(p2))[0] + '.baseline')
        with open(p_out, 'wb') as f_out:
            f_out.write(encoder(i1, i2))

        if count > 0 and count % 50 == 0:
            elapsed = time.time() - start
            per_img = elapsed / count
            remaining = (N - count) * per_img
            print('\rWrote {}/{} files. Time: {:.1f}s // {:.3e} per img // ~{:.1f}s remaining'.format(
                    count, N, elapsed, per_img, remaining), end='', flush=True)
        continue

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
    p.add_argument('output_dir')
    flags = p.parse_args()

    compress_folder(flags.data_dir, flags.output_dir)


if __name__ == '__main__':
    main()
