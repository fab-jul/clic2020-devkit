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

import argparse
import time
from io import BytesIO
import os
import glob
from PIL import Image
import numpy as np

import pframe_dataset_shared


EXTENSION = 'baseline.jpg'


def encoder(frame1, frame2):
    # Convert to long so that the subtraction does not overflow
    residual_normalized = (frame2.astype(np.long) - frame1) // 2 + 127
    # Convert back to uint8
    residual_normalized = residual_normalized.astype(np.uint8)
    f = BytesIO()
    Image.fromarray(residual_normalized).save(f, format='jpeg')

    return f.getvalue()


def decoder(frame1, frame2_compressed: bytes):
    residual_normalized = np.array(Image.open(BytesIO(frame2_compressed)))
    return frame1 + (residual_normalized - 127) * 2


def decode(p):
    """
    Assumes that the input frame corresponding to `p` is in the current working directory.
    The output will be saved in the current working directory.
    """
    assert p.endswith('.' + EXTENSION)
    p2 = os.path.basename(p).replace('.' + EXTENSION, '.png')
    p1 = pframe_dataset_shared.get_previous_frame_path(p2)
    assert os.path.isfile(p1), (p2, p1, p, len(glob.glob('*.png')))
    with open(p, 'rb') as f_in:
        b = f_in.read()
    f2_reconstructed = decoder(np.array(Image.open(p1)), b)
    Image.fromarray(f2_reconstructed).save(p2)


def compress_folder(data_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # These folders exists if validation.zip was downloaded.
    inputs = os.path.join(data_dir, 'inputs')
    targets = os.path.join(data_dir, 'targets')
    assert os.path.isdir(inputs), inputs
    assert os.path.isdir(targets), targets

    inputs = sorted(glob.glob(os.path.join(inputs, '*.png')))
    targets = sorted(glob.glob(os.path.join(targets, '*.png')))
    assert len(inputs) == len(targets)
    assert len(inputs) > 0, 'No inputs!'

    N = len(inputs)
    start = time.time()
    for count, (p1, p2) in enumerate(zip(inputs, targets)):
        p2_expected = pframe_dataset_shared.get_frame_path(p1, offset=1)
        assert os.path.basename(p2_expected) == os.path.basename(p2), (p1, p2)
        i1, i2 = np.array(Image.open(p1)), np.array(Image.open(p2))

        p_out = os.path.join(output_dir, os.path.splitext(os.path.basename(p2))[0] + '.' + EXTENSION)
        with open(p_out, 'wb') as f_out:
            f_out.write(encoder(i1, i2))

        if count > 0 and count % 50 == 0:
            elapsed = time.time() - start
            per_img = elapsed / count
            remaining = (N - count) * per_img
            print('\rWrote {}/{} files. Time: {:.1f}s // {:.3e} per img // ~{:.1f}s remaining'.format(
                    count, N, elapsed, per_img, remaining), end='', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('data_dir')
    p.add_argument('output_dir')
    flags = p.parse_args()

    compress_folder(os.path.expanduser(flags.data_dir),
                    os.path.expanduser(flags.output_dir))


if __name__ == '__main__':
    main()
