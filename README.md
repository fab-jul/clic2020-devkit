# clic2020-devkit  

The following helps working with the **P-frame challenge**.

## Downloading data

To download all files, run:

```bash
bash download.sh path/to/data
```

It will create a folder `path/to/data` and extract all frames there, into a structure like:

```
video1/
    video1_frame1_y.png
    video1_frame1_u.png
    video1_frame1_v.png
    video1_frame2_y.png
    video1_frame2_u.png
    video1_frame2_v.png
    ...
video2/
    video2_frame1_y.png
    video2_frame1_u.png
    video2_frame1_v.png
    ...
```

For this, one of `gsutil`, `wget`, or `curl` must be available. `gsutil` is probably the most efficient way.

To download only some videos, use `--max_vides`: `bash download.sh path/to/data --max_videos 10`

**NOTE**: The script first downloads all vidoes as .zip files, resulting in 250GB+ of data.
Then all zips are decompressed one by one and subsequently deleted. If you interrupt the script
while unpacking, and later re-run it, it will re-download those that were already unpacked.
To prevent this at the expense of more hard-drive space used, you can keep the zip files by passing `--no_delete_zip`.

## P-Frame Baseline

We implement a simple non-learned baseline in `baseline_np.py`. The algorithm can be described as follows:

```
ENCODE:
  Inputs: frame 1 (F1) and frame 2 (F2).
  Encode F2 given F1:
  1. Calculate Residual_Normalized = (F2 - F1) // 2 + 127
     (this is now in range {0, ..., 255}.
      The idea of the //2 is to have 256 possible values, because
      otherwise we would have 511 values.)
  2. Compress Residual_Normalized with JPG
  -> Save to Bitstream
DECODE:
  Inputs: F1 and Bitstream
  1. Get Residual_Normalized from JPG in Bistream
  2. F2' = F1 + ( Residual_normalized - 127 ) * 2
     (F2' is the reconstruction)
```

The `run_baseline.sh` script describes how this would be used to create a submission to the challenge server. Running it produces a decoder.zip and valout.zip which could be uploaded to the `P-frame (validation)` track at [challenge.compression.cc](http://challenge.compression.cc). The reason `run_baseline.sh` compresses all decoder and data files using zip is to allow efficient uploads (some browsers hang if you try to upload 160000 files).

## P-Frame Dataloading Helpers

We have data loaders for PyTorch and TensorFlow. By default, they yield pairs of frames, where each frame is represented
as a tuple (Y, U, V). The dimensions of U and V are half the those of Y (420 format):

```
    [ ((Y1, U1, V1), (Y2, U2, V2))  # pair 1
      ((Y2, U2, V2), (Y3, U3, V3))  # pair 2
      ... ]
```

To get a single YUV tensor, we also provide a way to load merged YUV tensors (444 format):

```
    [ (YUV1, YUV2),
      (YUV2, YUV3),
      ...]
```

### PyTorch Example

```python
import pframe_dataset_tf as ds_torch

ds_420 = ds_torch.FrameSequenceDataset(data_root='data')
ds_444 = ds_torch.FrameSequenceDataset(data_root='data', merge_channels=True)
```

### TensorFlow Example

Code tested in eager and graph mode, in TensorFlow 1.15. _TODO_ Test in TensorFlow 2.0.

```python
import pframe_dataset_tf as ds_tf

ds_420 = ds_tf.frame_sequence_dataset(data_root='data')
ds_444 = ds_tf.frame_sequence_dataset(data_root='data', merge_channels=True)
```


