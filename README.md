# clic2020-devkit

## Downloading data

*_Upcoming_*

## P-Frame Baseline

*_Upcoming_*

## P-Frame Dataloading

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

ds_422 = ds_torch.FramePairsDataset(data_root='data')
ds_444 = ds_torch.FramePairsDataset(data_root='data', merge_channels=True)
```

### TensorFlow Example

Code tested in eager and graph mode, in TensorFlow 1.15. _TODO_ Test in TensorFlow 2.0.

```python 
import pframe_dataset_tf as ds_tf

ds_422 = ds_tf.frame_pairs_dataset(data_root='data')
ds_444 = ds_tf.frame_pairs_dataset(data_root='data', merge_channels=True)
```


