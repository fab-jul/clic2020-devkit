import pframe_dataset_shared
import tensorflow as tf


def frame_pairs_dataset(data_root, merge_channels=False):
    """
    Create a tf.data.Dataset that yields frames either as tuples (Y, U, V) or, if merge_channels=True,
    as a single tensor (YUV).

    Dataformat is always HWC, and dtype is uint8, output is in {0, ..., 127} (non-normalized).
    """
    # all frames, i.e., frames 0, 1, 2, 3, ...
    d_a = yuv_dataset(data_root)
    # same, but skip first frame, i.e., frames 1, 2, 3, ...
    d_b = yuv_dataset(data_root).skip(1)
    if merge_channels:
        d_a = d_a.map(yuv_422_to_444)
        d_b = d_b.map(yuv_422_to_444)
    # zip together, so we get pairs (0, 1), (1, 2), (2, 3), ...
    return tf.data.Dataset.zip((d_a, d_b))


def yuv_dataset(data_root):
    y, u, v = (tf.data.Dataset.list_files(glob, shuffle=False).map(load_frame)
               for glob in pframe_dataset_shared.get_yuv_globs(data_root))
    return tf.data.Dataset.zip((y, u, v))


def yuv_422_to_444(y, u, v):
    """ Convert Y, U, V, given in 422, to RGB 444. """
    u, v = map(_upsample_nearest_neighbor, (u, v))  # upsample U, V
    return tf.concat([y, u, v], -1)                 # merge


def _upsample_nearest_neighbor(t, factor=2):
    """ Upsample tensor `t`, given in H,W,C, by `factor`. """
    t_shape = t.shape.as_list()  # expected to be H, W, C
    assert len(t_shape) == 3
    new_shape = tf.shape(t)[:2]  # just get H, W
    new_shape *= tf.constant([factor, factor])
    return tf.image.resize(t, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def load_frame(p):
    return tf.image.decode_png(tf.io.read_file(p))

