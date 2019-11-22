
import pframe_dataset_shared
import tensorflow as tf

def yuv_422_to_444(y, u, v):
    """ Convert Y, U, V, given in 422, to RGB 444. """
    # upsample U, V
    u, v = map(_upsample_nearest_neighbor, (u, v))
    # merge
    return tf.concat([y, u, v], -1)
    # return _merge_yuv_to_rgb(y, u, v)


# def _merge_yuv_to_rgb(y, u, v):
#     # convert uint8 to floats, where y is in [0,1] and u, v are in [-0.5, 0.5], because that's what yuv_to_rgb_wants
#     y, u, v = _yuv_uint8_to_float(y, u, v)
#     # concatenate to H, W, 3
#     yuv_full = tf.concat([y, u, v], -1)
#     # convert to RGB
#     rgb = tf.image.yuv_to_rgb(yuv_full)
#     # back to uint8
#     return tf.image.convert_image_dtype(rgb, tf.uint8)
#
#
# def _yuv_uint8_to_float(y, u, v):
#     assert y.dtype == tf.uint8 and u.dtype == tf.uint8 and v.dtype == tf.uint8
#     y = tf.to_float(y) / 255.
#     u = tf.to_float(u) / 255. - 0.5
#     v = tf.to_float(v) / 255. - 0.5
#     return y, u, v


def _upsample_nearest_neighbor(t, factor=2):
    """ Upsample tensor `t`, given in H,W,C, by `factor`. """
    t_shape = t.shape.as_list()  # expected to be H, W, C
    assert len(t_shape) == 3
    new_shape = tf.shape(t)[:2]  # just get H, W
    new_shape *= tf.constant([factor, factor])
    return tf.image.resize(t, new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)


def load_frame(p):
    return tf.image.decode_png(tf.io.read_file(p))


def yuv_dataset(data_root):
    y, u, v = (tf.data.Dataset.list_files(glob, shuffle=False).map(load_frame)
               for glob in pframe_dataset_shared.get_yuv_globs(data_root))
    return tf.data.Dataset.zip((y, u, v))


def frame_pairs_dataset(data_root, merge_channels=False):
    # all frames, i.e., frames 0, 1, 2, 3, ...
    d_a = yuv_dataset(data_root)
    # same, but skip first frame, i.e., frames 1, 2, 3, ...
    d_b = yuv_dataset(data_root).skip(1)
    if merge_channels:
        d_a = d_a.map(yuv_422_to_444)
        d_b = d_b.map(yuv_422_to_444)
    # zip together, so we get pairs (0, 1), (1, 2), (2, 3), ...
    return tf.data.Dataset.zip((d_a, d_b))
