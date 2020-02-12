import pframe_dataset_shared
import tensorflow as tf


def frame_sequence_dataset(data_root, merge_channels=False, num_frames_per_sequence=2):
    """
    Create a tf.data.Dataset that yields sequences of `num_frames` frames, e.g.:
        first element:  ( (f11_y, f11_u, f11_v), (f12_y, f12_u, f12_v) ),  # tuple for video 1, frame 1, 2
        second element: ( (f12_y, f12_u, f12_v), (f13_y, f13_u, f13_v) ),  # tuple for video 1, frame 2, 3

    If merge_channels=True, the channels are merged into one tensor, yielding
        first element:  ( f11, f12 ),  # for video 1, frame 1, 2
        second element: ( f12, f13 ),  # for video 1, frame 2, 3

    Dataformat is always HWC, and dtype is uint8, output is in {0, ..., 127} (non-normalized).
    """
    tuple_ps = pframe_dataset_shared.get_paths_for_frame_sequences(data_root, num_frames_per_sequence)
    tuple_ps = tf.constant(tuple_ps)
    d = tf.data.Dataset.from_tensor_slices(tuple_ps)
    d = d.map(load_frames)
    if merge_channels:
        d = d.map(merge_channels_per_frame)
    return d


def load_frames(ps):
    # Unpack ps = ((f11_y, f11_u, f11_v), (f12_y, f12_u, f12_v))
    return tuple(tuple(load_frame(p) for p in tf.unstack(frames))
                 for frames in tf.unstack(ps))


def load_frame(p):
    img = tf.image.decode_png(tf.io.read_file(p))
    img = tf.ensure_shape(img, (None, None, 1))
    # img.set_sh = (None, None, 1)
    return img


def merge_channels_per_frame(*frames):
    return tuple(yuv_420_to_444(y, u, v) for y, u, v in frames)


def yuv_420_to_444(y, u, v):
    """ Convert Y, U, V, given in 420, to RGB 444. """
    target_shape = tf.shape(y)[:2]  # Get H, W
    u = _upsample_nearest_neighbor(u, target_shape)
    v = _upsample_nearest_neighbor(v, target_shape)
    return tf.concat([y, u, v], -1)                 # merge


def _upsample_nearest_neighbor(t, target_shape):
    """ Upsample tensor `t`, given in H,W,C, to shape `target_shape`. """
    return tf.image.resize(
        t, target_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

