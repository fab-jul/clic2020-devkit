import itertools
import numpy as np
from matplotlib import pyplot as plt

import pframe_dataset_tf as ds_tf
import pframe_dataset_torch as ds_torch
import tensorflow as tf


def visualize(image_pair, image_pair_444):
    """
    :param image_pair: a 2-tuple of (frame_1, frame_2), where each frame is a 3-table (Y,U,V)
    :param image_pair_444: a 2-tuple of (frame_1, frame_2), where each frame is a tensor YUV
    :return:
    """
    f, axs = plt.subplots(2, 4, figsize=(10, 5))
    for yuv_channels, yuv_merged, ax_horizontal in zip(image_pair, image_pair_444, axs):
        for channel, ax in zip(yuv_channels, ax_horizontal):
            # transpose CHW -> HWC
            if channel.shape[0] == 1:
                channel = np.transpose(channel, (1, 2, 0))
            ax.imshow(channel[..., 0])
        # transpose CHW -> HWC
        if yuv_merged.shape[0] == 3:
            yuv_merged = np.transpose(yuv_merged, (1, 2, 0))
        ax_horizontal[-1].imshow(yuv_merged)
    f.show()


def show_torch():
    d = ds_torch.FramePairsDataset('data')
    d_merged = ds_torch.FramePairsDataset('data', merge_channels=True)

    for image_pair, image_pair_444 in itertools.islice(zip(d, d_merged), 5):
        visualize(image_pair, image_pair_444)

    # TODO
    # dl = DataLoader(d, batch_size=10, shuffle=True, num_workers=2)
    # for batch in dl:
    #     ...


def show_tf_eager():
    tf.enable_eager_execution()

    ds = ds_tf.frame_pairs_dataset('data')
    ds_merged = ds_tf.frame_pairs_dataset('data', merge_channels=True)

    for image_pair, image_pair_444 in itertools.islice(zip(ds, ds_merged), 5):
        visualize(image_pair, image_pair_444)


def show_tf_graph():
    tf.disable_eager_execution()
    d = ds_tf.frame_pairs_dataset('data')
    d_rgb = ds_tf.frame_pairs_dataset('data', merge_channels=True)

    it = d.make_one_shot_iterator()
    ne = it.get_next()

    it = d_rgb.make_one_shot_iterator()
    ne_rgb = it.get_next()

    with tf.Session() as sess:
        for _ in range(3):
            image_pair, image_pair_444 = sess.run([ne, ne_rgb])
            visualize(image_pair, image_pair_444)

def main():
    show_torch()
    show_tf_eager()
    show_tf_graph()


if __name__ == '__main__':
    main()
