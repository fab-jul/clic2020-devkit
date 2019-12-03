import glob
import sys
import argparse
import os
import re


_SUFFIXES = ('_y.png', '_u.png', '_v.png')


def get_yuv_globs(data_root):
    """
    Expected structure:
    `root_dir`/
      video1_frame1_y.png
      video1_frame1_u.png
      video1_frame1_v.png
      video1_frame2_y.png
      ...
      video2_frame1_y.png
      video2_frame1_u.png
      video2_frame1_v.png
      video2_frame2_y.png
      ...

    :returns globs (as strings) for Y, U, V frames
    """
    y_glob, u_glob, v_glob = (os.path.join(data_root, '*' + suffix)
                              for suffix in _SUFFIXES)
    return y_glob, u_glob, v_glob


def validate_data(data_root):
    """ Check if for every frame we have Y, U, V files. """
    globs = get_yuv_globs(data_root)
    all_ps = tuple(sorted(glob.glob(g)) for g in globs)
    assert len(all_ps[0]) > 0, 'No files found in {}'.format(data_root)
    # get a set of prefixes for each Y, U, V by replacing the suffix of each path
    ys_pre, us_pre, vs_pre = (set(re.sub(suffix + '$', '', p) for p in ps)
                              for suffix, ps in zip(_SUFFIXES, all_ps))
    if len(ys_pre) == len(us_pre) == len(vs_pre):
        print('Found {} frames, and Y, U, V for each'.format(len(ys_pre)))
        return 0

    all_frames = ys_pre | us_pre | vs_pre
    ys_missing, us_missing, vs_missing = (all_frames - pre for pre in (ys_pre, us_pre, vs_pre))

    print('ERROR:\nMissing Y for: {}\nMissing U for: {}\nMissing V for: {}'.format(
            ys_missing or '-', us_missing or '-', vs_missing or '-'))
    return 1

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validate', metavar='DATA_ROOT_FOLDER')
    flags = p.parse_args()
    if flags.validate:
        sys.exit(validate_data(flags.validate))


if __name__ == '__main__':
    main()
