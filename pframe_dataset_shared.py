import glob
import argparse
import os


def get_yuv_globs(data_root):
    """
    Expected structure:
    `root_dir`/
      video1/
        video1_frame1_y.png
        video1_frame1_u.png
        video1_frame1_v.png
        video1_frame2_y.png
        ...
      video2/
        ...
    """
    subfolder_glob = os.path.join(data_root, '*')
    y_glob, u_glob, v_glob = (os.path.join(subfolder_glob, '*' + suffix)
                              for suffix in ('_y.png', '_u.png', '_v.png'))
    return y_glob, u_glob, v_glob


def validate_data(data_root):
    globs = get_yuv_globs(data_root)
    files = tuple(glob.glob(g) for g in globs)
    assert len(files[0]) > 0, 'No files found in {}'.format(data_root)
    assert len(set(map(len, files))) == 1, 'Expected y, u, v file for every frame! Got {}'.format(list(map(len, files)))
    print('Found {} frames, and Y, U, V for each'.format(len(files[0])))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--validate', metavar='DATA_ROOT_FOLDER')
    flags = p.parse_args()
    if flags.validate:
        validate_data(flags.validate)


if __name__ == '__main__':
    main()
