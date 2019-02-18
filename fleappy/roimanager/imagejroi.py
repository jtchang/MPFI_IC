""" Collection of functions to handle conversions of ImageJ ROIs.

Depends heavily on the `read_roi <https://pypi.org/project/read-roi/>` library.

"""


from read_roi import read_roi_zip
from matplotlib.path import Path as MplPath
import numpy as np
from pathlib import Path
from . import nproi
import imageio

DEFAULT_FRAME_SIZE = (512, 512)
"""tuple: Frame size in pixels to be used for generating roi masks."""


def _fill_contour(roi):
    return np.maximum.accumulate(roi, 1) & np.maximum.accumulate(roi[:, ::-1], 1)[:, ::-1]


def to_array(roi_value, framesize: tuple = DEFAULT_FRAME_SIZE)->np.ndarray:
    """Convert ImageJ roi to numpy array mask.

    Args:
        roi_value (ImageJ ROI): ImageJ Roi
        framesize (tuple, optional): Defaults to DEFAULT_FRAME_SIZE. Frame size to use (y,x) should be type int.

    Returns:
        numpy.ndarray: [description]
    """

    x, y = np.meshgrid(np.arange(framesize[0]), np.arange(framesize[1]))
    x, y = x.flatten(), y.flatten()
    points = np.vstack((x, y)).T

    pth = MplPath(list(zip(roi_value['x'], roi_value['y'])))
    grid = pth.contains_points(points)
    mask = grid.reshape(framesize[1], framesize[0])
    return mask


def to_stack(rois: dict, framesize: tuple = DEFAULT_FRAME_SIZE):
    """Convert dictionary or ImageJ roi to numpy stack of masks.

    Args:
        rois (dict): ImageJ rois from zip file.
        framesize (tuple, optional): Defaults to DEFAULT_FRAME_SIZE. Frame size to use (y,x) should be type int.

    Returns:
        list, numpy.ndarray: list of roi names, numpy array of masks (# cell , y , x)
    """

    tiffstack = np.zeros(
        (len(rois.keys()), framesize[0], framesize[1]), dtype=np.bool)

    for idx, (roi_name, roi_value) in enumerate(rois.items()):
        tiffstack[idx, :, :] = to_array(roi_value, framesize=framesize)
    names = list(rois.keys())
    return names, tiffstack


def zip_to_tif(filesource: str, filetarget: str, framesize: tuple = DEFAULT_FRAME_SIZE):
    """Open a .zip of imagej rois and write them to a tif file

    Opens imagej rois in \*.zip and writes them to a tif file. Currently does not save the ROI names.

    Args:
        filesource(str): File path for ImageJ rois as a zip file
        filetarget(str): File path to write tif stack of ROIS
        framesize(tuple, optional): Defaults to DEFAULT_FRAME_SIZE. [description]

    Returns:
        None

    TODO:
        * imageio tiff writing description seems to break- should write names of ROI as metadata for tiff
    """
    assert isinstance(filesource, str) and filesource.endswith(
        '.zip') and '\\' not in filesource, 'Specify file source as a .zip file as a string using unix style!'
    assert isinstance(filetarget, str) and filetarget.endswith(
        '.tif') and '\\' not in filetarget, 'Specify file target as a .tif file as a string using unix style!'

    names, tiffstack = to_stack(read_roi_zip(Path(filesource)))

    imageio.mimwrite(Path(filetarget), tiffstack.astype(np.uint8))
    with open(filetarget+'.names', 'w') as f:
        f.write(';'.join(names))

    return None
