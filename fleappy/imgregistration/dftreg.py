from skimage.feature import register_translation
import numpy as np
import cv2
from scipy.ndimage.interpolation import shift
from pathlib import Path


def register(avg_img, tiff_stack):
    """ Register tif stack using 

    Args:
        avg_img (numpy.ndarray): Template Image (y, x).
        tiff_stack (numpy.ndarray): Time series data to be registered (z, y, x).

    Returns:
        transform_spec (np array float (y,x): pixel shifts to register tiff_stack 
    """

    avg_img = avg_img.astype(np.float32)
    kernel = np.ones((10, 10), np.float32)/100
    transform_spec = np.zeros((tiff_stack.shape[0], 2))
    for idx, frame in enumerate(tiff_stack):
        frame = cv2.filter2D(frame.astype(np.float32), -1, kernel)

        transform_spec[idx, :], _, _ = register_translation(frame, avg_img, upsample_factor=5)

    return transform_spec


def transform(img_stack, transform_spec):
    """Applies (y,x) translation to a series of images

    Args:
        img_stack (numpy array):  Uncorrected tiff stack (z, y, x)
        transform_spec (numpy array): Shifts to be applied to tiff stack in format (frameNum, (y,x))

    Returns:
        numpy.ndarray: Motion corrected tiff stack (z, y, x)
    """

    z, w, h = img_stack.shape
    registered = np.zeros((z, w, h))
    for idx, frame in enumerate(img_stack):
        registered[idx, :, :] = shift(frame, transform_spec[idx, :])
    return registered.astype(np.int16)


def join(transform_list, transform_spec):
    """Appends the next set of transformations to a previous set.

    Args:
        transform_list (numpy.ndarray): Next set of frame by frame transformations.
        transform_spec (numpy.ndarray): Previous frame by frame transformations.

    Raises:
        ValueError: If the new transform_list isn't of a (n,2) numpy array.

    Returns:
        numpy.ndarray: Joined transformation specification.
    """

    if transform_list is None:
        return transform_spec
    elif isinstance(transform_list. np.ndarray) and transform_list.shape[1] == 2:
        return np.concatenate((transform_list, transform_spec), axis=0)
    else:
        raise ValueError('The transform list isn\'t a numpy array of dimensions (n,2)!')


def save(transform_list, target: Path):
    """Write the list of transformations to a file.

    Args:
        transform_list ([type]): [description]
        target (Path): [description]
    """

    np.savetxt(target, np.squeeze(transform_list), delimiter=',', fmt='%.3f', header=__name__)


def load():
    """Load frame by frame transformations from file.

    TODO:
        * Implement dftreg file loading.

    Raises:
        NotImplementedError: File loading is not yet supported
    """
    raise NotImplementedError('Transformation loading is not yet implemented.')


def create_template(img_stack):
    """Creates a template from the image stack.

    Args:
        img_stack (numpy.ndarray): image stack (t, y, x)

    Returns:
        numpy.ndarray: Mean Image 
    """

    return np.mean(img_stack, axis=0, dtype=np.float)
