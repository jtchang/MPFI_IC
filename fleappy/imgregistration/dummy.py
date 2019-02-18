from pathlib import Path
import numpy as np


def register(avg_img, tiff_stack):
    return []


def transform(img_stack, transform_spec):

    return img_stack


def join(transform_list, transform_spec):
    pass


def save(transform_list, target: Path):
    pass


def load():
    pass


def create_template(img_stack):
    return np.mean(img_stack, axis=0, dtype=np.float)
