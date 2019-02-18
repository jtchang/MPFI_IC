import os
import numpy as np
import fleappy.roimanager.nproi
from scipy.sparse import csr_matrix


class Roi(object):
    """ROI Class for handling cellular/subcellular ROI

    Attributes:
        id (str): Identifying string
        name (str): Name of roi
        type (str): Type of roi
        ts_data (dict): Dictionary of time series data.
        mask (scipy.sparse.csr.csr_matrix): Scipy sparse matrix with the associated ROI mask.

    """

    __slots__ = ['id', 'type', 'mask', 'ts_data', 'name']

    def __init__(self, id: str = None, roi_type: str = None, mask: csr_matrix = None, name=None):
        self.id = id
        self.name = name
        self.type = roi_type
        self.mask = mask
        self.ts_data = {}

    def __str__(self):
        ret_str = f'fleappy ROI object: {os.linesep}'
        ret_str = ret_str + f'id: {self.id}{os.linesep}'
        ret_str = ret_str + f'name: {self.name}{os.linesep}'
        ret_str = ret_str + f'type: {self.type}{os.linesep}'
        ret_str = ret_str + f'mask: Mask {self.mask.shape}{os.linesep}'
        ret_str = ret_str + f'ts_data: {list(self.ts_data.keys())}{os.linesep}'
        return ret_str

    def __eq__(self, other):
        if (self.mask.todense() != other.mask.todense()).any():
            return False
        return True

    def centroid(self)->tuple:
        """Returns the centroid of the roi

        Returns:
            tuple: (y,x) of centroid
        """

        return fleappy.roimanager.nproi.centroid(self.mask.toarray())
