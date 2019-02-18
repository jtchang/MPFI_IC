"""Class Definition for Two-Photon Imaging metadata
"""

import os
from pathlib import Path
import numpy as np
from fleappy.metadata.basemetadata import BaseMetadata
import math


class TPMetadata(BaseMetadata):
    """Two-Photon Imaging metadata.

    Class for handling Two-Photon imaging sessions. Extends the BaseMetadata class.

    Attributes:
        imaging (dict): Dictionary for two-photon imaging information including times.
    """

    __slots__ = ['imaging']

    def __init__(self, path=None, expt_id=None, **kwargs):
        BaseMetadata.__init__(self, path=path, expt_id=expt_id, **kwargs)
        self.imaging = {'times': np.empty(0,)}
        self.load_two_photon()
        self.load_stims()

    def __str__(self):
        str_ret = f'{self.__class__.__name__}: {os.linesep}'
        str_ret = str_ret + f'{BaseMetadata.__str__(self)}{os.linesep}'
        str_ret = str_ret + f'imaging: {self.imaging}{os.linesep}'
        return str_ret + '>'

    def load_two_photon(self, override_file: str = None):
        """Load frame triggers.
            override_file (str, optional): Defaults to None. Filepath as string to a file of 2p frame triggers.
        """

        if override_file == None:
            filepath = Path(self.expt['path'], self.expt['expt_id'], 'twophotontimes.txt')
        else:
            filepath = Path(override_file)

        tp_time_file = open(filepath, 'r')
        self.imaging['times'] = np.array(tp_time_file.readlines()[
                                         0].rstrip().split(' ')).astype(np.float)

    def find_frame_idx(self, timestamp: float):
        """Find the closest frame trigger for associated time.

        Args:
            timestamp (float): Target time to look for.

        Returns:
            (int, float): Closest two-photon frame idx, Closest two-photon frame time
        """

        idx = np.searchsorted(self.imaging['times'], timestamp)
        if idx > 0 and idx == len(
                self.imaging['times']) or math.fabs(
                timestamp - self.imaging['times'][idx - 1]) < math.fabs(
                timestamp - self.imaging['times'][idx]):
            return (idx-1, self.imaging['times'][idx-1])
        else:
            return (idx, self.imaging['times'][idx])

    def frame_rate(self)->float:
        """Get two-photon imaging frame rate.
        Returns:
            float: frames per second
        """

        return 1/np.mean(np.diff(self.imaging['times']))
