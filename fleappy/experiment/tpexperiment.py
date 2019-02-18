import logging
import os
from itertools import chain
from pathlib import Path

import fleappy.analysis
from fleappy.metadata import TPMetadata
from fleappy.experiment import BaseExperiment
from fleappy.experiment import baselinefunctions
from fleappy.roimanager import Roi
from fleappy.roimanager import nproi, imagejroi
import natsort as ns
import numpy as np
import skimage.io as io
from scipy.sparse import csr_matrix


class TPExperiment(BaseExperiment):
    """Two-Photon experient class.

    Class for handling two-photon experiments. Extends the BaseExperiment class.

    Attributes:
        roi (list): List of ROI objects.
    """

    __slots__ = ['roi']

    def __init__(self, path: str, expt_id: str, **kwargs):
        self.roi = []
        BaseExperiment.__init__(self)
        self.metadata = TPMetadata(path=path, expt_id=expt_id)

    def __str__(self):
        str_ret = f'{self.__class__.__name__}: {os.linesep}'
        for key in chain.from_iterable(getattr(cls, '__slots__', []) for cls in TPExperiment.__mro__):
            if key in ['roi']:
                str_ret = str_ret + \
                    f'{key}:{len(getattr(self, key))}{os.linesep}'
            else:
                str_ret = str_ret + f'{key}:{getattr(self, key)}{os.linesep}'

        return str_ret

    def load_roi(self):
        """Load roi from tif filed.

        Looks for rois in tif file in default path. If can not be found, looks for .zip file of ImageJ roi and converts 
        them to a tif file. Loads each roi into experiment roi array.

        Raises:
            OSError: ROI files (.zip and .tif) are not available
        """

        slice_id = 'slice1'
        roi_path = self._roi_path(slice_id=slice_id)
        logging.debug(roi_path)
        if roi_path.exists():
            rois = nproi.load_from_file(roi_path)
        else:
            zip_path = self._zip_path(slice_id=slice_id)
            if zip_path.exists():
                imagejroi.zip_to_tif(str(zip_path), str(roi_path))
                rois = nproi.load_from_file(roi_path)
            else:
                raise OSError('ROI files could not be found!')

        name_path = self._name_path(slice_id=slice_id)
        if name_path.exists():
            roi_names = np.loadtxt(name_path, dtype=str, delimiter=';')
        else:
            roi_names = []

        for idx, mask in enumerate(rois):
            roi_name = roi_names[idx] if len(
                roi_names) == rois.shape[0] else str(idx)

            roi = Roi(id=idx, name=roi_name, roi_type='primary',
                      mask=csr_matrix(mask))
            if roi not in self.roi:
                self.roi.append(roi)
            else:
                logging.debug(
                    'ROI#{0} already exists, skipping...'.format(roi.id))

    def load_ts_data(self):
        """Loads time series data based on the properties associated with the experiment.

           Load time series for roi preloaded and tif files specified in the file directory.
        """

        slice_id = 'slice1'
        tif_path = self._tif_path(slice_id)
        tif_files = ns.natsorted(
            list(tif_path.glob('stack_*.tif')), alg=ns.PATH)
        if len(self.roi) == 0:
            self.load_roi()

        rois = np.empty(
            (len(self.roi), self.roi[0].mask.shape[0], self.roi[0].mask.shape[1]), dtype=np.bool)
        for idx, roi in enumerate(self.roi):
            rois[idx, :, :] = roi.mask.todense()

        ts_data = np.empty((len(self.roi), 0))

        for ts_file in tif_files:
            logging.debug('Loading file: {0}'.format(ts_file.name))
            ts_temp = nproi.tseries_data(rois, io.imread(ts_file))
            ts_data = np.concatenate((ts_data, ts_temp), axis=1)

        for idx, data in enumerate(ts_data):
            self.roi[idx].ts_data['rawF'] = data

    def get_trial_responses(self, roi_id: int, field: str, prepad: float = 0, postpad: float = 0):
        """Returns single trial responses for a specified ROI.

        Args:
            roi_id (int): Desired roi # (0-N)
            field (str): Desired time series to chop into trial responses
            prepad (float, optional): Defaults to 0. Time to pad response before trial start
            postpad (float, optional): Defaults to 0. Time to pad response after trial end

        Returns:
            numpy.ndarray : Trial Responses (# stims x # trials x time)
        """

        stim_duration = self.metadata.stim_duration()
        frame_rate = self.metadata.frame_rate()

        window_length = int(
            np.round(prepad*frame_rate) + np.round(postpad * frame_rate) + np.round(stim_duration*frame_rate))

        trial_responses = np.empty(
            (self.metadata.num_stims(), self.metadata.num_trials(), window_length))

        if isinstance(self.roi[roi_id], Roi):
            for idx, (stim, time) in enumerate(self.metadata.stim['triggers']):
                frame_idx, _ = self.metadata.find_frame_idx(time)
                frame_idx = int(frame_idx - np.round(prepad*frame_rate))
                trial_responses[stim-1, int(np.floor(idx/self.metadata.num_stims(
                ))), :] = self.roi[roi_id].ts_data[field][frame_idx:frame_idx+window_length]
        return trial_responses

    def get_tseries(self, roi_id: int, field: str):
        """Returns the time series labeled with field for a roi.

        Args:
            roi_id (int): Desired roi # (0-N).
            field (str): Desired time series field.

        Returns:
            numpy.ndarray, numpy.ndarray: frame times, time series data
        """

        return self.metadata.imaging['times'], self.roi[roi_id].ts_data[field]

    def get_all_tseries(self, field: str):
        """Returns time series data for all roi.

        Args:
            field (str): Desires time series field.

        Returns:
            numpy.ndarray, numpy.ndarray: frame times, time series data (# cells x time)
        """

        if len(self.roi) is 0:
            return None
        times, roi_resp = self.get_tseries(0, field)

        responses = np.empty((len(self.roi), roi_resp.shape[0]))
        responses[0, :] = roi_resp
        for roi_id in range(1, len(self.roi)):
            _, responses[roi_id, :] = self.get_tseries(roi_id, field)

        return times, responses

    def get_all_trial_responses(self, field: str, prepad: float = 0, postpad: float = 0):
        """ Returns single trial responses for all ROI

        Args:
            field (str): Desired time series to chop into trial responses
            prepad (float, optional): Defaults to 0. Time to pad response before trial start
            postpad (float, optional): Defaults to 0. Time to pad response after trial end

        Returns:
            numpy.ndarry : Trial Responses ( # roi x # stims x # trials x time)
        """

        stim_duration = self.metadata.stim_duration()
        frame_rate = self.metadata.frame_rate()
        window_length = int(np.round(prepad*frame_rate) + np.round(postpad *
                                                                   frame_rate) + np.round(stim_duration*frame_rate))

        trial_responses = np.empty((self.num_roi(), self.metadata.num_stims(
        ), self.metadata.num_trials(), window_length))
        for idx in range(self.num_roi()):
            trial_responses[idx, :, :, :] = self.get_trial_responses(
                idx, field, prepad=prepad, postpad=postpad)
        return trial_responses

    def baseline_roi(self, field: str, target_field: str, baseline_func=baselinefunctions.percentile_filter, **kwargs):
        """Baseline roi time series

        Args:
            field (str): Desired time series to baseline.
            target_field (str): Time series name to save computed baseline
            baseline_func (function, optional): Defaults to baselinefunctions.percentile_filter. 
                Method used to compute the baseline
        """

        for idx in range(self.num_roi()):
            tseries = self.roi[idx].ts_data[field]
            self.roi[idx].ts_data[target_field] = baseline_func(
                self.roi[idx].ts_data[field], **kwargs)

    def compute_dff(self, field: str, baseline: str, target_field: str, clip_zero=True):
        """Compute Delta F/ F for timeseries

        Args:
            field (str): [description]
            baseline (str): [description]
            target_field (str): [description]
            clip_zero (bool, optional): Defaults to True. [description]
        """

        for idx in range(self.num_roi()):
            self.roi[idx].ts_data[target_field] = (
                self.roi[idx].ts_data[field]-self.roi[idx].ts_data[baseline])/self.roi[idx].ts_data[baseline]
            if clip_zero:
                self.roi[idx].ts_data[target_field][self.roi[idx].ts_data[target_field] < 0] = 0

    def num_roi(self):
        """Return the total number of ROI.

        Returns:
            int: Number of ROI associated with experiment.
        """

        return len(self.roi)

    def add_analysis(self, analysis_id: str, field: str, **kwargs):
        """Add an analysis to experiment.

        This function adds an analysis to the experiment. Currently it automatically adds an analysis of orientation of 
        the stimulus type is driftingGrating. 

        TODO: 
            * Let users overwrite the default analysis

        Args:
            analysis_id (str): Name for analysis set.
            field (str): Field to use analysis
        """

        analysis = None
        if self.metadata.stim_type() == 'driftingGrating':
            analysis = fleappy.analysis.OrientationAnalysis(
                self, analysis_id, field, **kwargs)
        if analysis is not None:
            analysis.run()
            self.analysis[analysis_id] = analysis

    def _tif_path(self, slice_id=1):
        return Path(self.metadata.expt['path'], self.metadata.expt['expt_id'], f'Registered/{slice_id}/')

    def _roi_path(self, slice_id=1):
        return Path(self.metadata.expt['path'], self.metadata.expt['expt_id'], f'Registered/{slice_id}_ROIs.tif')

    def _zip_path(self, slice_id=1):
        return Path(
            self.metadata.expt['path'],
            self.metadata.expt['expt_id'],
            f'Registered/{slice_id}/{slice_id}_ROIs.zip')

    def _name_path(self, slice_id=1):
        return Path(self.metadata.expt['path'], self.metadata.expt['expt_id'], f'Registered/{slice_id}_ROIs.tif.names')
