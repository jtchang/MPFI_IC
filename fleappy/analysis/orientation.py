import os
from itertools import chain
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import i0
from scipy.optimize import curve_fit
from fleappy.analysis import BlockwiseAnalysis


class OrientationAnalysis(BlockwiseAnalysis):
    """Analysis class for orientation tuned stimuli.

    Analysis for orientation stimuli where stimulus codes are directions equally spaced. This is a subclass of the 
    blockwise analysis.

    Attributes:
        metrics (pandas.dataframe): Collection of metrics

    """

    __slots__ = ['metrics']

    def __init__(self, expt, analysis_id, field, analysis_period=(0, -1), prepad=0, postpad=0):
        BlockwiseAnalysis.__init__(
            self, expt, analysis_id, field, analysis_period=(0, -1), prepad=0, postpad=0)
        self.metrics = pd.DataFrame()
        self.metrics['id'] = [r.name for r in self.expt.roi]

    def __setattr_(self, name, value):
        if hasattr(self, name):
            object.__setattr__(self, name, value)
        else:
            raise TypeError('Cannot set name %r on object of type %s' %
                            (name, self.__class__.__name___))

    def __str__(self):
        str_ret = f'{self.__class__.__name__}: {os.linesep}'
        for key in chain.from_iterable(getattr(cls, '__slots__', []) for cls in OrientationAnalysis.__mro__):
            if key is 'metrics':
                str_ret = str_ret + f'metrics: {self.metrics.columns}'
            else:
                str_ret = str_ret + f'{key}:{getattr(self, key)}{os.linesep}'
        return str_ret

    @staticmethod
    def _von_mises(kappa, mu, x):
        return np.exp(kappa * np.cos(x-mu)) / (2 * np.pi * i0(kappa))

    @staticmethod
    def _von_mises_or_fit(x, A, B, kappa, mu):
        return A * OrientationAnalysis._von_mises(kappa, mu, x) + B

    @staticmethod
    def _von_mises_dr_fit(x, Aa, Ab, B, kappa, mu):
        return Aa * OrientationAnalysis._von_mises(kappa, mu, x) + Ab * OrientationAnalysis._von_mises(kappa, mu+np.pi, x) + B

    def run(self):
        self.metrics['orientation'] = (
            (np.rad2deg(np.angle(self.vector_sum_responses()))+360) % 360) / 2

    def vector_sum_responses(self, orientation=True):
        """Calculates the vector sum for all roi.

        Args:
            orientation (bool, optional): Defaults to True. Collapse direction to orientation.

        Returns:
            [np.ndarray]: Complex vector sums for responses
        """

        responses = self.single_trial_responses()
        correction_factor = 2 if orientation else 1
        if self.expt.metadata.stim['doBlank'] is '1':
            responses = responses[:, :-1, :]
        if orientation:
            responses = np.concatenate((responses[:, :int(responses.shape[1]/2), :],
                                        responses[:, int(responses.shape[1]/2):, :]), axis=2)

        responses = np.median(responses, 2)
        orientations = np.arange(
            responses.shape[1]) * (2*np.pi/responses.shape[1])

        vector_sums = np.sum(
            responses * np.exp(correction_factor * 1j*(np.matlib.repmat(orientations, responses.shape[0], 1))), axis=1)

        return vector_sums

    def scatter_preferences(self, orientation=True, override=False, ax=None):
        """ Scatter plot of orientation preferences.


        Args:
            orientation (bool, optional): Defaults to True. Use orientation space.
            override (bool, optional): Defaults to False. Recompute orientation preferences.
            ax ([type], optional): Defaults to None. Axis to plot scatter.
        Returns: 
            [matplotlib.figure.Figure]: Figure with the scatter plot
        """

        if ax is None:
            fig = plt.figure(figsize=(5, 5))
            ax = fig.add_subplot(111)
        else:
            fig = ax.figure
        if orientation and 'orientation' not in self.metrics.columns and not override:
            self.metrics['orientation'] = (
                (np.rad2deg(np.angle(self.vector_sum_responses()))+360) % 360) / 2
        else:
            raise NotImplementedError('Other plotting options are not yet implemented')
        centroids = self.map_to_roi(lambda x: x.centroid())
        _ = ax.scatter([r[0] for r in centroids],
                       [r[1] for r in centroids],
                       s=25,
                       c=self.metrics['orientation'],
                       cmap='hsv', vmin=0, vmax=179)
        _ = ax.set_ylim((500, 0))
        return fig

    def preference_fits(self, orientation=True, override=False):
        """Fits von Mises function to responses.

        Handles the fitting of von Mises functions to response data using the analysis window specified in the 
        OrientationAnalysis.

        TODO:
            * Handle direction data

        Args:
            orientation (bool, optional): Defaults to True. [description]
            override (bool, optional): Defaults to False. [description]

        """

        responses = self.single_trial_responses()
        correction_factor = 2 if orientation else 1
        if self.expt.metadata.stim['doBlank'] is '1':
            responses = responses[:, :-1, :]
        if orientation:
            responses = np.concatenate((responses[:, :int(responses.shape[1]/2), :],
                                        responses[:, int(responses.shape[1]/2):, :]), axis=2)
            responses = np.median(responses, 2)
            orientations = np.arange(
                responses.shape[1]) * (2*np.pi/responses.shape[1])
            for cell_id in range(responses.shape[1]):
                popt, pcov = curve_fit(
                    self._von_mises_or_fit, orientations, responses[cell_id, :])
                return popt, pcov
        else:
            raise NotImplementedError('Direction fitting has not been implemented yet.')
