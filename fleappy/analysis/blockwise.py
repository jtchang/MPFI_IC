from fleappy.analysis import BaseAnalysis
import numpy as np


class BlockwiseAnalysis(BaseAnalysis):
    """Class for the handling of blockwise stimuli.

    Class for the analysis of stimuli which can be subdivided into stimulus x trials. This is a subclass of the 
    BaseAnalysis class.

    Attributes:
        stim_period (tuple): Stimulus period (start, stop).
        prepad (float): Period before stimulus onset to analyze.
        postpad (float): Period after stimulus offset to analyze.
        analysis_period (tuple): Period during stimulus to analyze (start, stop).
    """

    __slots__ = ['stim_period', 'prepad', 'postpad', 'analysis_period']

    def __init__(self, expt, analysis_id, field, analysis_period=(0, -1), prepad=0, postpad=0):
        BaseAnalysis.__init__(self, expt, analysis_id, field)
        self.analysis_period = analysis_period
        if self.analysis_period[1] == -1:
            self.analysis_period = (self.analysis_period[0], float(self.expt.metadata.stim_duration()))
        self.stim_period = (0, expt.metadata.stim_duration)
        self.prepad = prepad
        self.postpad = postpad

    def single_trial_responses(self):
        """Gets all single trial responses over analysis window.

        Returns:
            nd.array : Trial Responses (# stims x # trials x time)
        """

        frame_rate = self.expt.metadata.frame_rate()
        prepad_frames = np.round(self.prepad*frame_rate)

        analysis_start = int(prepad_frames + np.round(self.analysis_period[0]*frame_rate))
        analysis_stop = int(prepad_frames + np.round(self.analysis_period[1]*frame_rate))

        responses = self.expt.get_all_trial_responses(self.field, prepad=self.prepad, postpad=self.postpad)
        responses = np.mean(responses[:, :, :, analysis_start:analysis_stop], axis=3)
        return responses
