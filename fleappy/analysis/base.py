import fleappy.experiment


class BaseAnalysis(object):
    """A metaclass for analysis of experiments.

    This is a base class for experiment analysis that handles associating analyses with an experiment, an id, and the
    associated data field.


    Attributes:
        expt (fleappy.experiment.BaseExperiment): Experiment analysis is associated with
    """
    __slots__ = ['expt', 'id', 'field']

    def __init__(self, expt, analysis_id, field):
        if not issubclass(type(expt), fleappy.experiment.BaseExperiment):
            raise TypeError('%s is not a Experiment Object' % (expt))
        self.expt = expt
        self.id = analysis_id
        self.field = field

    def run(self):
        pass

    def map_to_roi(self, func):
        """Applies a function to each roi in the associated experiment.

        Args:
            func (method): method to employ

        Returns:
            (list): results of applying function to method
        """

        return [func(r) for r in self.expt.roi]
