import pickle
import fleappy.analysis


class BaseExperiment(object):
    """Base Experiment Class.

    Attributes:
        metadata (list): List of metadata objects.
        analysis (dict): Dictionary for analysis.
    """

    __slots__ = ['metadata', 'analysis']

    def __init__(self):
        self.analysis = {}
        self.metadata = []

    def save_to_file(self, filename):
        filehandler = open(filename, 'w')
        pickle.dump(self, filehandler)

    @staticmethod
    def load_file(filename):
        filehandler = open(filename, 'r')
        return pickle.load(filehandler)
