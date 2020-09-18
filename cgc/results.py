import datetime
import json

import numpy as np

from . import __version__


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class Results(object):
    """
    Base class to be inherited by the various calculators. It is meant to
    contain results and metadata of a calculation.
    """
    def __init__(self, **kwargs):
        self.version = __version__
        self.time_created = datetime.datetime.now().isoformat()
        self.input_parameters = {name: value for name, value in kwargs.items()}
        self.reset()

    def reset(self):
        """
        The attributes and their default value can be defined in the reset
        method.
        """
        pass

    def write(self, filename=''):
        """
        Serialize the object attributes in a JSON file.

        :param filename:
        """
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.__dict__, f, indent=4, cls=NumpyEncoder)

    def __setattr__(self, name, value):
        self.__dict__['time_updated'] = datetime.datetime.now().isoformat()
        self.__dict__[name] = value
