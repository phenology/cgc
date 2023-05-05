import datetime
import json

import numpy as np

from . import __version__


class ArrayEncoder(json.JSONEncoder):
    def default(self, obj):
        if hasattr(obj, '__array__'):
            return np.asarray(obj).tolist()
        return json.JSONEncoder.default(self, obj)


class Results(object):
    """
    Base class to be inherited by the various calculators. It is meant to
    contain results and metadata of a calculation.

    :param **input_parameters: Input parameters.
    """
    def __init__(self, **input_parameters):
        self.version = __version__
        self.time_created = datetime.datetime.now().isoformat()
        self.input_parameters = {
            name: value for name, value in input_parameters.items()
        }

    def write(self, filename=''):
        """
        Serialize the object attributes in a JSON file.

        :param filename: Name of the file.
        :type filename: str
        """
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.__dict__, f, indent=4, cls=ArrayEncoder)

    def __setattr__(self, name, value):
        self.__dict__['time_updated'] = datetime.datetime.now().isoformat()
        self.__dict__[name] = value
