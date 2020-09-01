import datetime
import json

from . import __version__


class Results(object):
    def __init__(self):
        self.version = __version__
        self.time_created = datetime.datetime.now().isoformat()
        self.reset()

    def reset(self):
        pass

    def write(self, filename=''):
        if filename:
            with open(filename, 'w') as f:
                json.dump(self.__dict__, f, indent=4)

    def __setattr__(self, name, value):
        self.__dict__['time_updated'] = datetime.datetime.now().isoformat()
        self.__dict__[name] = value
