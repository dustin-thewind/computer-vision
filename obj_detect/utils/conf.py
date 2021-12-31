# import packages
from json_minify import json_minify
import json

class Conf:
    def __init__(self, confPath):
        # load and store the config file and update the objects dict
        conf = json.loads(json_minify(open(confPath).read()))
        self.__dict__.update(conf)

    def __getitem__(self, k):
        # return the value assocaied with the supplied key
        return self.__dict__.get(k, None)
