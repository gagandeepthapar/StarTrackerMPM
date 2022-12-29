import numpy as np

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

import json

class Parameter:

    def __init__(self, ideal:float, stddev:float, mean:float=0, name:str=None)->None:

        self.ideal = ideal
        self.name = name

        self._err_mean = mean
        self._err_stddev = stddev
    
        self.range = self._err_mean + (3*self._err_stddev)
        self.minRange = self.ideal - self.range
        self.maxRange = self.ideal + self.range

        self.modulate()

        return
    
    def __repr__(self)->str:
        pname = '{}: {} [{}(\u03BC) +/- {}(3\u03C3)]'.format(self.name, np.round(self.value,3), self.ideal, 3*self._err_stddev)
        return pname 

    def modulate(self)->float:
        self.value = self.ideal + (np.random.normal(loc=self._err_mean, scale=self._err_stddev))
        return self.value
    
    def reset(self)->float:
        self.value = self.ideal
        return self.value

    def get_ideal_param(self)->None:
        return Parameter(ideal=self.ideal, stddev=0, mean=0, name="IDEAL_"+self.name)

    def _init_from_json(fp:str,name:str)->None:

        camDict = json.load(open(fp))

        ideal = camDict[name+"_IDEAL"]
        mean = camDict[name+"_MEAN"]
        stddev = camDict[name+"_STDDEV"]/3

        return Parameter(ideal=ideal, stddev=stddev, mean=mean, name=name)
