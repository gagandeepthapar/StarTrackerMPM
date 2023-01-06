import numpy as np

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

import json

class Parameter:

    def __init__(self, ideal:float, stddev:float, mean:float=0, name:str=None, units:str=None, color:str=c.DEFAULT)->None:

        self.ideal = ideal
        self.name = name

        self._err_mean = mean
        self._err_stddev = stddev
        self._color = color
        self.range = self._err_mean + (3*self._err_stddev)
        self.minRange = self.ideal - self.range
        self.maxRange = self.ideal + self.range

        if units == "deg":
            units = c.DEG
        if units is None:
            units = ""
        self.units = units
        self.value = self.modulate()
        self.reset()

        return
    
    def __repr__(self)->str:
        pname = f'{self._color}{self.name}: {np.round(self.value,3)}{self.units} [{self.ideal+self._err_mean}({c.MU}) +/- {3*self._err_stddev}(3{c.SIGMA})]{c.DEFAULT}'
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
        units = camDict[name+"_UNITS"]

        return Parameter(ideal=ideal, stddev=stddev, mean=mean, name=name, units=units)

class UniformParameter:

    def __init__(self, low:float, high:float, name:str=None, units:str=None, color:str=c.DEFAULT)->None:
        self.low = low
        self.high = high
        
        self.name = name
        self.units = units
        self.color = color

        self.value = self.modulate()

        return
    
    def __repr__(self)->str:
        name = f'{self.color}{self.name}:{self.low} - {self.high}{self.units}{c.DEFAULT}'
        return name
    
    def modulate(self)->float:
        self.value = np.random.uniform(self.low, self.high) 
        return self.value