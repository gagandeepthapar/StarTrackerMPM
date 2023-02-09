import sys

import numpy as np
import pandas as pd

sys.path.append(sys.path[0] + '/..')

import logging
from json import load as jsonload

import constants as c

from .Parameter import Parameter

logger = logging.getLogger(__name__)

class Software:

    def __init__(self, centroiding:Parameter=None, identification:Parameter=None, * , ctr_json:str=c.IDEAL_CENTROID):

        self.centroid = self. __set_centroid(centroiding, ctr_json)
        self.identification = self.__set_identification(identification)

        self.data = self.randomize()

        return

    def __repr__(self)->str:
        return 'Centroid: {}\nIdentification:{}'.format(self.centroid, self.identification)

    def randomize(self, num:int=1)->pd.DataFrame:
    
        df = pd.DataFrame()
        df['BASE_DEV_X'] = self.centroid.modulate(num)
        df['BASE_DEV_Y'] = self.centroid.modulate(num)

        self.data = df
        return df

    def ideal(self, num:int=10_000)->pd.DataFrame:

        df = pd.DataFrame()
        df['BASE_DEV_X'] = self.centroid.reset(num)
        df['BASE_DEV_Y'] = self.centroid.reset(num)

        self.data = df
        return df

    def __set_centroid(self, param:Parameter, json_path:str)->Parameter:
        if param is not None:
            return param
        
        with open(json_path) as fp_open:
            ctr_data = jsonload(fp_open)

        ideal = ctr_data['IDEAL']
        mean = ctr_data['MEAN']
        stddev = ctr_data['STDDEV']
        units = ctr_data['UNITS']

        return Parameter(ideal, stddev, mean, name='Centroid_Accuracy', units=units)

    def __set_identification(self, param:Parameter)->Parameter:
        return None
    
    