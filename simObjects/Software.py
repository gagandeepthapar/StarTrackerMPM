import sys

import numpy as np
import pandas as pd

sys.path.append(sys.path[0] + '/..')

import logging
from json import load as jsonload

import constants as c

from .Parameter import Parameter, UniformParameter

logger = logging.getLogger(__name__)

class Software:

    def __init__(self, centroiding:Parameter=None, identification:Parameter=None, * ,
                       ctr_json:str=c.IDEAL_CENTROID,
                       id_json:str=c.IDEAL_IDENT):

        self.dev_x, self.dev_y = self. __set_centroid(centroiding, ctr_json)
        # self.identification = self.__set_ident(identification, id_json)

        # self.fail_ident = self.__set_ident(identification, id_json, name='FAIL_IDENT_RATE')
        # self.data = self.randomize()
        
        self.__param_list = [self.dev_x, self.dev_y]#, self.fail_ident]
        # self.identification, 
        self.params = {param.name: param for param in self.__param_list}

        return

    def __repr__(self)->str:
        return 'Centroid: {}'.format(self.dev_x)

    def randomize(self, num:int=1)->pd.DataFrame:
    
        df = pd.DataFrame()
        df['BASE_DEV_X'] = np.ones(num) * (self.dev_x.stddev)
        df['BASE_DEV_Y'] = np.ones(num) * (self.dev_y.stddev)

        self.data = df
        return df

    def ideal(self, num:int=10_000)->pd.DataFrame:

        df = pd.DataFrame()
        df['BASE_DEV_X'] = self.dev_x.reset(num)
        df['BASE_DEV_Y'] = self.dev_y.reset(num)

        self.data = df
        return df

    def __set_centroid(self, param:Parameter, json_path:str)->tuple[Parameter, Parameter]:
        if param is not None:
            return param
        
        with open(json_path) as fp_open:
            ctr_data = jsonload(fp_open)

        ideal = ctr_data['IDEAL']
        mean = ctr_data['MEAN']
        stddev = ctr_data['STDDEV']
        units = ctr_data['UNITS']

        # devX = Parameter(ideal, stddev, mean, name='BASE_DEV_X', units=units)
        # devY = Parameter(ideal, stddev, mean, name='BASE_DEV_Y', units=units)

        devX = Parameter(0, stddev, name='BASE_DEV_X')
        devY = Parameter(0, stddev, name='BASE_DEV_Y')

        return devX, devY
    
    # def __set_ident(self, param:Parameter, json_path:str, name:str='IDENTIFICATION_ACCURACY')->UniformParameter:
    #     if param is not None:
    #         return param
        
    #     with open(json_path) as fp_open:
    #         id_data = jsonload(fp_open)

    #     min_acc = 0.9#id_data['MIN_ACCURACY']
    #     ident = UniformParameter(min_acc, 1.0, name=name, units='Percent')

    #     return ident