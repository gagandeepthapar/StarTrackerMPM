import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import pandas as pd

from dataclasses import dataclass

import matplotlib.pyplot as plt
from alive_progress import alive_bar
from Parameter import Parameter, UniformParameter

import constants as c
from simTools import generateProjection as gp

@dataclass
class Projection:
    def __init__(self)->None:
        self.ra = UniformParameter(-180, 180, name="RIGHT_ASCENSION", units=c.DEG)
        self.dec = UniformParameter(-180, 180, name="DECLINATION", units=c.DEG)
        self.roll = UniformParameter(-180, 180, name="ROLL", units=c.DEG)

        self.catalog = c.BSC5PKL
        self.cam = c.ALVIUM_CAM

        self.frame:pd.DataFrame=None

        self.randomize()
        return
    
    def __repr__(self)->str:
        name = "PROJECTION STRUCT"
        return name

    def randomize(self, mag:float=9,plot:bool=False)->None:
        self.ra.modulate()
        self.dec.modulate()
        self.roll.modulate()
        frame = gp.generate_projection(ra=self.ra.value,
                                        dec=self.dec.value,
                                        roll=self.roll.value,
                                        camera_mag=mag,
                                        cfg_fp=self.cam,
                                        catpkl_fp=self.catalog,
                                        plot=plot)

        self.frame = frame[['catalog_number',
                            'right_ascension',
                            'declination',
                            'ECI_X',
                            'ECI_Y',
                            'ECI_Z',
                            'CV_X',
                            'CV_Y',
                            'CV_Z']]
        return

class QUEST(Projection):

    def __init__(self)->None:
        super().__init__()
        return
    
    def __repr__(self)->str:
        name = 'QUEST SOLVER:\n{}\n'.format(self.frame)
        return name

    def get_attitude(self)->np.ndarray:

        return
    
    def __f(self, x, a, b, c, d)->float:

        return
    
    def __fp(self, x, a, b, c, d)->float:

        return

    
if __name__ == '__main__':
    roll = UniformParameter(0, 0, 'ROLL', units=c.DEG)
    Q = QUEST()
    print(Q)
