import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
from simObjects.Simulation import Simulation
from simObjects.AttitudeEstimation import QUEST, Projection
from simObjects.Software import Software
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker

logger = logging.getLogger(__name__)

class MonteCarlo(Simulation):

    def __init__(self, camera:StarTracker=None, centroid:Software=None, orbit:Orbit=None, num_runs:int=1_000)->None:
        super().__init__(camera=camera, centroid=centroid, orbit=orbit, num_runs=num_runs)

        return
    
    def __repr__(self)->str:
        return 'Monte Carlo Analysis: '+super().__repr__()
    
    def run_sim(self, params:list[str], obj_func:callable=None) -> pd.DataFrame:        
        if obj_func is None:
            obj_func = self.sun_etal_hardware_analysis

        start = time.perf_counter()        
        
        self.__create_data()
        self.sim_data['CALC_ACCURACY'] = self.sim_data.apply(obj_func, axis=1)
        
        end = time.perf_counter()
        logger.debug('Time to calculate: {}'.format(end-start))

        return self.sim_data

    def plot_data(self, **kwargs) -> None:
        return super().plot_data()
    
    def __create_data(self)->pd.DataFrame:

        # randomize all data from components
        f_data = self.camera.randomize(num=self.num_runs)
        c_data = self.centroid.randomize(num=self.num_runs)
        o_data = self.orbit.randomize(num=self.num_runs)

        self.sim_data = pd.concat([f_data, c_data, o_data], axis=1)

        return self.sim_data
