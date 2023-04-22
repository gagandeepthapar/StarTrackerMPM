import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
from simObjects.Simulation import Simulation
from simObjects.AttitudeEstimation import QUEST, RandomProjection
from simObjects.Software import Software
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker

logger = logging.getLogger(__name__)

class MonteCarlo(Simulation):

    def __init__(self, camera:StarTracker, software:Software, orbit:Orbit, num_runs:int)->None:
        super().__init__(camera=camera, software=software, orbit=orbit, num_runs=num_runs)

        return
    
    def __repr__(self)->str:
        return 'Monte Carlo Analysis: '+super().__repr__()

    def run_sim(self, params, obj_func:callable) -> pd.DataFrame:
        self.__create_data()
        return super().run_sim(params=params, obj_func=obj_func)

    def plot_data(self, data:pd.DataFrame, *kwargs) -> None:
        return super().plot_data(data)
    
    def __create_data(self)->pd.DataFrame:

        fov = np.deg2rad(10)

        # randomize all data from components
        q_data = pd.DataFrame({'RIGHT_ASCENSION': np.random.uniform(fov, 2*np.pi - fov, self.num_runs),
                              'DECLINATION': np.random.uniform(-np.pi/2 + fov, np.pi/2 - fov, self.num_runs),
                              'ROLL': np.random.uniform(-np.pi, np.pi, self.num_runs)})
        f_data = self.camera.randomize(num=self.num_runs)
        c_data = self.software.randomize(num=self.num_runs)
        o_data = self.orbit.randomize(num=self.num_runs)

        self.sim_data = pd.concat([q_data, f_data, c_data, o_data], axis=1)
        
        # update focal_length based on temperature
        df_dtemp = self.sim_data['FOCAL_LENGTH'] * (self.sim_data['D_TEMP'] * self.sim_data['FOCAL_THERMAL_COEFFICIENT'])

        logger.debug('\n{}DTEMP: {} +/- {}{}'.format(c.RED, df_dtemp.mean(), df_dtemp.std(), c.DEFAULT))
        logger.debug('\n{}flen: {}{}'.format(c.RED, self.camera.f_len.ideal, c.DEFAULT))
        
        self.sim_data['FOCAL_LENGTH'] = self.sim_data['FOCAL_LENGTH'] + df_dtemp
        self.sim_data['D_FOCAL_LENGTH'] = self.sim_data['FOCAL_LENGTH'] - self.camera.f_len.ideal
        
        return self.sim_data
