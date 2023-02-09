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
    
    def run_sim(self, obj_func:callable=None) -> pd.DataFrame:        
        if obj_func is None:
            obj_func = self.sun_etal_hardware_analysis

        start = time.perf_counter()        
        
        self.__create_data()
        self.sim_data['CALC_ACCURACY'] = self.sim_data.apply(obj_func, axis=1)
        
        end = time.perf_counter()
        logger.debug('Time to calculate: {}'.format(end-start))

        return self.sim_data
    
    def plot_data(self, plot_params:bool=False) -> None:
        
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.hist(self.sim_data['CALC_ACCURACY'], bins=int(np.sqrt(self.num_runs)))
        # ax.set_ylabel('Number of Runs')
        ax.set_xlabel('Calculated Accuracy [arcsec]')
        ax.set_title('Star Tracker Accuracy: {} +/-{} arcsec:\n{:,} Runs'.\
                    format(np.round(self.sim_data['CALC_ACCURACY'].mean(),3),\
                           np.round(self.sim_data['CALC_ACCURACY'].std(),3),
                           self.num_runs))

        if plot_params:
            param_fig = plt.figure()
            size = (3, 2)
            params = ['FOCAL_LENGTH', 'FOCAL_ARRAY_INCLINATION', 'DISTORTION', 'PRINCIPAL_POINT_ACCURACY', 'BASE_DEV_X', 'TEMP']

            for i, param in enumerate(params):
                param_ax = param_fig.add_subplot(size[0], size[1], i+1)

                param_ax.hist(self.sim_data[param], bins=int(np.sqrt(self.num_runs)), label=param)
                param_ax.set_title('{}: {} +/- {}'.\
                                   format(param.replace('_', ' ').title(),\
                                          np.round(self.sim_data[param].mean(), 3),\
                                          np.round(self.sim_data[param].std(), 3)))

                if param == 'FOCAL_LENGTH':
                    f_len_real = self.sim_data['FOCAL_LENGTH'][0] - self.sim_data['D_FOCAL_LENGTH'][0]
                    param_ax.axvline(f_len_real, color='r', label='True Focal Length ({})'.format(np.round(f_len_real,3)))
                    param_ax.legend()
                    
                
                if param == 'BASE_DEV_X':
                    param_ax.hist(self.sim_data['BASE_DEV_Y'], color='g', label='BASE_DEV_Y', bins=int(np.sqrt(self.num_runs)))
                    param_ax.legend()
                    param_ax.set_title('Centroiding Accuracy:\n{} +/- {} (X); {} +/- {} (Y)'.\
                                       format(np.round(self.sim_data['BASE_DEV_X'].mean(),3),\
                                              np.round(self.sim_data['BASE_DEV_X'].std(),3),\
                                              np.round(self.sim_data['BASE_DEV_Y'].mean(),3),\
                                              np.round(self.sim_data['BASE_DEV_Y'].std(),3)))
                    


        return
    
    def __create_data(self)->pd.DataFrame:

        f_data = self.camera.randomize(num=self.num_runs)
        c_data = self.centroid.randomize(num=self.num_runs)
        o_data = self.orbit.randomize(num=self.num_runs)

        self.sim_data = pd.concat([f_data, c_data, o_data], axis=1)

        return self.sim_data