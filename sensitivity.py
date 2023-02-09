import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
from simObjects.Simulation import Simulation
from simObjects.AttitudeEstimation import QUEST, Projection
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker

logger = logging.getLogger(__name__)

class Sensitivity(Simulation):

    def __init__(self, camera:StarTracker, centroid:Parameter, orbit:Orbit, num_runs:int=1_000)->None:
        super().__init__(camera, centroid, orbit, num_runs)

        return
    
    def __repr__(self)->str:
        return 'Sensitivity Analysis: {} Data Points'.format(self.num_runs)
    
    def run_sim(self, param_name:str='FOCAL_LENGTH', obj_func: callable=None, *, single_sim:bool=False) -> pd.DataFrame:
        if obj_func is None:
            obj_func = self.sun_etal_hardware_analysis

        start = time.perf_counter()

        self.__create_data(param_name)
        self.sim_data['CALC_ACCURACY'] = self.sim_data.apply(obj_func, axis=1)

        end = time.perf_counter()
        logger.debug('Time to calculate: {}'.format(end-start))

        return

    def plot_data(self, plot_params:bool=False) -> None:
        
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.hist(self.sim_data['CALC_ACCURACY'], bins=int(np.sqrt(self.num_runs)))
        ax.set_ylabel('Number of Runs')
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
                    param_ax.axvline(self.camera.f_len.ideal, color='r', label='True Focal Length ({} px)'.format(np.round(self.camera.f_len.ideal,3)))
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
    
    def __create_data(self, param_name:str)->pd.DataFrame:

        # idealize parameters from components
        f_data = self.camera.ideal(num=self.num_runs)
        c_data = self.centroid.ideal(num=self.num_runs)
        o_data = self.orbit.ideal(num=self.num_runs)

        # modulate parameter
        if param_name in f_data.columns:
            f_data[param_name] = self.camera.params[param_name].modulate(self.num_runs)

            if param_name == 'FOCAL_LENGTH':
                f_data['D_FOCAL_LENGTH'] = f_data['FOCAL_LENGTH'] - self.camera.f_len.ideal
        
        if param_name == 'CENTROID':
            c_data['BASE_DEV_X'] = self.centroid.centroid.modulate(self.num_runs)
            c_data['BASE_DEV_Y'] = self.centroid.centroid.modulate(self.num_runs)
        
        if param_name == 'IDENTIFICATION':
            pass
        
        if param_name in o_data.columns:
            o_data[param_name] = self.orbit.params[param_name].modulate(self.num_runs)
        
        self.sim_data = pd.concat([f_data, c_data,o_data], axis=1)

        logger.info('params:\n{}\n{}'.format(self.sim_data.mean(), self.sim_data.std()))

        return self.sim_data
