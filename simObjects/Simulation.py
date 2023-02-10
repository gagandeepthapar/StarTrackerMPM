import sys

sys.path.append(sys.path[0] + '/..')

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import constants as c
from simObjects.Software import Software
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker
import threading

logger = logging.getLogger(__name__)

@dataclass
class Simulation:
    camera:StarTracker = None
    centroid:Software = None
    orbit:Orbit = None
    num_runs:int = None
    
    def __post_init__(self)->None:
        """
        Allows default values to be set even from derived classes without copying default parameters
        """
        if self.camera is None:
            self.camera = StarTracker(cam_json=c.IDEAL_CAM, cam_name='Ideal Camera')

        if self.centroid is None:
            self.software = Software()

        if self.orbit is None:
            self.orbit = Orbit()

        if self.num_runs is None:
            self.num_runs = 1_000

        self.sim_data = pd.DataFrame()

        self.params = {**self.camera.params, **self.centroid.params, **self.orbit.params}

        return

    def __repr__(self)->str:
        return '{} Data Points'.format(self.num_runs)


    def run_sim(self, params:list[str], obj_func:callable)->pd.DataFrame:  # method to be overloaded
        raise RuntimeError('run_sim not implemented for {} class'.format(type(self)))
        return 


    def plot_data(self)->None: # method to be overloaded
        
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.hist(self.sim_data['CALC_ACCURACY'], bins=int(np.sqrt(self.num_runs)))
        ax.set_ylabel('Number of Runs')
        ax.set_xlabel('Calculated Accuracy [arcsec]')
        ax.set_title('Star Tracker Accuracy: {} +/-{} arcsec:\n{:,} Runs'.\
                    format(np.round(self.sim_data['CALC_ACCURACY'].mean(),3),\
                           np.round(self.sim_data['CALC_ACCURACY'].std(),3),
                           self.num_runs))

        param_fig = plt.figure()
        param_fig.suptitle('Input Parameter Distribution')
        size = (int(len(self.params)/2), 2)

        for i, param in enumerate(list(self.params.keys())):

            param_ax = param_fig.add_subplot(size[0], size[1], i+1)

            if i%2 == 0: param_ax.set_ylabel('Number of Runs')

            param_ax.hist(self.sim_data[param], bins=int(np.sqrt(self.num_runs)), label=param)
            param_ax.set_title('{}: {} +/- {}'.\
                                format(param.replace('_', ' ').title(),\
                                        np.round(self.sim_data[param].mean(), 3),\
                                        np.round(self.sim_data[param].std(), 3)))

            if param == 'FOCAL_LENGTH':
                param_ax.axvline(self.camera.f_len.ideal, color='r', label='True Focal Length ({} px)'.format(np.round(self.camera.f_len.ideal,3)))
                param_ax.legend()
                                        
        return


    def sun_etal_hardware_analysis(self, row:pd.Series, star_angle:float=np.random.uniform(-8, 8))->float:
        """
        function to calculate accuracy of star tracker from hardware as outlined in "Optical System Error Analysis and Calibration Method of High-Accuracy Star Trackers", Sun et al (2013)
        
        Args:
            row (pd.Series): row item from data set containing information about star tracker hardware and centroiding ability
            star_angle (float, optional): random angle star makes wrt the focal array. Defaults to np.random.uniform(-8, 8).

        Returns:
            float: accuracy of the star tracker described by that row
        """

        delta_s = np.linalg.norm([row.BASE_DEV_X, row.BASE_DEV_Y])
        theta = np.deg2rad(row.FOCAL_ARRAY_INCLINATION)
        star_angle = np.deg2rad(star_angle)

        foc_len = row.FOCAL_LENGTH - row.D_FOCAL_LENGTH

        fa_num = row.FOCAL_LENGTH + row.PRINCIPAL_POINT_ACCURACY * np.tan(theta)
        fa_dec = np.cos(theta + star_angle)

        fb = row.PRINCIPAL_POINT_ACCURACY / np.cos(theta)

        eA_A = np.arctan((fa_num/fa_dec * np.sin(star_angle) + fb + delta_s + row.DISTORTION)/foc_len)
        eA_B = np.arctan(row.PRINCIPAL_POINT_ACCURACY/(foc_len * np.cos(theta)))

        return 3600 * np.rad2deg(eA_A - eA_B - star_angle)
    

    def __generate_data(self)->None: # method to be overloaded
        raise RuntimeError('generate data not implemented for {} class'.format(type(self)))
        return

class SimThread(threading.Thread):
    
    def __init__(self, simulation:Simulation, idx_start:int, idx_end:int, params:list[str], obj_func:callable, thread_num:int)->None:
        threading.Thread.__init__(self)
        self.thread_num = thread_num

        self.thread_data:pd.DataFrame() = None

        self.sim = deepcopy(simulation)
        self.sim.num_runs = (idx_end - idx_start + 1)
        self.sim.sim_data = self.sim.sim_data.iloc[idx_start:idx_end]

        self.params = params
        self.obj_func = obj_func
        
        return
    
    def run(self)->None:
        logger.critical('Thread {}: Spinning Up'.format(self.thread_num))
        self.sim.run_sim(self.params, self.obj_func)
        self.thread_data = self.sim.sim_data
        logger.warning('Thread {}: Spinning Down'.format(self.thread_num))
        return
