import sys

sys.path.append(sys.path[0] + '/..')

import logging
from dataclasses import dataclass
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
from simObjects.Software import Software
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker
from simObjects.AttitudeEstimation import QUEST

import threading

logger = logging.getLogger(__name__)

class Simulation:

    def __init__(self, camera:StarTracker=StarTracker(cam_json=c.IDEAL_CAM, cam_name='Ideal Camera'),
                       software:Software=Software(),
                       orbit:Orbit=Orbit(),
                       num_runs:int=1_000)->None:
        """
        Allows default values to be set even from derived classes without copying default parameters
        """

        self.camera = camera
        self.software = software
        self.orbit = orbit
        self.num_runs = num_runs

        self.sim_data = pd.DataFrame()
        self.params = {**self.camera.params, **self.software.params, **self.orbit.params}

        self.obj_func_out = {
                                self.sun_etal_star_mismatch:'STAR_ANGLE_MISMATCH',
                                self.quest_objective:'QUATERNION_ERROR'
                            }

        return

    def __repr__(self)->str:
        return '{} Data Points'.format(self.num_runs)


    def run_sim(self, obj_func:callable=None)->pd.DataFrame:  # method to be overloaded
        if obj_func is None or obj_func not in list(self.obj_func_out.keys()):
            obj_func = self.sun_etal_star_mismatch

        column = self.obj_func_out.get(obj_func)
        start = time.perf_counter()        
        
        match obj_func:    
            case self.sun_etal_star_mismatch:
                
                self.sim_data[column] = self.sim_data.apply(obj_func, axis=1)
                mean = self.sim_data[column].mean()
                std = self.sim_data[column].std()
                rng_min = mean - 3*std
                rng_max = mean + 3*std

            
            case self.quest_objective:
                self.sim_data[column] = self.sim_data.apply(obj_func, axis=1)

            case _:
                self.sim_data['CALC_ACCURACY'] = self.sim_data.apply(obj_func, axis=1)

        self.sim_data['CALC_ACCURACY'] = self.sim_data[column]

        end = time.perf_counter()
        logger.debug('Time to calculate: {}'.format(end-start))

        return self.sim_data


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


    def sun_etal_star_mismatch(self, row:pd.Series, star_angle:float=np.random.uniform(-8, 8))->float:
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

    def quest_objective(self, row:pd.DataFrame)->float:
        """
        evaluation of hardware by passing it through QUEST.
        Exploitation of Camera Pinhole Model        

        Args:
            row (pd.DataFrame): row item from data set containing information about star tracker hardware and centroiding ability

        Returns:
            float: calculated accuracy

        """
        
        quest_obj = QUEST(self.software.dev_x, self.software.dev_y,
                          1024, 1024, self.camera.f_len.ideal, sim_row=row)
        
        quat_calc = quest_obj.get_attitude()
        quat_diff = quest_obj.calc_diff(quat_calc)

        if quat_diff > 3600: # QUEST Fails if diff > 1 deg; return 0 (or NO sol'n)
            quat_diff = 0

        return quat_diff

    def __px_to_cv(self, x, y, f)->np.ndarray:

        v = np.array([x, y, f])

        return v/np.linalg.norm(v)

    def __create_data(self)->None: # method to be overloaded
        raise RuntimeError('generate data not implemented for {} class'.format(type(self)))
        return
