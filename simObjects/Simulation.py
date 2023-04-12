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
from simObjects.Parameter import Parameter, UniformParameter
from simObjects.StarTracker import StarTracker
from simObjects.AttitudeEstimation import Projection, QUEST

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
                                self.quest_objective:'QUATERNION_ERROR',
                                self.sun_etal_quest:'QUATERNION_ERROR'
                            }

        return

    def __repr__(self)->str:
        return '{} Data Points'.format(self.num_runs)

    def run_sim(self, params, obj_func:callable)->pd.DataFrame: 

        if obj_func is None or obj_func not in list(self.obj_func_out.keys()):
            obj_func = self.sun_etal_star_mismatch

        column = self.obj_func_out.get(obj_func)

        logger.debug('{}{} Objective Function{}'.format(c.RED,column, c.DEFAULT))
        
        match obj_func:    
            case self.sun_etal_star_mismatch:
                
                self.sim_data[column] = self.sim_data.apply(obj_func, axis=1)
                mean = self.sim_data[column].mean()
                std = self.sim_data[column].std()
                rng_min = mean - 3*std
                rng_max = mean + 3*std
                
                self.sim_data['CALC_ACCURACY'] = np.max([np.abs(rng_max/2), np.abs(rng_min/2)])
            
            case self.quest_objective:
                self.sim_data[column] = self.sim_data.apply(obj_func, axis=1)
                self.sim_data['CALC_ACCURACY'] = self.sim_data[column]

            case _:
                self.sim_data['CALC_ACCURACY'] = self.sim_data.apply(obj_func, axis=1)

        self.sim_data = self.sim_data[self.sim_data['CALC_ACCURACY'] < 3600]

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
        # cam_sw_params = {**self.camera.params, **self.software.params}
        
        cam_sw_params    = {"Focal Length [px]": self.camera.f_len,
                         "Number of Stars": self.camera.sensor,
                         r'$\epsilon_x$ [px]':self.camera.eps_x,
                         r"$\epsilon_y$ [px]":self.camera.eps_y,
                         r"$\epsilon_z$ [px]":self.camera.eps_z,
                         r"$\phi$ [deg]":self.camera.phi,
                         r"$\theta$ [deg]":self.camera.theta,
                         r"$\psi$ [deg]":self.camera.psi,
                         "Centroid-X Deviation [px]":self.software.dev_x,
                         "Centroid-Y Deviation [px]":self.software.dev_y,
                         "Identification Performance [%]":self.software.fail_ident}

        # cam_sw_params = {param.name:param for param in param_list}

        size = (int(len(cam_sw_params)/2)+1, 2)

        
        for i, param in enumerate(list(cam_sw_params.keys())):

            param_data = self.sim_data[cam_sw_params[param].name]

            param_ax = param_fig.add_subplot(size[0], size[1], i+1)

            if i%2 == 0: param_ax.set_ylabel('Number of Runs')

            param_ax.hist(param_data, bins=int(np.sqrt(self.num_runs)))
            param_ax.set_title('{}: {} +/- {}'.\
                                format(param,\
                                        np.round(param_data.mean(), 3),\
                                        np.round(param_data.std(), 3)))

            if param == 'FOCAL_LENGTH':
                param_ax.axvline(self.camera.f_len.ideal, color='r', label='True Focal Length ({} px)'.format(np.round(self.camera.f_len.ideal,3)))
            
            param_ax.legend()

        param_ax = param_fig.add_subplot(size[0], size[1], len(cam_sw_params)+1)
        param_ax.hist(self.sim_data['CALC_ACCURACY'], bins=int(np.sqrt(self.num_runs)))
        param_ax.set_ylabel('Number of Runs')
        param_ax.set_xlabel('Calculated Accuracy [arcsec]')
        param_ax.set_title('Star Tracker Accuracy: {} +/-{} arcsec:\n{:,} Runs'.\
                    format(np.round(self.sim_data['CALC_ACCURACY'].mean(),3),\
                           np.round(self.sim_data['CALC_ACCURACY'].std(),3),
                           self.num_runs))

        # print(len(self.sim_data.index))

        return

    """ 
    SUN ET AL STAR LOCATION ERROR FUNCTION
    """

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

    """ 
    FIRST PRINCIPLES HARDWARE DEVIATION + QUEST FUNCTION
    """

    def quest_objective(self, sim_row:pd.DataFrame)->float:
        """
        evaluation of hardware by passing it through QUEST.
        Exploitation of Camera Pinhole Model        

        Args:
            sim_row (pd.DataFrame): row item from data set containing information about star tracker hardware and centroiding ability

        Returns:
            float: calculated accuracy

        """

        project = Projection(sim_row)
        # logger.debug('{}STAR_FRAME:\n{}{}'.format(c.RED, project.frame.to_string(), c.DEFAULT))
        
        eci_real = project.frame['ECI_TRUE'].to_numpy()
        cv_real = project.frame['CV_TRUE'].to_numpy()
        cv_est = project.frame['CV_MEAS'].to_numpy()
        q_real = project.quat_real
        
        quest = QUEST(eci_real, cv_real, cv_est, sim_row.FAIL_IDENT_RATE,q_real)

        q_diff = quest.calc_acc()
        # logger.debug('{}Q_DIFF: {}{}'.format(c.RED, q_diff, c.DEFAULT))

        return q_diff

    """ 
    SUN ET AL STAR LOCATION -> QUEST FUNCTION
    """

    def __sun_etal_px_diff(self, row:pd.Series, star_angle:float=np.random.uniform(-8,8))->float:

        theta = np.deg2rad(row.FOCAL_ARRAY_INCLINATION)
        star_angle = np.deg2rad(star_angle)
        delta_s = np.linalg.norm([row.BASE_DEV_X, row.BASE_DEV_Y])

        fa_num = row.FOCAL_LENGTH + row.D_FOCAL_LENGTH + row.PRINCIPAL_POINT_ACCURACY*np.tan(theta)
        fa_den = np.cos(theta + star_angle)

        delta_star = fa_num/fa_den * np.sin(star_angle) + delta_s + row.DISTORTION - row.FOCAL_LENGTH*np.tan(star_angle) 

        return delta_star

    def __set_real_rotation(self)->None:
        
        # quaternion representation
        q = np.random.uniform(-1,1,4)
        q = q/np.linalg.norm(q)
        
        # rotation matrix
        
        skew = lambda n: np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        e = -1*q[:3]
        n = q[3]
        
        Ca = (2*n**2 - 1) * np.identity(3)
        Cb = 2*np.outer(e, e)
        Cc = -2*n*skew(e)

        C = Ca + Cb + Cc

        return q, C

    def __px_to_cv(self, row:pd.Series, f_len:float)->np.ndarray:
        x = row[0]
        y = row[1]
        z = f_len

        cvx = x - c.SENSOR_WIDTH/2
        cvy = y - c.SENSOR_HEIGHT/2
        cvz = z

        v = np.array([cvx, cvy, cvz])
        return v/np.linalg.norm(v)

    def __create_starlist(self,num_stars:int, mis_match:float, rot_matr:np.ndarray, f_len:float, df_len:float)->pd.DataFrame:
    
        frame = pd.DataFrame()
        
        # set perceived information
        frame['IMAGE_X'] = np.random.uniform(0, c.SENSOR_WIDTH, int(num_stars))
        frame['IMAGE_Y'] = np.random.uniform(0, c.SENSOR_HEIGHT, int(num_stars))

        frame['IMAGE_DEV_X'] = frame['IMAGE_X'] + np.random.uniform(-1*np.abs(mis_match), np.abs(mis_match), len(frame.index))
        frame['IMAGE_DEV_Y'] = frame['IMAGE_Y'] + np.random.uniform(-1*np.abs(mis_match), np.abs(mis_match), len(frame.index))

        # create camera vectors
        frame['CV_EST'] = frame[['IMAGE_DEV_X', 'IMAGE_DEV_Y']].apply(self.__px_to_cv, axis=1, args=(f_len, ))

        # set real information
        frame['CV_REAL'] = frame[['IMAGE_X', 'IMAGE_Y']].apply(self.__px_to_cv, axis=1, args=(f_len+df_len, ))

        rot = lambda x:  rot_matr @ x
        frame['ECI_REAL'] = frame['CV_REAL'].apply(rot)

        return frame
    
    def sun_etal_quest(self,row:pd.Series)->float:
        """
        combines star mismatch due to hardware from sunetal with QUEST quaternion solver

        Args:
            row (pd.Series): row item from dataset containing star tracker information
            star_angle (float, optional): random angle star makes wrt the focal array. Defaults to np.random.uniform(-8, 8).
        
        Returns:
            float: accuracy of system
        """

        q, C = self.__set_real_rotation()
        mismatch = self.__sun_etal_px_diff(row)

        flen = row.FOCAL_LENGTH - row.D_FOCAL_LENGTH

        star_list = self.__create_starlist(row.NUM_STARS_SENSOR, mismatch, C, flen, row.D_FOCAL_LENGTH)
        
        eci_real = star_list['ECI_REAL'].to_numpy()
        cv_real = star_list['CV_REAL'].to_numpy()
        cv_est = star_list['CV_EST'].to_numpy()
        
        quest = QUEST(eci_real, cv_real, cv_est, row.IDENTIFICATION_ACCURACY,q)
        q_diff = quest.calc_acc() 

        return q_diff
