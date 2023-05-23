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
from simObjects.StarTracker import StarTracker
from simObjects.AttitudeEstimation import QUEST
from simObjects.Projection import Projection, RandomProjection, StarProjection

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

        self.catalog:pd.DataFrame = pd.read_pickle(c.BSC5PKL)

        self.obj_func_out = {
                                # self.sun_etal_star_mismatch:'STAR_ANGLE_MISMATCH',
                                self.quest_objective:'QUATERNION_ERROR_[arcsec]',
                                self.phi_model:'ANGULAR_SEPARATION_[arcsec]',
                                self.star_mag:'MAXIMUM MAGNITUDE VISIBLE'
                            }

        self.output_name:str=None

        self.half_normal_std = lambda x: x / np.sqrt(1 - 2/np.pi)
        self.half_normal_mean = lambda x: self.half_normal_std(x) * np.sqrt(2/np.pi)
        self.type:str=None
        return

    def __repr__(self)->str:
        return '{} Data Points'.format(self.num_runs)

    def run_sim(self, params, obj_func:callable, **kwargs)->pd.DataFrame: 

        sim_data = kwargs.get('trueData', self.sim_data)
        
        if obj_func is None or obj_func not in list(self.obj_func_out.keys()):
            obj_func = self.quest_objective

        # obj_func = 

        self.output_name = self.obj_func_out.get(obj_func, 'CALC_ACCURACY')
        preapply = time.perf_counter()

        sim_data[self.output_name] = sim_data.apply(obj_func, axis=1)
    
        
        if self.output_name == self.obj_func_out[self.star_mag]:
            logger.info('\tMean: {}'.format(sim_data[self.output_name].mean()))
        
        else:
            sim_data = sim_data[sim_data[self.output_name] >= 0]
        # raise ValueError
        # logger.critical(sim_data[self.output_name].mode())
        
        datastd = sim_data[self.output_name].std() / (np.sqrt(1 - 2/np.pi))
        # datamean = sim_data[self.output_name].mean()
        sim_data = sim_data[sim_data[self.output_name] <= 6*datastd]
        logger.debug('\n{}'.format(sim_data.head().to_string()))
        # raise ValueError
        # logger.critical(f'{c.RED}POST: {len(self.sim_data.index)}{c.DEFAULT}')
        # logger.critical(len(self.sim_data.index))
        # logger.debug(f'{c.RED}THRESHOLD: {5*datastd}{c.DEFAULT}')
        # logger.debug(f'{c.RED}Objective Function Time: {time.perf_counter() - preapply}{c.DEFAULT}')
        return sim_data

    def plot_data(self, data:pd.DataFrame=None, true_std:float=None)->None: # method to be overloaded
        
        if data is None:
            data = self.sim_data

        if true_std is None:
            true_std = self.half_normal_std( self.sim_data[self.output_name].std())
            
        fig = plt.figure()
        ax = fig.add_subplot()

        ax.hist(data[self.output_name], bins=int(np.sqrt(len(data.index))), density=True)
        ax.set_ylabel('Number of Runs')

        ax.set_xlabel('{}'.format(self.output_name.replace('_', ' ').title()))
        ax.set_title('{}\n{} +/- {}:\n{:,} Runs'.\
                    format(self.output_name.title().replace('_', ' '),
                           np.round(data[self.output_name].mean(),5),\
                           np.round(true_std,5),
                           len(data.index)), fontsize=30)

        param_fig = plt.figure()
        param_fig.suptitle('Input Parameter Distribution', fontsize=40)
        # cam_sw_params = {**self.camera.params, **self.software.params}
        
        cam_sw_params    = {"Number of Stars": self.camera.sensor,
                         r"$\eta_X$ [px]":self.software.dev_x,
                         r"$\eta_Y$ [px]":self.software.dev_y,
                         r'$\epsilon_x$ [px]':self.camera.eps_x,
                         r"$\epsilon_y$ [px]":self.camera.eps_y,
                         r"$\epsilon_z$ [px]":self.camera.eps_z,
                         r"$\phi$ [deg]":self.camera.phi,
                         r"$\theta$ [deg]":self.camera.theta,
                         r"$\psi$ [deg]":self.camera.psi}

        size = (3,3)
        
        for i, param in enumerate(list(cam_sw_params.keys())):

            param_data = data[cam_sw_params[param].name]

            param_ax = param_fig.add_subplot(size[0], size[1], i+1)

            if i%3 == 0: param_ax.set_ylabel('Number of Runs')

            param_ax.hist(param_data, bins=int(np.sqrt(len(data.index))))
            param_ax.set_title('{}:\n{} +/- {}'.\
                                format(param,\
                                        np.round(param_data.mean(), 3),\
                                        np.round(param_data.std(), 3)), fontsize=15)

            if param == 'FOCAL_LENGTH':
                param_ax.axvline(self.camera.f_len.ideal, color='r', label='True Focal Length ({} px)'.format(np.round(self.camera.f_len.ideal,3)))

        return

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

        # Real Star Catalog
        # logger.critical(sim_row)
        create_proj = time.perf_counter()
        projection = StarProjection(sim_row, self.catalog, self.type)
        post_proj = time.perf_counter()
        # visible = NoiseModel(projection.frame, sim_row)

        # Random Stars in FOV
        # projection = RandomProjection(sim_row)

        if len(projection.frame.index) <= 1:
            return -1

        # update data for number of stars generated
        sim_row.NUM_STARS = len(projection.frame.index)

        # QUEST
        eci_real = projection.frame['ECI_TRUE'].to_numpy()
        cv_real = projection.frame['CV_TRUE'].to_numpy()
        cv_est = projection.frame['CV_MEAS'].to_numpy()

        pre_quest = time.perf_counter()
        quest = QUEST(eci_real, cv_real, cv_est, q=projection.quat_real)
        post_quest = time.perf_counter()

        # logger.debug(f'{c.RED}Projection: {post_proj - create_proj}{c.DEFAULT}')
        # logger.debug(f'{c.RED}QUEST: {post_quest - pre_quest}{c.DEFAULT}')

        try:
            q_diff = quest.calc_acc()    

        except ValueError:
            print('{}SIM DATA:{}\n{}'.format(c.GREEN, c.DEFAULT, sim_row))
            
            return -1

        return q_diff
    
    """
    PHI MODEL
    """
   
    def phi_model(self, sim_row:pd.DataFrame)->float:
        """
        evaluation of hardware by passing it through QUEST.
        Exploitation of Camera Pinhole Model        

        Args:
            sim_row (pd.DataFrame): row item from data set containing information about star tracker hardware and centroiding ability

        Returns:
            float: calculated accuracy

        """

        # Real Star Catalog
        projection = StarProjection(sim_row, self.catalog)
        # Random Stars in FOV
        # projection = RandomProjection(sim_row)

        if len(projection.frame.index) <= 1:
            logger.critical(f'{c.RED}MAGNITUDE FAILURE{c.DEFAULT}')
            return -1

        # update data for number of stars generated
        sim_row.NUM_STARS = len(projection.frame.index)

        # QUEST
        # eci_real = projection.frame['ECI_TRUE'].to_numpy()
        cv_real = projection.frame['CV_TRUE'].to_numpy()
        cv_est = projection.frame['CV_MEAS'].to_numpy()

        dot_prod = np.array([real.dot(measure) for real, measure in zip(cv_real, cv_est)])
        dot_prod[np.isclose(dot_prod, 1, 1e-7)] = 1
        separation = np.mean(3600 * np.rad2deg(np.arccos(dot_prod)))

        return separation

    def star_mag(self, sim_row:pd.Series)->float:

        projection = StarProjection(sim_row, self.catalog, self.type)
        
        # sim_row.NUM_STARS_PRE = projection.pre_numstar
        # sim_row.NUM_STARS_VIS = len(projection.frame.index)
        
        # sim_row.MAX_MAG_PRE = projection.pre_maxmag
        # sim_row.MAX_MAG_VIS = projection.frame.v_magnitude.max()

        return projection.frame.v_magnitude.max()

