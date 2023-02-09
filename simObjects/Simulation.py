import sys

sys.path.append(sys.path[0] + '/..')

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

import constants as c
from simObjects.Software import Software
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker

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
            self.centroid = Software()

        if self.orbit is None:
            self.orbit = Orbit()

        if self.num_runs is None:
            self.num_runs = 1_000

        self.sim_data = pd.DataFrame()

        return

    def __repr__(self)->str:
        return '{} Data Points'.format(self.num_runs)


    def run_sim(self, obj_func:callable)->pd.DataFrame:  # method to be overloaded
        raise RuntimeError('run_sim not implemented for {} class'.format(type(self)))
        return 


    def plot_data(self)->None: # method to be overloaded
        raise RuntimeError('plot_data not implemented for {} class'.format(type(self)))
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
    