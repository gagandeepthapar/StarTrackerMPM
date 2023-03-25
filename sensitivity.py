import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import constants as c
from simObjects.Simulation import Simulation
from simObjects.AttitudeEstimation import QUEST, Projection
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker

logger = logging.getLogger(__name__)

class Sensitivity(Simulation):

    def __init__(self, params:list[Parameter], camera:StarTracker, centroid:Parameter, orbit:Orbit, num_runs:int)->None:
        self.__mod_params = params
        super().__init__(camera, centroid, orbit, num_runs)
        return
    
    def __repr__(self)->str:
        return 'Sensitivity Analysis: {} Data Points'.format(self.num_runs)

    def run_sim(self, params, obj_func:callable) -> pd.DataFrame:
        self.__create_data(params)
        return super().run_sim(params, obj_func=obj_func)

    def plot_data(self, params:list[str]) -> None:
        if len(params) < 2:
            params.append('FOCAL_LENGTH') 
        
        if len(params) < 1:
            params.append('FOCAL_ARRAY_INCLINATION')    
            

        self.__plot_surface_data(params[0], params[1])
        super().plot_data()
        return 
    
    def __plot_surface_data(self, paramA:str, paramB:str)->None:

        row = deepcopy(self.sim_data.iloc[0].squeeze())

        paramA = self.params[paramA]
        paramB = self.params[paramB]

        param_a_range = np.linspace(paramA.minRange, paramA.maxRange, int(np.sqrt(self.num_runs)))
        param_b_range = np.linspace(paramB.minRange, paramB.maxRange, int(np.sqrt(self.num_runs)))

        p_a_mesh, p_b_mesh = np.meshgrid(param_a_range, param_b_range)
        acc = np.zeros(np.shape(p_a_mesh))

        for i in range(len(p_a_mesh)):
            for j in range(len(p_a_mesh[0])):
                row[paramA.name] = p_a_mesh[i][j]
                row[paramB.name] = p_b_mesh[i][j]
                
                acc[i][j] = self.sun_etal_star_mismatch(row)

        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        ax.plot_surface(p_a_mesh, p_b_mesh, acc, cmap='coolwarm')

        ax.set_xlabel(paramA.name)
        ax.set_ylabel(paramB.name)
        ax.set_zlabel('Calculated Accuracy [arcsec]')

        ax.set_title('Surface Map of Accuracy')

        return

    def __create_data(self, param_names:list[str])->pd.DataFrame:

        if type(param_names) is not list:
            param_names = [param_names]

        # idealize parameters from components
        f_data = self.camera.ideal(num=self.num_runs)
        c_data = self.software.ideal(num=self.num_runs)
        o_data = self.orbit.ideal(num=self.num_runs)

        self.sim_data = pd.concat([f_data, c_data,o_data], axis=1)

        # modulate all indicated params in argument
        for param in param_names:
            self.sim_data[param] = self.params[param].modulate(self.num_runs)
            
            if param == 'FOCAL_LENGTH': # update d_focal_length
                self.sim_data['D_FOCAL_LENGTH'] = self.sim_data['FOCAL_LENGTH'] - self.camera.f_len.ideal

        logger.info('params:\n{}\n{}'.format(self.sim_data.mean(), self.sim_data.std()))

        return self.sim_data
