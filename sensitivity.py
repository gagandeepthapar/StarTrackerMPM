import argparse
import logging
import time

import numpy as np
from numpy.matlib import repmat
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy

import constants as c
from simObjects.Simulation import Simulation
# from simObjects.AttitudeEstimation import QUEST, RandomProjection
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.Software import Software
from simObjects.StarTracker import StarTracker

PER_PARAM = 200
NUM_PARAMS = 100
NUM_FRAMES = PER_PARAM//2
logger = logging.getLogger(__name__)

class Sense(Simulation):

    def __init__(self, camera: StarTracker, software: Software, orbit: Orbit, num_runs: int) -> None:
        num_runs = PER_PARAM * NUM_PARAMS
        super().__init__(camera=camera, software=software, orbit=orbit, num_runs=num_runs)
        self.type:str='SENS'
        return
    
    def run_sim(self, params, obj_func: callable, **kwargs) -> pd.DataFrame:
        # return super().run_sim(params, obj_func)
        # super().create_data()
        self.create_data()
        # print(params)
        # raise ValueError
        
        for param in params:
            if param in self.sim_data.columns:
                # logger.critical(param)
                # logger.critical(self.params[param])
                upper = self.params[param].stddev * 3
                paramset = np.linspace(-upper + self.params[param].ideal, upper + self.params[param].ideal, NUM_PARAMS)
                
                if param == 'BASE_DEV_X' or param == 'BASE_DEV_Y':
                    paramset = np.linspace(0, -1*upper, NUM_PARAMS)
                
                if param == 'MAX_MAGNITUDE':
                    id = self.camera.mag_sensor.ideal
                    diff = 1 * 3
                    paramset = np.linspace(id-diff, id+diff, NUM_PARAMS)
                
                paramdata = repmat(paramset, PER_PARAM, 1).flatten()
                self.sim_data[param] = paramdata
                # logger.critical(self.sim_data[param].min())

        # raise ValueError
        # return super().run_sim(params=params, obj_func=obj_func)

        if 'FOCAL_LENGTH' in params and 'MAX_MAGNITUDE' in params:            
            self.sim_data['MAX_MAGNITUDE'] = np.random.normal(self.camera.mag_sensor.ideal, 1, len(self.sim_data.index))

        # logger.critical(self.sim_data[params])
        # raise ValueError

        fin_df = pd.DataFrame()
        count = 0
        acc_col:str=None

        for i in range((int(len(self.sim_data.index)/NUM_FRAMES))):

            st = time.perf_counter()
            count += 1
            df = self.sim_data.iloc[i*NUM_FRAMES:(i+1)*NUM_FRAMES].copy()
            # logger.critical(df.to_string())
            df = super().run_sim(params=params, obj_func=obj_func, trueData=df.copy()).copy()
            if count == 1:
                acc_col = self.output_name
                fin_df = df
            
            else:
                fin_df = pd.concat([fin_df, df], axis=0)
            
            logger.info(f'Run:\n'
                        f'\tCount: {count}\n'
                        f'\tTime: {time.perf_counter() - st}\n'
                        f'\t\n\tSize: {len(fin_df.index)}\n'

                        f'\tSTD: {self.half_normal_std(fin_df[acc_col].std())}\n')

            # if count == 10:
            #     break
            
        self.sim_data = fin_df
        self.count = count
        return fin_df
    
    def create_data(self) -> pd.DataFrame:
        
        fov = np.deg2rad(10)

        # randomize all data from components
        q_data = pd.DataFrame({'RIGHT_ASCENSION': np.random.uniform(fov, 2*np.pi - fov, self.num_runs),
                              'DECLINATION': np.random.uniform(-np.pi/2 + fov, np.pi/2 - fov, self.num_runs),
                              'ROLL': np.random.uniform(-np.pi, np.pi, self.num_runs)})
        f_data = self.camera.ideal(num=self.num_runs)
        c_data = self.software.ideal(num=self.num_runs)
        o_data = self.orbit.ideal(num=self.num_runs)

        self.sim_data = pd.concat([q_data, f_data, c_data, o_data], axis=1)
        self.sim_data['MAX_MAGNITUDE'] = 20 + np.zeros(self.num_runs)
        
        # update focal_length based on temperature
        df_dtemp = self.sim_data['FOCAL_LENGTH'] * (self.sim_data['D_TEMP'] * self.sim_data['FOCAL_THERMAL_COEFFICIENT'])

        self.sim_data['FOCAL_LENGTH'] = self.sim_data['FOCAL_LENGTH'] + df_dtemp
        self.sim_data['D_FOCAL_LENGTH'] = self.sim_data['FOCAL_LENGTH'] - self.camera.f_len.ideal

        return self.sim_data