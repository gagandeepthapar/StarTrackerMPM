import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
from simObjects.Simulation import Simulation
# from simObjects.AttitudeEstimation import QUEST, RandomProjection
from simObjects.Software import Software
from simObjects.Orbit import Orbit
# from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker

logger = logging.getLogger(__name__)
class MonteCarlo(Simulation):

    def __init__(self, camera:StarTracker, software:Software, orbit:Orbit, num_runs:int)->None:
        super().__init__(camera=camera, software=software, orbit=orbit, num_runs=num_runs)

        self.type:str='MC'

        return
    
    def __repr__(self)->str:
        return 'Monte Carlo Analysis: '+super().__repr__()

    def run_sim(self, params, obj_func:callable, **kwargs) -> pd.DataFrame:
        # pre_createdata = time.perf_counter()
        # self.create_data()
        # logger.debug(f'{c.RED}Generate Data: {time.perf_counter() - pre_createdata}{c.DEFAULT}')
        # return super().run_sim(params=params, obj_func=obj_func)
        
        maxRuns = kwargs.get('maxRuns', -1)

        # print(params, obj_func, kwargs)
        # raise ValueError

        fin_df = pd.DataFrame()
        acc_col:str=None
        count = 0
        
        std_ratio = 1
        prev_std = 1

        while std_ratio > 1e-5 or len(fin_df.index) < 10_000:
            st = time.perf_counter()
            count += 1
            df = self.create_data()
            df = super().run_sim(params=params, obj_func=obj_func).copy()
            if count == 1:
                acc_col = self.output_name
                fin_df = df
            
            else:
                fin_df = pd.concat([fin_df, df], axis=0)
            
            std_ratio = np.abs((prev_std - self.half_normal_std(fin_df[acc_col]).std())/prev_std)
            prev_std = self.half_normal_std(fin_df[acc_col].std())

            # std_ratio = np.abs((prev_std - fin_df[acc_col].std())/prev_std)
            # prev_std= fin_df[acc_col].std()
            
            logger.info(f'Run:\n'
                        f'\tCount: {count}\n'
                        f'\tTime: {time.perf_counter() - st}\n'
                        f'\t\n\tSize: {len(fin_df.index)}\n'
                        # f'\tMean: {(fin_df[acc_col].mean())}\n'
                        f'\tFull STD: {prev_std}\n'
                        f'\tRatio: {std_ratio}')

            if maxRuns > 0 and len(fin_df.index) >= maxRuns:
                logger.critical(f'{c.RED}MAXIMUM RUNS REACHED{c.DEFAULT}')
                break
        
        self.sim_data = fin_df
        self.count = count
        return fin_df

    def plot_data(self, data:pd.DataFrame, *kwargs) -> None:
        return super().plot_data(data, *kwargs)
    
    def create_data(self)->pd.DataFrame:

        fov = np.deg2rad(10)

        # randomize all data from components
        q_data = pd.DataFrame({'RIGHT_ASCENSION': np.random.uniform(fov, 2*np.pi - fov, self.num_runs),
                              'DECLINATION': np.random.uniform(-np.pi/2 + fov, np.pi/2 - fov, self.num_runs),
                              'ROLL': np.random.uniform(-np.pi, np.pi, self.num_runs)})
        f_data = self.camera.randomize(num=self.num_runs)
        c_data = self.software.randomize(num=self.num_runs)
        o_data = self.orbit.randomize(num=self.num_runs)

        self.sim_data = pd.concat([q_data, f_data, c_data], axis=1)
        
        # update focal_length based on temperature
        # df_dtemp = self.sim_data['FOCAL_LENGTH'] * (self.sim_data['D_TEMP'] * self.sim_data['FOCAL_THERMAL_COEFFICIENT'])

        # self.sim_data['FOCAL_LENGTH'] = self.sim_data['FOCAL_LENGTH'] + df_dtemp
        # self.sim_data['D_FOCAL_LENGTH'] = self.sim_data['FOCAL_LENGTH'] - self.camera.f_len.ideal

        return self.sim_data
