import argparse

import numpy as np
import pandas as pd

from simObjects.Parameter import Parameter
from simObjects.AttitudeEstimation import QUEST, Projection
from simObjects.Orbit import Orbit
from simObjects.StarTracker import StarTracker

import constants as c 
import time
import logging

logger = logging.getLogger(__name__)

class Simulation:

    def __init__(self, camera:StarTracker=StarTracker(cam_json=c.SUNETAL_CAM),
                       centroid_accuracy:Parameter=Parameter(0, 0.1, 0, name='Centroid_Accuracy', units='px'),
                       orbit:Orbit=Orbit(),
                       num_runs:int=1_000)->None:

        self.star_tracker = camera
        self.centroid = centroid_accuracy
        self.orbit = orbit

        self.sim_data = self.generate_data(num_runs)
        return

    def generate_data(self, num_runs:int)->pd.DataFrame:
        
        s = time.perf_counter()
        odata = self.orbit.randomize(num=num_runs)
        fdata = self.star_tracker.randomize(num=num_runs)

        cdata = pd.DataFrame()        
        cdata['BASE_DEV_X'] = self.centroid.modulate(num_runs)
        cdata['BASE_DEV_Y'] = self.centroid.modulate(num_runs)


        data = pd.concat([odata, fdata, cdata], axis=1)

        self.sim_data = data
        logging.log(logging.DEBUG,'Data Gen Time: {}'.format(time.perf_counter() - s))
        return data 

def parse_arguments()->argparse.Namespace:
    
    parser = argparse.ArgumentParser(prog='Star Tracker Measurement Process Model',
                                     description='Simulates a star tracker on orbit with hardware deviation, various software implementations, and varying environmental effects to analyze attitude determination accuracy and precision.',
                                    epilog='Contact Gagandeep Thapar @ gthapar@calpoly.edu for additional assistance')

    parser.add_argument('-log', metavar='', type=str, nargs=1, help='Set logging level (DEBUG/INFO/WARNING/ERROR/CRITICAL); use first letter or full name. Default INFO.', default='INFO')

    parser.add_argument('-T', '--threads', metavar='', type=int, nargs=1, help='Number of threads to split task. Default 1.', default=1)
    
    parser.add_argument('-n', '--NumberRuns', metavar='', type=int, help='Number of Runs (Monte Carlo Analysis). Default 1,000', default=1_000)
    parser.add_argument('-mca', '--MonteCarlo', action='store_true', help='Run Monte Carlo Analysis')
    parser.add_argument('-surf', '--Sensitivity', action='store_true', help='Run Sensitivity Analysis')


    return parser.parse_args()

def set_logging_level(argument:list[str])->None:
    """
    Sets logging level from log argument    

    Args:
        argument (list[str]): argument from argparse; should indicate level

    Notes:
        logger level is set by multiples of 10 increasing with severity and starting at 10
    """

    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']

    argument = argument[0].upper()    
    level_dict = {level[0]:level for level in levels}   # creates dict of first letter to full name
    
    # matches string against either first letter or full name
    matched = level_dict.get(argument)
    if matched is None:
        level = (levels.index(argument) + 1) * 10
    else:
        level = (levels.index(matched) + 1) * 10

    logging.basicConfig(level=level)
    return

if __name__ == '__main__':
    args = parse_arguments()

    set_logging_level(args.log)

    logger.debug('Arguments: {}'.format(args))
    sim = Simulation(num_runs=args.NumberRuns)

    logger.debug(sim.sim_data.columns) 
    logger.info(sim.sim_data.head())
    