import argparse

import numpy as np
import pandas as pd

from simObjects.AttitudeEstimation import QUEST, Projection
from simObjects.Orbit import Orbit
from simObjects.StarTracker import StarTracker

import constants as c 
import time

class Simulation:

    def __init__(self, camera:StarTracker=StarTracker(cam_json=c.SUNETAL_CAM),
                       software: Projection=QUEST(),
                       orbit:Orbit=Orbit(),
                       num_runs:int=1_000)->None:

        self.star_tracker = camera
        self.centroid = software
        self.orbit = orbit

        self.sim_data = self.generate_data(num_runs)
        return

    def generate_data(self, num_runs:int)->pd.DataFrame:
        
        s = time.perf_counter()
        odata = self.orbit.randomize(num=num_runs)
        fdata = self.star_tracker.randomize(num=num_runs)

        data = pd.concat([odata, fdata], axis=1)

        self.sim_data = data
        print(time.perf_counter() - s)
        return data 

def parse_arguments()->argparse.Namespace:
    
    parser = argparse.ArgumentParser(prog='Star Tracker Measurement Process Model',
                                     description='Simulates a star tracker on orbit with hardware deviation, various software implementations, and varying environmental effects to analyze attitude determination accuracy and precision.',
                                    epilog='Contact Gagandeep Thapar @ gthapar@calpoly.edu for additional assistance')

    parser.add_argument('-T', '--threads', metavar='', type=int, nargs=1, help='Number of threads to split task. Default 1.', default=1)
    
    parser.add_argument('-n', '--NumberRuns', metavar='', type=int, help='Number of Runs (Monte Carlo Analysis). Default 1,000', default=1_000)
    parser.add_argument('-mca', '--MonteCarlo', action='store_true', help='Run Monte Carlo Analysis')
    parser.add_argument('-surf', '--Sensitivity', action='store_true', help='Run Sensitivity Analysis')


    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    print(args)
    sim = Simulation(num_runs=args.NumberRuns)
    print(sim.sim_data)