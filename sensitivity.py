import argparse
import logging
import time

import numpy as np
import pandas as pd

import constants as c
from simObjects.Simulation import Simulation
from simObjects.AttitudeEstimation import QUEST, Projection
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker

class Sensitivity(Simulation):

    def __init__(self, camera:StarTracker, centroid:Parameter, orbit:Orbit, num_runs:int=1_000)->None:

        self.num_runs = num_runs
        super().__init__(camera, centroid, orbit)

        return
    
    def __repr__(self)->str:
        return 'Sensitivity Analysis: {} Data Points'.format(self.num_runs)