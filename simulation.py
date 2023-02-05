import numpy as np
import pandas as pd

from simObjects import AttitudeEstimation as ae
from simObjects.Orbit import Orbit
from simObjects.StarTracker import StarTracker


class Simulation:

    def __init__(self, camera:StarTracker=None,
                       software: ae.Projection=None,
                       orbit:Orbit=None)->None:

        self.star_tracker = camera
        self.centroid = software
        self.orbit = orbit

        return