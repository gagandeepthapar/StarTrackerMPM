import os

import numpy as np
import pandas as pd

""" SOURCE FOR UTIL FUNCS, FILEPATHS """

""" CURRENT FILE """
curFile = os.path.abspath(os.path.dirname(__file__))

""" CAMERA CONFIGS """
simCameraSunEtal = os.path.join(curFile, 'utils/', 'simCameraSunEtAl.json')

""" SATELLITE/MATERIAL PROPERTIES """

""" MEDIA """
MEDIA = os.path.join(curFile, 'media/')

""" SYMBOLS """
MU = "\u03BC"
SIGMA = "\u03C3"
DEG = "\u00B0"