import os
import numpy as np

""" SOURCE FOR UTIL FUNCS, FILEPATHS """

""" CURRENT FILE """
curFile = os.path.abspath(os.path.dirname(__file__))

""" CAMERA CONFIGS """
IDEAL_CAM = os.path.join(curFile, 'utils/', 'idealCamera.json')
SUNETAL_CAM = os.path.join(curFile, 'utils/', 'sunEtalCamera.json')
ALVIUM_CAM = os.path.join(curFile, 'utils/', 'alviumCamera.json')
BAD_CAM = os.path.join(curFile, 'utils/', 'badCamera.json')
""" SOFTWARE CONFIGS """
IDEAL_CENTROID = os.path.join(curFile, 'utils/', 'idealCentroid.json')
SIMPLE_CENTROID = os.path.join(curFile, 'utils/', 'simpleCentroid.json')
LEAST_SQUARES_CENTROID = os.path.join(curFile, 'utils/', 'leastSquaresCentroid.json')

IDEAL_IDENT = os.path.join(curFile, 'utils/', 'idealIdentification.json')
TYP_IDENT = os.path.join(curFile, 'utils/', 'typicalIdentification.json')

""" ORBIT CONFIGS """
ISSORBIT = os.path.join(curFile, 'utils/','issOrbit.json')
CUBESATS = os.path.join(curFile, 'utils/', 'celestrakCubeSats.json')
ACTIVESATS = os.path.join(curFile, 'utils/', 'celestrakActiveSats.json')

""" SATELLITE CONFIGS """
TEMP_DATA = os.path.join(curFile, 'utils/', 'TEMPERATURE_DATA.pkl')

""" CATALOG INFO """
BSC5PKL = os.path.join(curFile, 'utils/', 'BSC5PKL.pkl')
BSC5BIN = os.path.join(curFile, 'utils/', 'BSC5')

""" CATALOG INFO """
BSC5PKL = os.path.join(curFile, 'utils/', 'BSC5PKL.pkl')
BSC5BIN = os.path.join(curFile, 'utils/', 'BSC5')

""" MEDIA """
MEDIA = os.path.join(curFile, 'media/')
EFFECTPLOTS = os.path.join(MEDIA, 'effectAnalysis/')
TOOLSCREENS = os.path.join(MEDIA, 'inAction/')

""" SYMBOLS """
MU = "\u03BC"
SIGMA = "\u03C3"
DEG = "\u00B0"
NEWLINE = '*'*100

""" COLORS """
MAGENTA = '\033[95m'
BLUE = '\033[94m'
CYAN = '\033[96m'
GREEN = '\033[92m'
YELLOW = '\033[93m'
RED = '\033[91m'
DEFAULT = '\033[0m'    

""" FREQ USE """
NEWSECTION = '{}{}{}'.format(RED, NEWLINE, DEFAULT)

""" USEFUL FUNCS """
def cosd(theta:float)->float:
    return np.cos(np.deg2rad(theta))

def sind(theta:float)->float:
    return np.sin(np.deg2rad(theta))

def tand(theta:float)->float:
    return np.tan(np.deg2rad(theta))

def acosd(val:float)->float:
    return np.rad2deg(np.arccos(val))

def asind(val:float)->float:
    return np.rad2deg(np.arcsin(val))

def atand(val:float)->float:
    return np.rad2deg(np.arctan(val))

""" CONSTANTS """
SOLARFLUX = 1367
EARTHFLUX = 213
EARTHMU = 398600
EARTHRAD = 6378
AU = 149597870.691
J2000 = 2451545
JYEAR = 365.25
STEFBOLTZ = 5.670367e-8
C1U = 0.01
C3U = 0.03
TEMP_GAMMA = 0.273 