import argparse
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alive_progress import alive_bar

import constants as c
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker
import monteCarlo as mc
import sensitivityAnalysis as fac

def _est_StarTracker_Accuracy(mean:float, stddev:float, num_stars:float)->tuple[float]:

    stddev_range = 3*stddev

    data_minRange = mean - stddev_range
    data_maxRange = mean + stddev_range

    accMin = data_minRange/np.sqrt(num_stars)
    accMax = data_maxRange/np.sqrt(num_stars)

    print('\nSTAR TRACKER ACCURACY:')
    print('\t{}({}) +/- {}(1{}) Star Mismatch'.format(mean, c.MU, stddev, c.SIGMA))
    print('\t{}{}" ~ {}" 3{}-Accuracy{}'.format(c.GREEN,accMin, accMax, c.SIGMA, c.DEFAULT))

    return (accMin, accMax)

def sunEtalHardwareFunc(camera:StarTracker, star_angle:float=np.random.uniform(0, 10))->float:

    c = camera

    beta = np.deg2rad(star_angle)
    theta = np.deg2rad(c.array_tilt.value)
    
    f1 = (c.f_len.value + (c.ppt_acc.value * np.tan(theta)))/np.cos(theta + beta)
    f2 = c.ppt_acc.value/np.cos(theta)
    f3 = f2/c.f_len.ideal

    eA_a = np.arctan((f1*np.sin(beta) + f2 + c.ctr_acc.value + c.distortion.value)/c.f_len.ideal)
    eA_b = np.arctan(f3)
    eA_c = beta

    eA = eA_a - eA_b - eA_c

    return np.rad2deg(eA)*3600

def parseArguments()->argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', help="Number of Runs. Default 1000", type=int, default=1_000)
    parser.add_argument('-s', help='Save data', action='store_true')
    parser.add_argument('-fp', help='File path to save data', type=str, default=c.MEDIA)

    parser.add_argument('-mca', help='Run Monte Carlo Simulation', action='store_true')
    parser.add_argument('-surf', help='Run Surface Simulation', action='store_true')

    parser.add_argument('-all', help='Flag to modulate Focal Length', action='store_true')

    parser.add_argument('-flen', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-array', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-dist', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-ctr', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-ppt', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-temp', help='Flag to modulate Focal Length', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()

    # ideal_cam = StarTracker(cam_json=c.IDEAL_CAM)
    sim_cam = StarTracker(cam_json=c.SUNETAL_CAM)
    alvium_cam = StarTracker(cam_json=c.ALVIUM_CAM)

    if args.mca:
        mc.handle_arguments(args, sim_cam)
        mc.handle_arguments(args, alvium_cam)
    
    if args.surf:
        fac.handle_arguments(args, sim_cam)
