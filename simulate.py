import argparse
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from alive_progress import alive_bar

import constants as c
from disturbanceEffects.hardwareEffects import *


def sunEtalHardwareFunc(camera:Camera, star_angle:float=0)->float:

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

    return eA*3600

def monteCarlo(cam:Camera, numRuns:int=1_000):

    data = np.zeros((numRuns,1))
    cam.reset_params()
    print(cam)

    with alive_bar(numRuns, title='Running Monte Carlo Analysis') as bar:
        for i in range(numRuns):
            cam.modulate_array_tilt()
            cam.modulate_centroid()
            cam.modulate_distortion()
            cam.modulate_focal_length()
            cam.modulate_principal_point_accuracy()

            data[i] = sunEtalHardwareFunc(cam, 8.5)
            bar()

    data_mean = np.mean(data)
    data_std = np.std(data)

    data_minRange = data_mean - (3*data_std)
    data_maxRange = data_mean + (3*data_std)

    accMin = data_minRange/np.sqrt(cam._num_stars)
    accMax = data_maxRange/np.sqrt(cam._num_stars)

    print('\n{}({}) +/- {}(1{})'.format(data_mean,c.MU, data_std, c.SIGMA))
    print('{}" ~ {}" 3{}-Accuracy\n'.format(accMin, accMax, c.SIGMA))

    return

def surfaceSim(cam:Camera, param:Parameter, maxAngle:float=10, numRuns:int=1_000, save:bool=False)->None:

    numRuns = int(np.sqrt(numRuns))

    cam.reset_params()
    print('Modulating {}'.format(param))

    param_range = np.linspace(param.minRange, param.maxRange, numRuns)
    angle_range = np.linspace(0, maxAngle, numRuns).reshape(numRuns,1)

    param_space = np.tile(param_range, (numRuns,1))
    angle_space = np.tile(angle_range, (1, numRuns))

    data = np.empty((numRuns, numRuns), dtype=float)

    simname = '{} Surface:'.format(param.name)

    with alive_bar(numRuns*numRuns, title=simname) as bar:
        for i in range(numRuns):
            for j in range(numRuns):
                param.value = param_space[i][j]
                angle = angle_space[i][j]

                data[i][j] = sunEtalHardwareFunc(cam, angle)

                bar()
    
    xlbl = '{}: {} +/- {} (3{})'.format(param.name, param.ideal, 3*param._err_stddev, c.SIGMA)
    ylbl = 'Incident Angle: 0 - {}{}'.format(maxAngle, c.DEG)
    zlbl = 'Accuracy (")'
    title = '{} x Incident Angle'.format(param.name)

    # setup plot
    warnings.filterwarnings("ignore")
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    fig.add_axes(ax)

    fig.set_size_inches(8,6)
    ax.plot_surface(param_space-param.ideal, angle_space, data, cmap='plasma')

    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)
    ax.set_title(title)

    if save:
        fname = c.MEDIA + "{}_UnivariateEffect_{}".format(param.name, datetime.now().strftime("%Y_%m_%d_%H_%M"))
        plt.savefig(fname)
    
    print()

    return

def parseArguments()->argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', help="Number of Runs. Default 1000", type=int, default=1_000)
    parser.add_argument('-mca', help='Run Monte Carlo Simulation', action='store_true')
    parser.add_argument('-surf', help='Run Surface Simulation', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()

    ideal_cam = Camera(cam_json=c.IDEAL_CAM)
    sim_cam = Camera(cam_json=c.SUNETAL_CAM)
    alvium_cam = Camera(cam_json=c.ALTIUM_CAM)

    alvium_cam._num_stars = 7

    if args.mca:
        monteCarlo(cam=ideal_cam, numRuns=args.n)
        monteCarlo(cam=sim_cam, numRuns=args.n)
        monteCarlo(cam=alvium_cam, numRuns=args.n)

    if args.surf:
        surfaceSim(cam=sim_cam, param=sim_cam.f_len, numRuns=args.n)
        surfaceSim(cam=sim_cam, param=sim_cam.ctr_acc, numRuns=args.n)
        surfaceSim(cam=sim_cam, param=sim_cam.ppt_acc, numRuns=args.n)
        surfaceSim(cam=sim_cam, param=sim_cam.array_tilt, numRuns=args.n)
        surfaceSim(cam=sim_cam, param=sim_cam.distortion, numRuns=args.n)
