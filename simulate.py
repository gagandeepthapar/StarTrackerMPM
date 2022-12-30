import argparse
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from alive_progress import alive_bar

import constants as c
from disturbanceEffects.Parameter import Parameter
from disturbanceEffects.StarTracker import StarTracker


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

def _dispMonteCarloResults(starframe:pd.DataFrame, numStars:float)->None:

    data = starframe['Mismatch']
    star_angle = starframe['StarAngle']
    numRuns = len(starframe.index)

    data_mean = np.mean(data)
    data_std = np.std(data)

    print('\nSIM PARAMS:\n\t{:,} Runs\n\tMean Star Angle: {} +/- {}{}(1{})\n'.format(numRuns,
                                                                                    np.mean(star_angle),
                                                                                    np.std(star_angle),
                                                                                    c.DEG,
                                                                                    c.SIGMA))
    _est_StarTracker_Accuracy(data_mean, data_std, numStars)
    print(c.NEWSECTION)

    return

def _plotSurfaceSimResults(starframe:pd.DataFrame, param:Parameter, savePlot:bool=False)->None:
    
    numRuns = len(starframe.index)
    gridsize = int(np.sqrt(numRuns))

    param_space = np.reshape(np.array(starframe[param.name]), (gridsize, gridsize))
    angle_space = np.reshape(np.array(starframe['AngleSpace']), (gridsize, gridsize))
    data =        np.reshape(np.array(starframe['Data']), (gridsize, gridsize))

    maxAngle = starframe['AngleSpace'].iloc[-1]

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

    if savePlot:
        fname = c.EFFECTPLOTS + "{}_UnivariateEffect_{}".format(param.name, datetime.now().strftime("%Y_%m_%d_%H_%M"))
        plt.savefig(fname)
    
    return

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

def runMonteCarlo(cam:StarTracker, params:tuple[Parameter], numRuns:int=1_000)->pd.DataFrame:

    data = np.zeros(numRuns)    # preallocate space
    star_angle = np.zeros(numRuns)

    # start monte carlo analysis
    with alive_bar(numRuns, title='Running Monte Carlo Analysis') as bar:
        for i in range(numRuns):
            for param in params:
                param.modulate()

            # set star incident angle [deg]
            angle = np.random.uniform(-10, 10)
            # angle = 8.5

            # store data in arrays
            star_angle[i] = angle
            data[i] = sunEtalHardwareFunc(cam, star_angle=angle)
            
            bar()

    # store data in frame
    frame = {
                'StarAngle': star_angle,
                'Mismatch': data,
            }

    return pd.DataFrame(frame)

def runSurfaceSim(cam:StarTracker, param:Parameter, maxAngle:float=10, gridSize:int=1_000)->None:

    param_range = np.linspace(param.minRange, param.maxRange, gridSize)
    angle_range = np.linspace(0, maxAngle, gridSize)

    param_space, angle_space = np.meshgrid(param_range, angle_range)

    data = np.empty((gridSize, gridSize), dtype=float)

    simname = 'Mapping {} Surface:'.format(param.name)

    with alive_bar(gridSize*gridSize, title=simname) as bar:
        for i in range(gridSize):
            for j in range(gridSize):
                param.value = param_space[i][j]
                angle = angle_space[i][j]

                data[i][j] = sunEtalHardwareFunc(cam, angle)

                bar()
    
    frame = {
                'AngleSpace': angle_space.flatten(),
                param.name : param_space.flatten(),
                'Data': data.flatten()
            }

    return pd.DataFrame(frame)

def monteCarlo(cam:StarTracker, *params:Parameter, numRuns:int=1_000)->None:

    print(c.NEWSECTION)
    print(cam)  # print cam info

    # if no params were passed in then modulate all params
    if len(params) == 0:
        params = (cam.f_len, cam.array_tilt, cam.distortion, cam.ppt_acc, cam.ctr_acc)

    for param in params:
        print('Modulating {}'.format(param.name))
    print()

    starframe = runMonteCarlo(cam=cam, params=params, numRuns=numRuns)
    _dispMonteCarloResults(starframe=starframe, numStars=cam._num_stars)

    return

def surfaceSim(cam:StarTracker, *params:Parameter, maxAngle:float=10, numRuns:int=1_000, save:bool=False)->pd.DataFrame:

    # determine size of grid 
    gridSize = int(np.sqrt(numRuns))

    # set parameters if empty
    if len(params) == 0:
        params = (cam.f_len, cam.array_tilt, cam.distortion, cam.ppt_acc, cam.ctr_acc)

    # run surface sim
    for param in params:
        starframe = runSurfaceSim(cam=cam, param=param, maxAngle=maxAngle, gridSize=gridSize)
        _plotSurfaceSimResults(starframe, param=param, savePlot=save)
    
    plt.show()

    return

def parseArguments()->argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', help="Number of Runs. Default 1000", type=int, default=1_000)
    parser.add_argument('-mca', help='Run Monte Carlo Simulation', action='store_true')
    parser.add_argument('-surf', help='Run Surface Simulation', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()

    ideal_cam = StarTracker(cam_json=c.IDEAL_CAM)
    sim_cam = StarTracker(cam_json=c.SUNETAL_CAM)
    alvium_cam = StarTracker(cam_json=c.ALVIUM_CAM)

    if args.mca:
        # monteCarlo(ideal_cam, numRuns=args.n)

        sim_cam.reset_params()
        monteCarlo(sim_cam, numRuns=args.n)

        sim_cam.ctr_acc = alvium_cam.ctr_acc
        sim_cam.ctr_acc._color = c.YELLOW

        sim_cam.reset_params()
        monteCarlo(sim_cam, numRuns=args.n)

        # monteCarlo(alvium_cam, numRuns=args.n)

    if args.surf:
        surfaceSim(sim_cam, numRuns=args.n, save=True)

    