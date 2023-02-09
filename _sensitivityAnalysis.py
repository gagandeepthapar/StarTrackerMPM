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
import _simulate as sim

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

    plt.locator_params(axis='x', nbins=5)
    plt.locator_params(axis='y', nbins=5)
    plt.locator_params(axis='z', nbins=5)

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

def runSurfaceSim(cam:StarTracker, param:Parameter, maxAngle:float=10, gridSize:int=1_000)->pd.DataFrame:

    param_range = np.linspace(param.minRange, param.maxRange, gridSize)
    angle_range = np.linspace(0, maxAngle, gridSize)

    param_space, angle_space = np.meshgrid(param_range, angle_range)

    data = np.empty((gridSize, gridSize), dtype=float)

    simname = '{}Mapping {} Surface:{}'.format(c.YELLOW, param.name, c.DEFAULT)

    with alive_bar(gridSize*gridSize, title=simname) as bar:
        for i in range(gridSize):
            for j in range(gridSize):
                param.value = param_space[i][j]
                angle = angle_space[i][j]

                data[i][j] = sim.sunEtalHardwareFunc(cam, angle)

                bar()
    
    frame = {
                'AngleSpace': angle_space.flatten(),
                param.name : param_space.flatten(),
                'Data': data.flatten()
            }

    return pd.DataFrame(frame)

def surfaceSim(cam:StarTracker, *params:Parameter, maxAngle:float=10, numRuns:int=1_000, save:bool=False)->None:

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

def handle_arguments(args:argparse.Namespace, camera:StarTracker)->None:

    if args.all:
        params = camera.all_params()
    else:
        params = []
        if args.flen:
            params.append(camera.f_len)
        
        if args.array:
            params.append(camera.array_tilt)
        
        if args.dist:
            params.append(camera.distortion)
        
        if args.ctr:
            params.append(camera.ctr_acc)

        if args.ppt:
            params.append(camera.ppt_acc)

    surfaceSim(camera, *params ,numRuns=args.n, save=args.s)

    return

def parse_arguments()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-n', help="Number of Runs. Default 1000", type=int, default=1_000)
    parser.add_argument('-s', help='Save data', action='store_true')
    parser.add_argument('-fp', help='File path to save data', type=str, default=c.MEDIA)

    parser.add_argument('-all', help='Flag to modulate Focal Length', action='store_true')

    parser.add_argument('-flen', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-array', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-dist', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-ctr', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-ppt', help='Flag to modulate Focal Length', action='store_true')
    parser.add_argument('-temp', help='Flag to modulate Focal Length', action='store_true')

    return parser.parse_args()

if __name__ == '__main__':
    
    cam = StarTracker(cam_json=c.ALVIUM_CAM)

    args = parse_arguments()
    handle_arguments(args, cam)  # calls monteCarlo