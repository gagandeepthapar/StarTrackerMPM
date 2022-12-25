import numpy as np
import matplotlib.pyplot as plt
import argparse
from alive_progress import alive_bar
from disturbanceEffects.hardwareEffects import *
import constants as c
import warnings

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

def monteCarlo(cam:Camera, numRuns:int=1):

    data = np.zeros((numRuns,1))
    cam.reset_params()

    with alive_bar(numRuns, title='Running Monte Carlo Analysis') as bar:
        for i in range(numRuns):
            cam.modulate_principal_point_accuracy()
            data[i] = sunEtalHardwareFunc(cam, 8.5)
            bar()
    
    print('\n{}({}) +/- {}(3{})\n'.format(np.mean(data),c.MU, np.std(data), c.SIGMA))

    return

def surfaceSim(cam:Camera, param:Parameter, maxAngle:float=10, numRuns:int=1_000_000)->None:

    numRuns = int(np.sqrt(numRuns))

    cam.reset_params()
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

    warnings.filterwarnings("ignore")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    fig.set_size_inches(10, 12)
    ax.plot_surface(param_space, angle_space, data, cmap='plasma')

    ax.set_xlabel(xlbl)
    ax.set_ylabel(ylbl)
    ax.set_zlabel(zlbl)
    ax.set_title(title)

    fname = c.MEDIA + "{}_UnivariateEffect".format(param.name)
    plt.savefig(fname)

    return ax

def parseArguments()->argparse.Namespace:

    parser = argparse.ArgumentParser()

    parser.add_argument('-n', help="Number of Runs. Default 1", type=int, default=1)
    # parser.add_argument('--type', help="Sim to run. Default Monte Carlo", type=str, default='m')

    return parser.parse_args()

if __name__ == '__main__':
    args = parseArguments()
    cam = Camera()

    # monteCarlo(cam=cam, numRuns=args.n)

    surfaceSim(cam=cam, param=cam.f_len, numRuns=args.n)
    surfaceSim(cam=cam, param=cam.ctr_acc, numRuns=args.n)
    surfaceSim(cam=cam, param=cam.array_tilt, numRuns=args.n)
    surfaceSim(cam=cam, param=cam.distortion, numRuns=args.n)
    surfaceSim(cam=cam, param=cam.ppt_acc, numRuns=args.n)

    plt.show()