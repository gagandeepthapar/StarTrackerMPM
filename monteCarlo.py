import argparse
from datetime import datetime

import numpy as np
import pandas as pd
from alive_progress import alive_bar

import constants as c
from simObjects.Parameter import Parameter
from simObjects.StarTracker import StarTracker
import simulate as sim

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
    sim._est_StarTracker_Accuracy(data_mean, data_std, numStars)
    print(c.NEWSECTION)

    return

def runMonteCarlo(cam:StarTracker, params:tuple[Parameter], numRuns:int=1_000)->pd.DataFrame:

    data = np.zeros(numRuns)    # preallocate space
    star_angle = np.zeros(numRuns)
    
    param_dict = {}
    for param in cam.all_params():
        param_dict[param.name] = np.zeros(numRuns)

    # start monte carlo analysis
    with alive_bar(numRuns, title='Running Monte Carlo Analysis') as bar:
        for i in range(numRuns):

            for param in cam.all_params():
                if param in params:
                    param.modulate()
                param_dict[param.name][i] = param.value    
                

            for param in params:
                param.modulate()

            # set star incident angle [deg]
            angle = np.random.uniform(-10, 10)
            # angle = 8.5

            # store data in arrays
            star_angle[i] = angle
            data[i] = sim.sunEtalHardwareFunc(cam, star_angle=angle)
            
            bar()

    # store data in frame
    param_dict['StarAngle'] = star_angle
    param_dict['Mismatch'] = data

    return pd.DataFrame(param_dict)

def monteCarlo(cam:StarTracker, *params:Parameter, numRuns:int=1_000, saveFramePath:str=None)->None:

    print(c.NEWSECTION)
    print(cam)  # print cam info

    # if no params were passed in then modulate all params
    if len(params) == 0:
        params = (cam.f_len, cam.array_tilt, cam.distortion, cam.ppt_acc)

    for param in cam.all_params():
        if param in params:
            print('{}Modulating {}{}'.format(c.YELLOW, param.name, c.DEFAULT))
        else:
            print('{}Holding {}{}'.format(c.GREEN, param.name, c.DEFAULT))
    
    print()

    starframe = runMonteCarlo(cam=cam, params=params, numRuns=numRuns)

    if saveFramePath is not None:
        starframe.to_csv(saveFramePath)
        print('Saved Data to {}{}{}'.format(c.BLUE, saveFramePath.replace(c.curFile, '.'), c.DEFAULT))

    _dispMonteCarloResults(starframe=starframe, numStars=cam._num_stars)

    return

def handle_arguments(args:argparse.Namespace, camera:StarTracker)->None:

    # number of runs
    numRuns = args.n

    # save data
    if not args.s:
        args.fp = None
    
    # modulating params
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

        if args.ppt:
            params.append(camera.ppt_acc)
        
    monteCarlo(camera, *params, numRuns=numRuns, saveFramePath=args.fp)

    return

def parse_arguments()->argparse.Namespace:
    parser = argparse.ArgumentParser()

    defaultSave = 'DataFrame_{}.csv'.format(datetime.today().strftime('%Y_%m_%d'))

    parser.add_argument('-n', help="Number of Runs. Default 1000", type=int, default=1_000)
    parser.add_argument('-s', help='Save data', action='store_true')
    parser.add_argument('-fp', help='File path to save data', type=str, default=c.MEDIA+defaultSave)

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