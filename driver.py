import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
import monte_carlo as mc
import sensitivity as sense
from copy import deepcopy

from simObjects.AttitudeEstimation import QUEST, Projection
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.Simulation import Simulation, SimThread
from simObjects.Software import Software
from simObjects.StarTracker import StarTracker

DEFAULT_RUNS = 1_000
THREAD_MAX = 10
logger = logging.getLogger('driver')

def set_logging_level(argument:str)->None:
    """
    Sets logging level from log argument    

    Args:
        argument (list[str]): argument from argparse; should indicate level

    Notes:
        logger level is set by multiples of 10 increasing with severity and starting at 10
    """
    argument = argument.upper()    
    
    levels = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
    level_dict = {level[0]:level for level in levels}   # creates dict of first letter to full name
    
    # matches string against either first letter or full name
    matched = level_dict.get(argument)
    if matched is None:
        level = (levels.index(argument) + 1) * 10
    else:
        level = (levels.index(matched) + 1) * 10

    logging.basicConfig(level=level)
    return

def parse_arguments()->argparse.Namespace:
    
    parser = argparse.ArgumentParser(prog='Star Tracker Measurement Process Model',
                                     description='Simulates a star tracker on orbit with hardware deviation, various software implementations, and varying environmental effects to analyze attitude determination accuracy and precision.',
                                    epilog='Contact Gagandeep Thapar @ gthapar@calpoly.edu for additional assistance')

    parser.add_argument('-log', '--logger', metavar='', type=str, help='Set logging level: (D)ebug, (I)nfo, (W)arning, (E)rror, (C)ritical. Default INFO.', default='INFO')

    parser.add_argument('-T', '--threads', metavar='', type=int, help='Number of threads to split task. Default 1, Max 10.', default=1)
    
    parser.add_argument('-sim', '--simulation', metavar='', type=str, help='Determine Sim Type: (M)onteCarlo or (S)ensitivityAnalysis. Default M, Monte Carlo Analysis.', default='M')
    parser.add_argument('-n', '--numberOfRuns', metavar='', type=int, help='Number of Runs (Monte Carlo) or Data Points Generated (Sensitivity Analysis). Default {:,}'.format(DEFAULT_RUNS), default=DEFAULT_RUNS)
    
    parser.add_argument('-cam', '--camera', metavar='', type=str, help='Star Tracker Hardware: (I)deal, (S)un etal, (A)lvium, (B)ad Camera, or path to JSON. Default Sun etal', default='S')
    parser.add_argument('-sw', '--software', metavar='', type=str, help='Centroid Software: (I)deal, (B)asic, (A)dvanced, or path to JSON. Default Basic', default='B')

    parser.add_argument('-par', '--parameters', metavar='', type=str, nargs='*',
                        help='Set parameters to analyze for Sensitivity Plot:\n \
                            (F)ocal Length; \
                            Focal (A)rray Inclination; \
                            (C)entroid Accuracy; \
                            (D)istortion; \
                            (P)rincipal Point Deviation; \
                            (T)emperature. Default Focal Length.', default='F')

    parser.add_argument('-p', '--plot', help='Plot Parameter/Input Data', action='store_true')

    args = parser.parse_args()
    
    set_logging_level(args.logger)  # set logging directly from argparse
    logger.debug('Arguments:\t{}'.format(args))
    
    return args

def setup_star_tracker(args_cam:str)->StarTracker:
    """
    Instantiate Star Tracker for simulation

    Args:
        args_cam (str): Star Tracker argument (model or path to model)

    Returns:
        StarTracker: Built star tracker
    """

    args_cam_upper = args_cam.upper()

    match args_cam_upper:

        case 'I' | 'IDEAL':
            return StarTracker(cam_json=c.IDEAL_CAM)
        
        case 'S' | 'SUN ETAL':
            return StarTracker(cam_json=c.SUNETAL_CAM, cam_name='Sun Etal')

        case 'A' | 'ALVIUM':
            return StarTracker(cam_json=c.ALVIUM_CAM, cam_name='Alvium')
        
        case 'B' | 'BAD':
            return StarTracker(cam_json=c.BAD_CAM, cam_name='Bad Camera')

        case _:
            return StarTracker(cam_json=args_cam)

def setup_software(args_sw:str)->Software:
    """
    Instantiate indicated Software class for data generation    

    Args:
        args_sw (str): cmd argument to create class

    Returns:
        Software: indicated Software class
    """

    args_sw_upper = args_sw.upper()

    match args_sw_upper:

        case 'I' | 'IDEAL':
            sw = Software(ctr_json=c.IDEAL_CENTROID)

        case 'B' | 'BASIC':
            sw = Software(ctr_json=c.SIMPLE_CENTROID)
        
        case 'A' | 'ADVANCED':
            sw = Software(ctr_json=c.LEAST_SQUARES_CENTROID)
        
        case _:
            sw = Software(ctr_json=args_sw)

    return sw

def setup_sim_class(sim_type:str, run_count:int, * , cam:StarTracker, centroid:Software, orbit:Orbit)->Simulation:
    """
    Instantiate indicated Simulation class for data generation    

    Args:
        sim_type (str): cmd argument to create class
        run_count (int): number of data points to generate for system

    Returns:
        Simulation: indicated Simulation class
    """

    sim_type = sim_type.upper()

    match sim_type:

        case 'M' | 'Monte Carlo':
            return  mc.MonteCarlo(cam, centroid, orbit, run_count)

        case 'S' | 'Sensitivity Analysis':
            return sense.Sensitivity(cam, centroid, orbit, run_count)

    return

def setup_params(sim_params:list[str])->list[str]:

    named_params = []

    for param in sim_params:
        param = param.upper()

        match param:

            case 'F' :
                named_params.append('FOCAL_LENGTH')

            case 'A':
                named_params.append('FOCAL_ARRAY_INCLINATION')

            case 'C':
                named_params.append('BASE_DEV_X')
                named_params.append('BASE_DEV_Y')

            case 'D':
                named_params.append('DISTORTION')
        
            case 'P':
                named_params.append('PRINCIPAL_POINT_ACCURACY')
        
            case 'T':
                named_params.append('TEMPERATURE')

    return named_params

def __calc_thread_idx(args_thread:int, sim_idx:int)->list[tuple]:
    """
    generate list of start and end indices to slice dataframe 

    Args:
        args_thread (int): number of threads to spin up
        sim_idx (int): total number of runs

    Returns:
        list[int]: list of start/end indices
    """

    args_thread = np.max([1, np.min([THREAD_MAX, np.abs(args_thread)])])    # calculate number of threads 1 <= T <= 10    
    num_rows = sim_idx//args_thread

    args_idx = [[i*num_rows, (i+1)*num_rows-1] for i in range(args_thread)]
    args_idx[-1][-1] = sim_idx-1

    return args_idx

def run_simulation(args_thread:int, args_numruns:int, full_sim:Simulation, params:list[str], obj_func)->pd.DataFrame:
    
    args_idx = __calc_thread_idx(args_thread, args_numruns)
    logger.critical(args_idx)

    start = time.perf_counter()
    if args_thread > 1:
        threads = list()
        cat_data = list()

        for i, arg_idx in enumerate(args_idx):
            thread = SimThread(full_sim, arg_idx[0], arg_idx[1], params, obj_func, i+1)
            threads.append(thread)
            thread.start()
        
        for i, thread in enumerate(threads):
            thread.join()
            cat_data.append(thread.thread_data)

        sim.sim_data = pd.concat(cat_data, axis=0, ignore_index=True)
        end = time.perf_counter()

    else:
        sim.run_sim(params, obj_func)
        end = time.perf_counter()

    logger.critical('\n\nTIME: {}\n\n'.format(end-start))

    return sim

if __name__ == '__main__':
    args = parse_arguments()    # parse command line arguments

    """ 
    SET SIMULATION PARAMETERS
    """
    camera = setup_star_tracker(args.camera)
    sw = setup_software(args.software)

    params = setup_params(args.parameters)

    sim = setup_sim_class(args.simulation, args.numberOfRuns, cam=camera, centroid=sw, orbit=None)
    
    logger.debug('Camera:\n{}'.format(sim.camera))
    logger.debug('Orbit:\n{}'.format(sim.orbit))
    logger.debug('Centroid:\n{}'.format(sim.centroid))
    logger.debug('Runs: {}'.format(sim.num_runs))

    """ 
    RUN SIMULATION
    """
    run_simulation(args.threads, args.numberOfRuns, sim, params, sim.sun_etal_hardware_analysis)
    # sim.run_sim(params)



    logger.info('\n{}'.format(sim.sim_data.columns))
    
    logger.debug('\n{}'.format(sim.sim_data))

    logger.info('MEAN: {}'.format(sim.sim_data['CALC_ACCURACY'].mean()))
    logger.info('STD: {}'.format(sim.sim_data['CALC_ACCURACY'].std()))

    """ 
    PLOT SIMULATION RESULTS
    """
    if args.plot:
        sim.plot_data(params)
        plt.show()

    logger.critical(sim.sim_data)
    logger.critical(sim.sim_data.iloc[1:5])