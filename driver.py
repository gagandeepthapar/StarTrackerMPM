import argparse
import logging
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
import monte_carlo as mc
import sensitivity as sense

from simObjects.AttitudeEstimation import QUEST, Projection
from simObjects.Orbit import Orbit
from simObjects.Parameter import Parameter
from simObjects.Simulation import Simulation
from simObjects.Software import Software
from simObjects.StarTracker import StarTracker

DEFAULT_RUNS = 1_000
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
    """
    Parse arguments from command line (intended method to run)    

    Returns:
        argparse.Namespace: struct containing cmd arguments
    """
    parser = argparse.ArgumentParser(prog='Star Tracker Measurement Process Model',
                                     description='Simulates a star tracker on orbit with hardware deviation, various software implementations, and varying environmental effects to analyze attitude determination accuracy and precision.',
                                    epilog='Contact Gagandeep Thapar @ gthapar@calpoly.edu for additional assistance')

    parser.add_argument('-func', '--function', metavar='', type=str, help='Set Objective Function: (Q)uest First Principles, (S)un Et. al Star Location Error, (C)ombine QUEST with Sun Et. al Errors. Default Q.', default='Q')

    parser.add_argument('-log', '--logger', metavar='', type=str, help='Set logging level: (D)ebug, (I)nfo, (W)arning, (E)rror, (C)ritical. Default INFO.', default='INFO')

    # parser.add_argument('-T', '--threads', metavar='', type=int, nargs=1, help='Number of threads to split task. Default 1.', default=1)
    
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
            sw = Software(ctr_json=c.IDEAL_CENTROID, id_json=c.IDEAL_IDENT)

        case 'B' | 'BASIC':
            sw = Software(ctr_json=c.SIMPLE_CENTROID, id_json=c.TYP_IDENT)
        
        case 'A' | 'ADVANCED':
            sw = Software(ctr_json=c.LEAST_SQUARES_CENTROID, id_json=c.TYP_IDENT)
        
        case _:
            sw = Software(ctr_json=args_sw)

    return sw

def setup_sim_class(sim_type:str, run_count:int, * , cam:StarTracker, centroid:Software, orbit:Orbit, mod_params:list[str])->Simulation:
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
            return sense.Sensitivity(mod_params, cam, centroid, orbit, run_count)

    return

def setup_params(sim_params:list[str])->list[str]:
    """
    Determine list of parameters to modify according to cmd arguments    

    Args:
        sim_params (list[str]): cmd argument containing parameter abbreviations

    Returns:
        list[str]: list of full parameter names that can be called in Simulation class
    """
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

def setup_objective(simulation:Simulation, func_str:str):
    """
    Set the objective function to calculate accuracy of star tracker

    Args:
        simulation (Simulation): sim object (pre-created)
        func_str (str): argument from command line
    """

    func_str = func_str.upper()

    match func_str:

        case 'Q':
            obj_func = simulation.quest_objective

        case 'S':
            obj_func = simulation.sun_etal_star_mismatch
            
        case 'C':
            obj_func = simulation.sun_etal_quest

        case _:
            raise NameError('Improper Input. Either Q/S/C for QUEST, Sun Et. Al Mismatch, Combination of QUEST and Sun Et Al., respectively')

    return obj_func

if __name__ == '__main__':
    args = parse_arguments()    # parse command line arguments

    """ 
    SET SIMULATION PARAMETERS
    """
    camera = setup_star_tracker(args.camera)
    sw = setup_software(args.software)

    params = setup_params(args.parameters)

    sim = setup_sim_class(args.simulation, args.numberOfRuns, cam=camera, centroid=sw, orbit=Orbit(), mod_params=params)
    
    logger.debug('Camera:\n{}'.format(sim.camera))
    logger.debug('Orbit:\n{}'.format(sim.orbit))
    logger.debug('Software:\n{}'.format(sim.software))
    logger.debug('Runs: {}'.format(sim.num_runs))
    obj_func = setup_objective(sim, args.function)


    """ 
    RUN SIMULATION
    """
    logger.info('\n{}'.format(sim.sim_data.columns))
    
    logger.debug('\n{}'.format(sim.sim_data))
    logger.debug('\n{}'.format(sim.sim_data[['BASE_DEV_X', 'BASE_DEV_Y']]))

    logger.info('\n\n{}\n\n'.format(sim.sim_data['CALC_ACCURACY']))
    df = sim.run_sim(params=params, obj_func=obj_func)

    logger.info('MEAN: {}'.format(sim.sim_data['CALC_ACCURACY'].mean()))
    logger.info('STD: {}'.format(sim.sim_data['CALC_ACCURACY'].std()))

    """ 
    PLOT SIMULATION RESULTS
    """
    if args.plot:
        sim.plot_data(params)
        plt.show()
