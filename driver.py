import argparse
import logging
import time
import os
import numpy as np
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt

import constants as c
import monte_carlo as mc
import sensitivity as sense

# from simObjects.AttitudeEstimation import QUEST, RandomProjection
from simObjects.Orbit import Orbit
# from simObjects.Parameter import Parameter
from simObjects.Simulation import Simulation
from simObjects.Software import Software
from simObjects.StarTracker import StarTracker

DEFAULT_RUNS = 100
Z_SCORE = 1.96
ACC = 0.01
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

    parser.add_argument('-func', '--function', metavar='', type=str, help='Set Objective Function: (Q)uest First Principles or (P)hi Model Analysis. Default QUEST.', default='Q')

    parser.add_argument('-log', '--logger', metavar='', type=str, help='Set logging level: (D)ebug, (I)nfo, (W)arning, (E)rror, (C)ritical. Default INFO.', default='INFO')

    # parser.add_argument('-T', '--threads', metavar='', type=int, nargs=1, help='Number of threads to split task. Default 1.', default=1)
    
    parser.add_argument('-sim', '--simulation', metavar='', type=str, help='Determine Sim Type: (M)onteCarlo or (S)ensitivityAnalysis. Default M, Monte Carlo Analysis.', default='M')
    parser.add_argument('-n', '--numberOfRuns', metavar='', type=int, help='Maximum number of runs. Default 95% Confidence Interval', default=-1)
    
    parser.add_argument('-cam', '--camera', metavar='', type=str, help='Star Tracker Hardware: (I)deal, (B)asic, (P)oor, or path to JSON. Default Basic', default='B')
    parser.add_argument('-sw', '--software', metavar='', type=str, help='Centroid Software: (I)deal, (B)asic, (A)dvanced, or path to JSON. Default Basic', default='B')

    parser.add_argument('-b', '--batchRuns', metavar='', type=int, help='Number of runs per batch. Default {}.'.format(DEFAULT_RUNS), default=DEFAULT_RUNS)

    parser.add_argument('-par', '--parameters', metavar='', type=str, nargs='*',
                        help='Set parameters to analyze for Sensitivity Plot:\n \
                            (F)ocal Length; \
                            Focal (A)rray Inclination; \
                            (C)entroid Accuracy; \
                            (D)istortion; \
                            (P)rincipal Point Deviation; \
                            (T)emperature. Default Focal Length.', default='F')

    parser.add_argument('-p', '--plot', help='Plot Parameter/Input Data', action='store_true')
    parser.add_argument('-name', help='Name file', default=None)

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
            return StarTracker(cam_json=c.IDEAL_CAM, cam_name="CENTROID")
        
        case 'B' | 'BASIC':
            return StarTracker(cam_json=c.BASIC_CAM, cam_name='Basic Camera')

        case 'P' | 'POOR':
            return StarTracker(cam_json=c.POOR_CAM, cam_name='Poor Camera')
        
        case 'S' | 'SUN':
            return StarTracker(cam_json=c.SUN_CAM, cam_name='Sun Camera')
        
        # case 'B' | 'BAD':
        #     return StarTracker(cam_json=c.BAD_CAM, cam_name='Bad Camera')

        case _:
            # print(args_cam_upper)
            st = StarTracker(cam_json=args_cam)
            # print(st.cam_name)
            return st

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

def setup_sim_class(sim_type:str, run_count:int, * , cam:StarTracker, software:Software, orbit:Orbit, mod_params:list[str])->Simulation:
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
            return  mc.MonteCarlo(cam, software, orbit, run_count)

        case 'S' | 'Sensitivity Analysis':
            # return sense.(mod_params, cam, software, orbit, DEFAULT_RUNS)
            return sense.Sense(cam, software, orbit, run_count)

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
            
            case _:
                named_params.append(param)

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

        case 'P':
            obj_func = simulation.phi_model
            
        case 'M':
            obj_func = simulation.star_mag
        # case 'C':
        #     obj_func = simulation.sun_etal_quest

        case _:
            raise NameError('Improper Input. Either Q/P/M for QUEST, Phi Model, or Magnitude, respectively')

    return obj_func

if __name__ == '__main__':
    args = parse_arguments()    # parse command line arguments

    """ 
    SET SIMULATION PARAMETERS
    """
    camera = setup_star_tracker(args.camera)
    sw = setup_software(args.software)
    params = setup_params(args.parameters)
    sim = setup_sim_class(args.simulation, args.batchRuns, cam=camera, software=sw, orbit=Orbit(), mod_params=params)
    
    obj_func = setup_objective(sim, args.function)

    # logger.info('Camera:\n{}'.format(sim.camera))
    # logger.info('Orbit:\n{}'.format(sim.orbit))
    # logger.info('Software:\n{}'.format(sim.software))
    # logger.info('Runs: {}'.format(sim.num_runs))

    """ 
    RUN SIMULATION
    """
    fin_df = pd.DataFrame()
    # acc_col:str=None
    count = 0
    start = time.perf_counter()
    
    sim.run_sim(params=params, obj_func=obj_func, maxRuns=args.numberOfRuns)
    fin_df = sim.sim_data
    count = sim.count
    acc_col = sim.output_name


    # std_ratio = 1
    # prev_std = 1
    # half_normal_std = lambda x: x / np.sqrt(1 - 2/np.pi)
    # half_normal_mean = lambda x: half_normal_std(x) * np.sqrt(2/np.pi)

    # while std_ratio > 1e-4 or len(fin_df.index) < 2_000:
    #     st = time.perf_counter()
    #     count += 1
    #     df = sim.run_sim(params=params, obj_func=obj_func).copy()
    #     if count == 1:
    #         acc_col = sim.output_name
    #         fin_df = df
        
    #     else:
    #         fin_df = pd.concat([fin_df, df], axis=0)
        
    #     std_ratio = np.abs((prev_std - half_normal_std(fin_df[acc_col]).std())/prev_std)
    #     prev_std = half_normal_std(fin_df[acc_col].std())
        
    #     logger.info(f'Run:\n'
    #                 f'\tCount: {count}\n'
    #                 f'\tTime: {time.perf_counter() - st}\n'
    #                 f'\t\n\tSize: {len(fin_df.index)}\n'
    #                 # f'\tMean: {half_normal_mean(fin_df[acc_col].std())}\n'
    #                 f'\tFull STD: {prev_std}\n'
    #                 f'\tRatio: {std_ratio}')
        
    #     if args.numberOfRuns > 0 and len(fin_df.index) >= args.numberOfRuns:
    #         logger.critical(f'{c.RED}MAXIMUM RUNS REACHED{c.DEFAULT}')
    #         break
        
    delta = time.perf_counter() - start
    
    numFail = DEFAULT_RUNS*count - len(fin_df.index)
    failRate = numFail/(DEFAULT_RUNS * count)
    mean = sim.half_normal_mean(fin_df[acc_col].std())
    true_std = sim.half_normal_std(fin_df[acc_col].std())

    logger.critical('{}TOTAL TIME: {} s{}'.format(c.GREEN, delta, c.DEFAULT))
    logger.critical('{}PER RUN TIME: {} ms{}'.format(c.GREEN, delta/(count*DEFAULT_RUNS) , c.DEFAULT))
    # logger.critical('{}MEAN ACC: {}\"{}'.format(c.GREEN, mean, c.DEFAULT))
    logger.critical('{}STD ACC: {}\"{}'.format(c.GREEN, true_std, c.DEFAULT))
    logger.critical('{}FAILURE RATE: {}% ({}/{}){}\n'.format(c.GREEN, failRate*100, numFail, DEFAULT_RUNS*count, c.DEFAULT))
    # logger.critical('\n{}MEAN DATA:\n{}{}'.format(c.GREEN, fin_df.mean(), c.DEFAULT))
    
    logger.debug('{}SIM COLS:\n\n{}{}'.format(c.RED, sim.sim_data.columns, c.DEFAULT))
    logger.debug('{}SIM DATA:\n\n{}{}'.format(c.RED, sim.sim_data.to_string(), c.DEFAULT))
    
    # save data
    if args.name is not None:
        fp = os.path.join(c.curFile, 'data/')
        pd.to_pickle(fin_df, fp+args.name+'.pkl')
    # plt.show()
    """ 
    PLOT SIMULATION RESULTS
    """
    if args.plot:
        # df.CALC_ACCURACY.hist(bins=100)
        sim.plot_data(fin_df, true_std)
        # sim.plot
        
        plt.show()
