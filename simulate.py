import numpy as np
import matplotlib.pyplot as plt
import argparse
from alive_progress import alive_bar

from hardwareEffects import camera

def sunEtalHardwareFunc(camera:camera)->float:

    c = camera
    
    delF = c.f_len - c._true_f_len
    delS = c.principal_pt_acc - c._true_principal_pt_acc
    delX = c.centroid_acc - c._true_centroid_acc
    delD = c.distortion - c._true_distortion
    theta = np.deg2rad(c.f_array_inc_angle - c._true_f_array_inc_angle)
    beta = np.deg2rad(c._true_incident_angle)

    delF = 0
    delX = 0
    delD = 0
    theta = 0

    delS = np.random.normal(scale=1.5, loc=0)
    beta = np.deg2rad(8.5)

    # print(np.rad2deg(beta))

    chars = 'delF: {}, delS: {}, delX: {}, delD: {}, theta: {}, beta: {}'.format(delF, delS, delX, delD, theta, beta)
    trues = 'focal: {}, centroid: {}, princ: {}, distortion: {}'.format(c.f_len, c.centroid_acc, c.principal_pt_acc, c.distortion)
    # print(chars)
    # print(f'true: {c._true_principal_pt_acc}; mod: {c.principal_pt_acc}')
    # print(trues)

    f1 = (c._true_f_len + delF + delS*np.tan(theta))/np.cos(theta + beta)
    f2 = delS/np.cos(theta)
    f3 = delS/(c._true_f_len*np.cos(theta))

    eA = np.arctan((f1*np.sin(beta) + f2 + delX + delD)/c._true_f_len) - np.arctan(f3) - beta

    eB = np.arctan(np.tan(beta)) + np.arctan(delS/(c._true_f_len*np.cos(theta))) - np.arctan(f3) - beta
    
    return eB * 3600
    # return delS

def simulateSunEtal(numRuns:int=1)->float:

    cam = camera()
    cam.reset_params()

    runs = np.array([])
    data = np.zeros((numRuns, 1))

    with alive_bar(numRuns) as bar:
        for i in range(numRuns):

            cam.modulate_params(principal=True)
            data[i] = sunEtalHardwareFunc(cam)

            bar()

    mean = np.mean(data)
    std_dev = np.std(data)
    
    print(f'{numRuns} Runs:\n\tMean: {mean} +/- {std_dev} asec')
    
    # fig = plt.figure()
    # ax = plt.axes()

    # ax.plot(runs, data)
    # plt.show()

    return mean, std_dev

def parse_arguments():

    parser = argparse.ArgumentParser(description='Arguments to run Monte Carlo Analysis')

    parser.add_argument('-n', help='Number of runs; Default 1', type=int, default=1)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    simulateSunEtal(args.n)
    print(np.deg2rad(8.5)*3600)
    # a = np.array([])