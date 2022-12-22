import numpy as np
import matplotlib.pyplot as plt

from hardwareEffects import camera

def sunEtalHardwareFunc(camera:camera)->float:

    c = camera
    
    delF = c.f_len - c._true_f_len
    delS = c.principal_pt_acc - c._true_principal_pt_acc
    delX = c.centroid_acc - c._true_centroid_acc
    delD = c.distortion - c._true_distortion
    theta = np.deg2rad(c.f_array_inc_angle - c._true_f_array_inc_angle)
    beta = np.deg2rad(c._true_incident_angle)

    f1 = (c.f_len + delF + delS*np.tan(theta))/np.cos(theta + beta)
    f2 = delS/np.cos(theta)
    f3 = delS/(c.f_len*np.cos(theta))

    eA = np.arctan((f1*np.sin(beta) + f2 + delX + delD)/c.f_len) - np.arctan(f3) - beta

    return eA * 3600

def simulateSunEtal(numRuns:int=1)->float:

    cam = camera()
    cam.reset_params()

    runs = np.array([])
    data = np.array([])

    for i in range(numRuns):
        runs = np.append(runs, i)

        cam.modulate_params(principal=True)
        acc = sunEtalHardwareFunc(cam)
        
        data = np.append(data, acc)

    mean = np.mean(data)
    std_dev = np.std(data)
    
    print(f'{numRuns} Runs:\n\tMean: {mean} +/- {std_dev} asec')
    
    # fig = plt.figure()
    # ax = plt.axes()

    # ax.plot(runs, data)
    # plt.show()

    return mean, std_dev


if __name__ == '__main__':

    simulateSunEtal(100000)