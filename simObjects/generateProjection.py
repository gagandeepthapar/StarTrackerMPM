import sys
sys.path.append(sys.path[0] + '/..')

# import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# from AttitudeEstimation import QUEST

import constants as c
import logging
import time

logger = logging.getLogger(__name__)

def plot_map(starlist:pd.DataFrame, ra:float, dec:float, fov:float):

    fig = plt.figure()
    ax = fig.add_subplot()

    # starlist = starlist[starlist['v_magnitude'] >= 5]

    ax.scatter(starlist.right_ascension, starlist.declination, s=1, color='black')
    ax.scatter(ra, dec, color='red', marker='x', s=5)

    ax.scatter(ra + fov/2, dec, color='red', marker='x', s=5)
    ax.scatter(ra - fov/2, dec, color='red', marker='x', s=5)
    ax.scatter(ra, dec + fov/2, color='red', marker='x', s=5)
    ax.scatter(ra, dec - fov/2, color='red', marker='x', s=5)

    plt.show()
    return

def ra_dec_to_eci(row:pd.Series)->np.ndarray:
    ra = row.right_ascension
    dec = row.declination
    return np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

def calc_fov(eci_vec:np.ndarray, boresight:np.ndarray)->float:
    dot_prod = boresight.dot(eci_vec)
    return np.arccos(dot_prod)

def eci_to_cv_rotation(roll:float, boresight:np.ndarray)->np.ndarray:

    perp_vec = ra_dec_to_eci(pd.Series({'right_ascension': 0, 'declination': -np.pi/2}))
    normal_vec = np.cross(boresight, perp_vec)

    k_hat = boresight / np.linalg.norm(boresight)
    j_hat = normal_vec / np.linalg.norm(normal_vec)
    i_hat = np.cross(j_hat, k_hat)

    C_eci_cv = np.array([i_hat,
                         j_hat,
                         k_hat])
    C_post_roll = c.Rz(roll) @ C_eci_cv
    return C_post_roll

def new_eci_cv(ra:float, dec:float, roll:float)->np.ndarray:

    C_ra = c.Rz(-ra)
    C_dec = c.Ry(dec - np.pi/2)
    C_roll = c.Rz(roll)

    return C_roll @ C_dec @ C_ra

def generate_projection(starlist:pd.DataFrame, ra:float=0, dec:float=0, roll:float=0, camera_fov:float=np.deg2rad(30), max_magnitude:float=5)->pd.DataFrame:

    # copy list and update parameters
    fstars = starlist.copy()
    fstars = fstars.drop(columns=['spectral_type_a', 'spectral_type_b', 'ascension_proper_motion', 'declination_proper_motion'])
    
    # remove dim stars
    fstars = fstars[fstars['v_magnitude'] <= max_magnitude]

    # calc boresight
    boresight = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

    fstars = fstars[(fstars['declination'] + np.pi/2 <= np.mod(dec + camera_fov/2 + np.pi/2 , np.pi)) & (fstars['declination'] + np.pi/2 >= np.mod(dec - camera_fov/2 + np.pi/2 , np.pi))]
    fstars = fstars[(fstars['right_ascension'] <= np.mod(ra + camera_fov/2, 2*np.pi)) & (fstars['right_ascension'] >= np.mod(ra - camera_fov/2, 2*np.pi))]

    if len(fstars.index) <= 1:
        # Failure to capture stars
        logger.critical('{}FAILURE!{}'.format(c.RED, c.DEFAULT))
        return pd.DataFrame({'ECI_TRUE':[],
                             'CV_TRUE':[]})

    # set ECI vectors
    fstars['ECI_TRUE'] = fstars[['right_ascension', 'declination']].apply(ra_dec_to_eci, axis=1)

    # set FOV and remove out of FOV stars 
    fstars['FOV'] = fstars['ECI_TRUE'].apply(calc_fov, args=(boresight, ))
    fstars = fstars[fstars['FOV'] <= camera_fov/2]
    
    # set CV vectors
    C_eci_cv = eci_to_cv_rotation(roll, boresight)
    C_new = new_eci_cv(ra, dec, roll)
    fstars['CV_TRUE'] = fstars['ECI_TRUE'].apply(lambda x: C_eci_cv @ x)
    fstars['CV_NEW'] = fstars['ECI_TRUE'].apply(lambda x: C_new @ x)

    return fstars

def plot_frame(frame:pd.DataFrame, boresight:np.ndarray):

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # boresight = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

    # Sphere
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.3)   # sphere

    ax.scatter(0, 0, 1, marker='x', s=30, color='black')
    ax.scatter(*boresight, marker='x', s=30, color='black')

    ax.plot([0, 0], [0, 0], [0, 1], linestyle='dashed', color='red')
    ax.plot([0, boresight[0]], [0, boresight[1]], [0, boresight[2]], linestyle='dashed', color='blue')

    for i, row in frame.iterrows():
        cv = row.CV_TRUE
        eci = row.ECI_TRUE
        cvnew = row.CV_NEW

        ax.scatter(*cv, marker='x', color='red')
        ax.scatter(*cvnew, marker='^', color='green')
        ax.scatter(*eci, marker='*', color='blue')


    ax.axis('equal')
    
    return

if __name__ == '__main__':
    N = 1000

    # boresight
    ra = np.random.uniform(0, 2*np.pi)
    dec = np.random.uniform(-np.pi/2, np.pi/2)
    roll = np.random.uniform(-np.pi, np.pi)
    
    # ra = 0
    # dec = 0
    # roll = 0

    

    # start projection process
    starframe:pd.DataFrame = pd.read_pickle(c.BSC5PKL)

    # print(starframe.columns)
    # print(starframe.v_magnitude.min())
    # print(starframe.v_magnitude.max())

    # ra = 3.361102
    # dec = -0.158793
    # roll = 0.388512
    # mag = 4.233144
    fov = 2 * np.arctan2(1024/2, 3316)

    boresight = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

    fs = generate_projection(starframe,ra, dec, roll, fov)
    print(fs)
    plot_frame(fs, boresight)

    plt.show()

    # plot_frame(fs, boresight)
    # print('RA {}, DEC {}, ROLL {}\n{}'.format(np.rad2deg(ra), np.rad2deg(dec), np.rad2deg(roll), fs))
    # plt.show()
    # q4 = -(np.trace(C) + 1)**0.5 / 2
    # q1 = C[1][2] - C[2][1] / (4 * q4)
    # q2 = C[2][0] - C[0][2] / (4 * q4)
    # q3 = C[0][1] - C[1][0] / (4 * q4)

    # q = np.array([q1, q2, q3, q4])
    # q = q / np.linalg.norm(q)
    # print(q)
    # print(np.linalg.norm(q))

    # skew = lambda v: np.array([[0, -v[2], v[1]],
    #                            [v[2], 0, -v[0]],
    #                            [-v[1], v[0], 0]])

    # Cprime = (2 * q[3]**2 - 1) * np.eye(3) + (2 * np.outer(q[:3], q[:3])) - (2 * q[3] * skew(q[:3]))

    # print(Cprime)
    # print(Cprime @ Cprime.T)
    # print(Cprime.T @ Cprime)

    # print(fstars)
    # # print(np.cross(np.array([-1, 0, 0]), np.array([0, 0, -1])))
    # print('RA: {}\nDEC: {}\n:ROLL: {}'.format(np.rad2deg(ra), np.rad2deg(dec), np.rad2deg(roll)))
    # plt.show()
    # print(ra_dec_to_eci(pd.Series({'right_ascension': 0, 'declination': np.pi/2})))

    # apply_ra = c.Rz(-ra)
    # apply_dec = c.Ry(-(np.pi/2 - dec))
    
    
    
    # skew_sym = lambda v: np.array([[0, -v[2], v[1]],
    #                                [v[2], 0, -v[0]],
    #                                [-v[1], v[0], 0]])
    # # Rodrigues Rotation Formula
    # a_hat = skew_sym(boresight)
    # # print(a_hat)
    # apply_roll = np.eye(3) + np.sin(roll)*a_hat + (1 - np.cos(roll))*(a_hat**2)

    # apply_roll = apply_roll

    # C = apply_roll @ (apply_dec @ apply_ra)
    # print('Rod C: {}'.format(C))
    # print('Det: {}'.format(np.linalg.det(C)))
    # print('Tpos: {}; {}'.format(C @ C.T, C.T @ C))

