import sys
sys.path.append(sys.path[0] + '/..')

import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import constants as c
import logging


logger = logging.getLogger(__name__)

def ra_dec_to_eci(row:pd.Series)->np.ndarray:
    ra = row.right_ascension
    dec = row.declination
    return np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

def calc_fov(eci_vec:np.ndarray, boresight:np.ndarray)->float:
    dot_prod = boresight.dot(eci_vec)
    return np.arccos(dot_prod)

def eci_to_cv(eci_vec:np.ndarray, ra:float, dec:float, roll:float, boresight:np.ndarray)->np.ndarray:

    apply_ra = c.Rz(-ra)
    apply_dec = c.Ry(-(np.pi/2 - dec))
    
    unrolled_cv = apply_dec @ apply_ra @ eci_vec
    
    # Rodrigues Rotation Formula
    rolled_cv = (np.cos(roll) * unrolled_cv) + \
                np.sin(roll)*(np.cross(boresight, unrolled_cv)) + \
                (1 - np.cos(roll))*(boresight.dot(unrolled_cv))*boresight

    return  rolled_cv

def generate_projection(starlist:pd.DataFrame, ra:float=0, dec:float=0, roll:float=0, camera_fov:float=np.deg2rad(30), max_magnitude:float=5)->pd.DataFrame:

    # copy list and update parameters
    fstars = starlist.copy()
    fstars = fstars.drop(columns=['spectral_type_a', 'spectral_type_b', 'ascension_proper_motion', 'declination_proper_motion'])
    
    # remove dim stars
    # fstars = fstars[fstars['v_magnitude'] <= max_magnitude]

    # update declination range
    fstars['declination'] += np.pi/2

    # calc boresight
    boresight = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

    # narrow list to single box
    if (dec - camera_fov/2 < 0) or (dec + camera_fov/2 > np.pi):
        fstars = fstars[(fstars['declination'] <= np.mod(dec + camera_fov/2, np.pi)) | (fstars['declination'] >= np.mod(dec - camera_fov/2, np.pi))]
    else:
        fstars = fstars[(fstars['declination'] <= np.mod(dec + camera_fov/2, np.pi)) & (fstars['declination'] >= np.mod(dec - camera_fov/2, np.pi))]

    # print('{}\n{}'.format(len(fstars.index),fstars))


    if (ra - camera_fov/2 < 0) or (ra + camera_fov/2 > 2*np.pi):
        fstars = fstars[(fstars['right_ascension'] <= np.mod(ra + camera_fov/2, 2*np.pi)) | (fstars['right_ascension'] >= np.mod(ra - camera_fov/2, 2*np.pi))]
    else:
        fstars = fstars[(fstars['right_ascension'] <= np.mod(ra + camera_fov/2, 2*np.pi)) & (fstars['right_ascension'] >= np.mod(ra - camera_fov/2, 2*np.pi))]
    
    # set ECI vectors
    fstars['ECI_TRUE'] = fstars[['right_ascension', 'declination']].apply(ra_dec_to_eci, axis=1)

    # set FOV and remove out of FOV stars 
    fstars['FOV'] = fstars['ECI_TRUE'].apply(calc_fov, args=(boresight, ))
    fstars = fstars[fstars['FOV'] <= camera_fov/2]
    
    # set CV vectors
    fstars['CV_TRUE'] = fstars['ECI_TRUE'].apply(eci_to_cv, args=(ra, dec, roll, boresight, ))

    return fstars

def plot_sphere(starlist:pd.DataFrame, ra:float, dec:float, roll:float, fov:float, img_wd:int, img_ht:int):

    def get_cone(start:np.array, end:np.array, rad:float, size:int)->np.array:

        def rotm(vec1:np.array, vec2:np.array)->np.array:
                a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
                v = np.cross(a, b)
                c = np.dot(a, b)
                s = np.linalg.norm(v)
                kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
                return rotation_matrix

        h = np.linalg.norm(end-start)
        t = np.linspace(0, 2*np.pi, num=size)
        x = rad*np.cos(t)
        y = rad*np.sin(t)
        z = h*np.ones(size)

        R = rotm(np.array([0, 0, 1]), end-start)

        xr = np.zeros(size)
        yr = np.zeros(size)
        zr = np.zeros(size)

        for i in range(size):
            r = np.matmul(R, np.array([x[i], y[i], z[i]]))
            xr[i] = r[0]
            yr[i] = r[1]
            zr[i] = r[2]
        
        filler = np.zeros(size)
        X = np.array([filler, xr]).reshape((2,size))
        Y = np.array([filler, yr]).reshape((2, size))
        Z = np.array([filler, zr]).reshape((2, size))
        
        return np.array([X, Y, Z])

    radec2vec = lambda ra, dec: np.array([np.cos(ra)*np.cos(dec),
                                            np.sin(ra)*np.cos(dec),
                                            np.sin(dec)])

    bs = radec2vec(ra, dec)
    bs_x = bs[0]
    bs_y = bs[1]
    bs_z = bs[2]

    abs_cone = get_cone(np.array([0,0,0]), bs, np.tan(fov/2), 25)
    rel_cone = get_cone(np.array([0, 0, 0]), np.array([1,0,0]), np.tan(fov/2), 25)

    fig = plt.figure()

    # 3D PLOT
    ax = fig.add_subplot(2, 2, (1,3), projection='3d')

    r = 1
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.3)   # sphere

    # absolute/ECI representation
    ax.scatter3D(bs_x, bs_y, bs_z, color='red', marker='x') # absolute boresight
    ab = ax.plot3D(np.array([0, bs_x]), np.array([0, bs_y]), np.array([0, bs_z]), color='red') # absolute boresight
    ac = ax.plot_surface(abs_cone[0], abs_cone[1], abs_cone[2], color='red', alpha=0.5)  # absolute cone

    # relative representation
    ax.scatter3D(1, 0, 0, color='blue', marker='x')
    cb = ax.plot3D(np.array([0, 1]), np.array([0, 0]), np.array([0, 0]), color='blue')
    ax.plot_surface(rel_cone[0], rel_cone[1], rel_cone[2], color='blue', alpha=0.5) 

    title = 'Celestial Sphere\nPointing at ({:.2f}\u00b0, {:.2f}\u00b0, {:.2f}\u00b0), Mv = {:.1f}'.format(ra*180/PI, dec*180/PI, roll*180/PI, MAXMAG)

    ax.axis('equal')
    ax.set_title(title)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')

    for i, row in starlist.iterrows():
        ax.scatter3D(row['ECI_X'], row['ECI_Y'], row['ECI_Z'], color='red', marker='*') # absolute star
        ax.scatter3D(row['CV_X'], row['CV_Y'], row['CV_Z'], color='blue', marker='*') # absolute star

    # 2D Plot
    ax = fig.add_subplot(2,2,2)
    
    theta = np.linspace( 0 , 2 * np.pi , 150 )
    radius = fov/2
    x = radius * np.cos( theta )
    y = radius * np.sin( theta )

    c = img_wd/img_ht
    rectY = -1*np.sqrt(fov**2 / (4*(c**2 +1)))
    rectX = c*rectY
    rect = Rectangle((rectX,rectY), 2*np.abs(rectX), 2*np.abs(rectY), color='black', fill=False)

    ax.plot(x, y, '--b')
    ax.scatter(0, 0, color='blue', marker='x')
    ax.add_patch(rect)

    ct = 0
    for i, row in starlist.iterrows():
        ax.scatter(-1*row['CV_Y'],row['CV_Z'], color='red', marker='*', alpha=0.1)
        ax.scatter(-1*row['CV_X_ROLL'], row['CV_Y_ROLL'], color='black', marker='*',)
        
        label = 'ID {:d}'.format(int(row['catalog_number']))
        ax.annotate(label, (-1*row['CV_X_ROLL'], row['CV_Y_ROLL']+0.01), fontsize=7)

        ax.plot([-1*row['CV_Y'], -1*row['CV_X_ROLL']], [row['CV_Z'], row['CV_Y_ROLL']], alpha=0.1, color='black')

        if rectX <= row['CV_X_ROLL'] and row['CV_X_ROLL'] <= rectX+(2*np.abs(rectX)):
            if rectY <= row['CV_Y_ROLL'] and row['CV_Y_ROLL'] <= rectY+(2*np.abs(rectY)):
                ct += 1

    ax.axis('equal')
    ax.set_xlabel('Relative Right Ascension, deg')
    ax.set_ylabel('Relative Declination, deg')
    ax.legend()

    title = '{} of {} Stars Captured in Image'.format(ct, len(starlist.index))
    ax.set_title(title)

    # simulated image test
    ax = fig.add_subplot(2,2,4)

    rect = Rectangle((0,0), img_wd, img_ht, color='black')
   
    ax.add_patch(rect)
    ax.scatter(img_wd/2, img_ht/2, color='red', marker='x')

    check_x = lambda x: 0 <= x and x <= img_wd
    check_y = lambda y: 0 <= y and y <= img_ht

    for i, row in starlist.iterrows():
        x = row['IMG_X']
        y = row['IMG_Y']
        if check_x(x) and check_y(y):
            ax.scatter(x, y, color='white', marker='.')
            label = 'ID {:d}'.format(int(row['catalog_number']))

            ax.annotate(label, (x, y+40), fontsize=7, color='red') 

    ax.axis('equal')
    ax.legend()
    ax.set_xlabel('IMAGE X [Pixels]')
    ax.set_ylabel('IMAGE Y [Pixels]')
    delta = 10
    ax.set_xlim(0 - delta, img_wd + delta)
    ax.set_ylim(0 - delta, img_ht + delta)
    return plt

if __name__ == '__main__':
    # boresight
    # ra = random.uniform(0, 2*np.pi)
    # dec = random.uniform(0, np.pi)
    # roll = random.uniform(-np.pi, np.pi)

    ra = 4.954006
    dec = 1.960962
    roll = -0.475697

    # start projection process
    starframe = pd.read_pickle(c.BSC5PKL)
    fstars = generate_projection(starframe, ra, dec, roll)
    print(fstars)

