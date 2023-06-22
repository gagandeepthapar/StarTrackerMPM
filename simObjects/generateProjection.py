# pylint: disable-all

import sys
sys.path.append(sys.path[0] + '/..')

# import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import constants as c
import logging

logger = logging.getLogger(__name__)
plt.rcParams['text.usetex'] = True
def plot_map(starlist:pd.DataFrame, ra:float, dec:float, fov:float):

    fig = plt.figure()
    ax = fig.add_subplot()

    ax.scatter(starlist.right_ascension, starlist.declination, s=1, color='black')
    ax.scatter(ra, dec, color='purple', marker='x', s=10)

    t = np.linspace(0, 2*np.pi, 1000)
    x = np.cos(t)*fov
    y = np.sin(t)*fov

    ax.plot(x+ra, y+dec, color='purple')
    ax.set_xlabel('Right Ascension [rad]', fontsize=12)
    ax.set_ylabel('Declination [rad]', fontsize=12)

    # ax.scatter(ra + fov/2, dec, color='red', marker='x', s=5)
    # ax.scatter(ra - fov/2, dec, color='red', marker='x', s=5)
    # ax.scatter(ra, dec + fov/2, color='red', marker='x', s=5)
    # ax.scatter(ra, dec - fov/2, color='red', marker='x', s=5)
    ax.axis('equal')
    plt.show()
    return

def ra_dec_to_eci(row:pd.Series)->np.ndarray:
    ra = row.right_ascension
    dec = row.declination
    return np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

def calc_fov(eci_vec:np.ndarray, boresight:np.ndarray)->float:
    dot_prod = boresight.dot(eci_vec)
    return np.arccos(dot_prod)

def eci_to_cv_RM(roll:float, boresight:np.ndarray)->np.ndarray:

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

def eci_to_cv_rotation(ra:float, dec:float, roll:float)->np.ndarray:

    C_ra = c.Rz(-ra)
    C_dec = c.Ry(dec - np.pi/2)
    C_roll = c.Rz(roll)

    return C_roll @ C_dec @ C_ra

def generate_projection(starlist:pd.DataFrame, ra:float=0, dec:float=0, roll:float=0, camera_fov:float=np.deg2rad(30), max_magnitude:float=5)->pd.DataFrame:

    # copy list and update parameters
    fstars = starlist.copy()
    fstars = fstars.drop(columns=['spectral_type_a', 'spectral_type_b', 'ascension_proper_motion', 'declination_proper_motion'])
    
    # logger.critical(f'{c.GREEN}NUM STARS: {len(fstars.index)}{c.DEFAULT}')


    # remove dim stars
    # if max_magnitude < 10:
    #     fstars = fstars[fstars['v_magnitude'] <= max_magnitude]

    # calc boresight
    boresight = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

    fstars = fstars[(fstars['declination'] + np.pi/2 <= np.mod(dec + camera_fov/2 + np.pi/2 , np.pi)) & (fstars['declination'] + np.pi/2 >= np.mod(dec - camera_fov/2 + np.pi/2 , np.pi))]
    fstars = fstars[(fstars['right_ascension'] <= np.mod(ra + camera_fov/2, 2*np.pi)) & (fstars['right_ascension'] >= np.mod(ra - camera_fov/2, 2*np.pi))]

    if len(fstars.index) <= 1:
        # Failure to capture stars
        logger.critical('{}STAR FAILURE!{}'.format(c.RED, c.DEFAULT))
        return pd.DataFrame({'ECI_TRUE':[],
                             'CV_TRUE':[]})

    # set ECI vectors
    fstars['ECI_TRUE'] = fstars[['right_ascension', 'declination']].apply(ra_dec_to_eci, axis=1)

    # set FOV and remove out of FOV stars 
    fstars['FOV'] = fstars['ECI_TRUE'].apply(calc_fov, args=(boresight, ))
    fstars = fstars[fstars['FOV'] <= camera_fov/2]
    
    # set CV vectors
    C_eci_cv = eci_to_cv_rotation(ra, dec, roll)
    # C_new = new_eci_cv(ra, dec, roll)

    fstars['CV_TRUE'] = fstars['ECI_TRUE'].apply(lambda x: C_eci_cv @ x)
    # fstars['CV_NEW'] = fstars['ECI_TRUE'].apply(lambda x: C_new @ x)
    # fstars['DIFF'] = fstars.CV_TRUE - fstars.CV_NEW

    return fstars

def plot_frame(frame:pd.DataFrame, boresight:np.ndarray, ra, dec):
    
    def get_cone(start:np.ndarray, end:np.ndarray, rad:float, size:int, zflag=None)->np.array:

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


        if zflag is None:
            R = rotm(np.array([0, 0, 1]), end-start)
        else:
            R = np.eye(3)

        print(R)

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

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    # boresight = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

    # Sphere
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    ax.plot_surface(x, y, z, cmap=plt.cm.YlGnBu_r, alpha=0.1)   # sphere

    # FOV Cone
    cone = get_cone([0,0,0], boresight, np.deg2rad(10), 20)
    # ax.plot_surface(cone[0], cone[1], cone[2],color='blue', alpha=0.3)

    relcone = get_cone(np.array([0, 0, 0]), np.array([0, 0, 1]), np.deg2rad(10), 20, 5)
    ax.plot_surface(relcone[0], relcone[1], relcone[2], color='red', alpha=0.3)
    # print(relcone)


    rabs = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), 0])
    rabs = rabs / np.linalg.norm(rabs)
    rax = rabs[0]
    ray = rabs[1]


    # ax.plot([0, boresight[0]], [0, boresight[1]], [0, boresight[2]], linestyle='dashed', color='purple')
    ax.plot([0, 1], [0, 0], [0, 0], linestyle='dashed', color='red')
    ax.plot([0, 0], [0, 1], [0, 0], linestyle='dashed', color='blue')
    ax.plot([0, 0], [0, 0], [0, 1], linestyle='dashed', color='green')
    # ax.plot([0, rax], [0, ray], [0, 0], linestyle='dotted', color='purple')
    ax.scatter(1,0,0,marker='>', color='red')
    ax.text(1.1, 0, -0.0, '$\overrightarrow{\mathbf{X}}$')
    ax.scatter(0,1,0,marker='>', color='blue')
    ax.text(0, 1.1, -0.0, '$\overrightarrow{\mathbf{Y}}$')
    ax.scatter(0,0,1,marker='>', color='green')
    ax.text(0, 0, 1.1, '$\overrightarrow{\mathbf{Z}}$')
    # ax.scatter(rax, ray, 0, marker='>', color='purple')

    arc_angles = np.linspace(0 * np.pi, ra, 20)
    arc_xs = 0.5 * np.cos(arc_angles) * rax
    arc_ys = 0.5 * np.sin(arc_angles) * ray
    arc_zs = np.zeros(arc_ys.shape)
    # ax.plot(arc_xs, arc_ys, arc_zs, color = 'purple', lw = 2)
    # ax.text(0.6*np.cos(ra/2), 0.6*np.sin(ra/2), 0, '$\mathbf{\\alpha}$', size=15)

    r = 0.5  # radius
    theta_values = np.linspace(0, dec, 100)  # angles for parameterizing the circle

    # Original circle (around z-axis)
    x = r * np.cos(theta_values)
    z = r * np.sin(theta_values)
    y = np.zeros_like(x)  # circle is in xy-plane

    # Rotation
    angle = ra  # convert angle to radians
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle),  np.cos(angle)]])

    # Apply rotation to each point on the circle
    for i in range(len(x)):
        x[i], y[i] = np.dot(rotation_matrix, [x[i], y[i]])
    
    # ax.plot(x, y, z, color = 'purple', lw = 2)
    

    for i, row in frame.iterrows():
        eci = row.ECI_TRUE
        cv = row.CV_TRUE
        # ax.scatter(*eci, marker='*', color='black')
        ax.scatter(*cv, marker='*', color='black')

    # ax.scatter(*boresight, marker='x', s=50, color='purple')    
    # ax.text(0.6*np.cos(ra)*np.cos(dec/3), 0.6*np.sin(ra)*np.cos(dec/3), 0.6*np.sin(dec/3), '$\delta$', size=15)
    ax.axis('equal')
    ax.axis('off')
    ax.grid(False)


    return

if __name__ == '__main__':

    ra = np.random.uniform(np.pi/6,np.pi/3)
    dec = np.random.uniform(np.pi/6, np.pi/3)
    roll = np.random.uniform(-np.pi, np.pi)
    
    ra = 0.8071793307712165
    dec = 0.5853173095736518
    roll = 0.6432197502210117

    # ra = 0
    # dec = 0
    roll = 0 

    print(ra, dec, roll)


    # ra = np.deg2rad(10)
    # dec = np.deg2rad(12)
    # roll = 0

    starframe:pd.DataFrame = pd.read_pickle(c.BSC5PKL)

    fov = np.deg2rad(20)

    boresight = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])

    fs = generate_projection(starframe,ra, dec, roll, fov)
    fs = fs[fs['v_magnitude'] <= 5.2]
    print(fs)
    plot_frame(fs, boresight, ra, dec)

    print(eci_to_cv_rotation(ra, dec, roll)@np.array([1, 0, 0]))

    # plot_map(starframe[starframe['v_magnitude'] <= 5.2], ra, dec, fov)

    plt.show()
