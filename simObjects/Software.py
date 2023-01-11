import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import pandas as pd

from dataclasses import dataclass

import matplotlib.pyplot as plt
from alive_progress import alive_bar
from Parameter import Parameter, UniformParameter

import constants as c
from simTools import generateProjection as gp

@dataclass
class Projection:
    def __init__(self)->None:
        self.ra = UniformParameter(-180, 180, name="RIGHT_ASCENSION", units=c.DEG)
        self.dec = UniformParameter(-180, 180, name="DECLINATION", units=c.DEG)
        self.roll = UniformParameter(0, 0, name="ROLL", units=c.DEG)

        self.catalog = c.BSC5PKL
        self.cam = c.ALVIUM_CAM

        self.frame:pd.DataFrame=None

        self.randomize()
        return
    
    def __repr__(self)->str:
        name = "PROJECTION STRUCT"
        return name

    def randomize(self, mag:float=9,plot:bool=False)->None:
        self.ra.modulate()
        self.dec.modulate()
        self.roll.modulate()
        frame = gp.generate_projection(ra=self.ra.value,
                                        dec=self.dec.value,
                                        roll=self.roll.value,
                                        camera_mag=mag,
                                        cfg_fp=self.cam,
                                        catpkl_fp=self.catalog,
                                        plot=plot)

        self.frame = frame[['catalog_number',
                            'right_ascension',
                            'declination',
                            'ECI_X',
                            'ECI_Y',
                            'ECI_Z',
                            'CV_X',
                            'CV_Y',
                            'CV_Z']]
        return

class QUEST(Projection):

    def __init__(self)->None:
        super().__init__()
        self.roll = UniformParameter(0, 0, "ROLL", c.DEG)
        return
    
    def __repr__(self)->str:
        name = 'QUEST SOLVER:\n{}\n'.format(self.frame)
        return name

    def get_attitude(self)->np.ndarray:
        #SBK -> CV
        #SAK -> ECI

        B = self.__calc_B()
        k_22 = np.trace(B)
        K_12 = self.__calc_K12(B)

        S = B + np.transpose(B)
        K_11 = S - k_22*np.identity(3)

        # get optimal lam
        a = k_22**2 - np.trace(self.__calc_adjoint(S))
        b = k_22**2 + K_12.T @ K_12
        c = np.linalg.det(S) + K_12.T @ S @ K_12
        d = K_12.T @ S @ S @ K_12

        print('start')
        lam = self.__calc_optimal_lambda(a, b, c, d, k_22)
        print('done')
        print(lam)

        alpha = lam**2 - k_22**2 + np.trace(self.__calc_adjoint(S))
        beta = lam - k_22
        gamma = (lam + k_22)*alpha - np.linalg.det(S)

        X = (alpha*np.identity(3) + beta*S + (S@S))@K_12

        f = np.sqrt(gamma**2 + X.T @ X)
        factor = 1/f

        quatE = factor * X
        quatN = factor * gamma
        quat = np.array([*quatE, quatN])
        return quat/np.linalg.norm(quat) 

    def quat_to_eul(self, quat:np.ndarray)->np.ndarray:

        # q0 = quat[3]
        # q1 = quat[0]
        # q2 = quat[1]
        # q3 = quat[2]

        # phi = np.arctan2(2*(q0*q1 + q2*q3), 1-2*(q1**2 + q2**2))
        # theta = -np.pi/2 + 2*np.arctan2(np.sqrt(1 + 2*(q0*q2 - q1*q3)), np.sqrt(1-2*(q0*q2-q1*q3)))
        # psi = np.arctan2(2*(q0*q3 + q1*q2), 1-2*(q2**2 + q3**2))

        w = quat[3]
        x = quat[0]
        y = quat[1]
        z = quat[2]

        yaw = np.arcsin(2*x*y + 2*z*w)
        pitch = np.arctan2(2*x*w - 2*y*z, 1 - 2*x*x - 2*z*z)
        roll = -1*np.arctan2(2*y*w - 2*x*z, 1 - 2*y*y - 2*z*z)

        eul = [yaw, pitch, roll]
        eulDeg = [np.rad2deg(ang) for ang in eul]

        return np.array(eulDeg)

    def __calc_B(self, weights:np.ndarray=None)->np.ndarray:
        
        eci = self.frame[['ECI_X', 'ECI_Y', 'ECI_Z']].to_numpy()
        cv = self.frame[['CV_X', 'CV_Y', 'CV_Z']].to_numpy()
        # cv = eci

        if weights is None:
            weights = np.ones(len(eci))/len(eci)

        B = np.zeros((3,3))
        
        for i in range(len(eci)):
            B += weights[i] * np.outer(eci[i], cv[i])
            
        return np.transpose(B)

    def __calc_K12(self, B:np.ndarray)->np.ndarray:

        k_12 = np.array([B[1][2] - B[2][1],
                         B[2][0] - B[0][2],
                         B[0][1] - B[1][0]])

        return np.transpose(k_12)

    def __calc_adjoint(self, B:np.ndarray)->np.ndarray:
        deter = np.linalg.det(B)
        cofactor = np.linalg.inv(B).T * deter

        return np.transpose(cofactor)

    def __calc_optimal_lambda(self, a:float, b:float, c:float, d:float, k:float, lam0:float=1.1, eps:float=1e-12):

        e0 = lam0
        e1 = lam0 - self.__optimal_lambda_f(e0, a, b, c, d, k)/self.__optimal_lambda_fp(e0, a, b, c)
        err = np.abs(e1 - e0)

        i = 0
        while err > eps:
            i+=1
            e0 = e1
            e1 = e0 - self.__optimal_lambda_f(e0, a, b, c, d, k)/self.__optimal_lambda_fp(e0, a, b, c)
            err = np.abs(e1 - e0)

        print(f'iters: {i}')
        return e1
    
    def __optimal_lambda_f(self, lam:float, a:float, b:float, c:float, d:float, k:float)->float:
        A = lam**4
        B = -1 * (a+b)*lam**2
        C = -c*lam
        D = (a*b + c*k - d)
        return A + B + C +D
    
    def __optimal_lambda_fp(self, lam:float, a:float, b:float, c:float)->float:
        A = 4*lam**3
        B = -2*(a + b)*lam
        C = -c
        return A + B + C

if __name__ == '__main__':
    # roll = UniformParameter(0, 0, 'ROLL', units=c.DEG)
    Q = QUEST()
    # print(Q.frame)

    ra = np.deg2rad(10)
    dec = np.deg2rad(20)

    print(ra)
    print(dec)

    bs_x = np.cos(ra)*np.cos(dec)
    bs_y = np.sin(ra)*np.cos(dec)
    bs_z = np.sin(dec)

    f = {'CV_X': [1], 'CV_Y':[0], 'CV_Z':[0],
         'ECI_X': [bs_x], 'ECI_Y':[bs_y], 'ECI_Z': [bs_z]}

    Q.frame = pd.DataFrame(f)

    print(Q.frame)
    q = Q.get_attitude()
    print(q)
    e = Q.quat_to_eul(q)
    print(e)

    pd.set_option("display.precision", 12)
    print(Q.frame['ECI_X'])
    # Q.roll = roll

    # Q.randomize(mag=5.5)
    # q = Q.get_attitude()
    # print(q)
    # e = Q.quat_to_eul(q)
    # print(e)