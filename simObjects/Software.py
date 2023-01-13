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

    def rot_to_quat(self, C:np.ndarray)->np.ndarray:

        n = 1/2 * np.sqrt(1 + np.trace(C))
        e1 = 1/4 * (C[1][2] - C[2][1])/n
        e2 = 1/4 * (C[2][0] - C[0][2])/n
        e3 = 1/4 * (C[0][1] - C[1][0])/n

        q = np.array([e1, e2, e3, n])
        return q

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

    def __calc_optimal_lambda(self, a:float, b:float, c:float, d:float, k:float, lam0:float=1.0, eps:float=1e-12):

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

def rotEul(alpha, beta, gamma):
    alpha = np.deg2rad(alpha)
    beta = np.deg2rad(beta)
    gamma = np.deg2rad(gamma)

    Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0,0,1]])
    Ry = lambda theta: np.array([[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
    Rx = lambda theta: np.array([[1,0,0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
    R = Rz(gamma) @ Ry(beta) @ Rx(alpha)

    return R

if __name__ == '__main__':
    # roll = UniformParameter(0, 0, 'ROLL', units=c.DEG)
    Q = QUEST()

    numStars = 10
    a = 10
    b = 20
    c = 30
    
    print('{}x{}x{}'.format(a, b, c))

    C = rotEul(a, b, c)

    eci = np.zeros((numStars,3))
    cv = np.zeros((numStars, 3))
    for i in range(numStars):
        c = np.random.uniform(0, 1, 3)
        eci[i] = c/np.linalg.norm(c)
        cv[i] = C @ eci[i]

    print(eci)
    print(cv)
    # print(eci.T[0])

    f = {'ECI_X': eci.T[0],
         'ECI_Y': eci.T[1],
         'ECI_Z': eci.T[2],
         'CV_X': cv.T[0],
         'CV_Y': cv.T[1],
         'CV_Z': cv.T[2],}

    Q.frame = pd.DataFrame(f)
    print(Q.frame)
    q = Q.get_attitude()
    print('CHECK:')
    print(Q.rot_to_quat(C))
    print(q)
    # e=Q.quat_to_eul(q)
    # print(e)

    # print(C@C.T)
    # print(eci)
    # print(cv)


