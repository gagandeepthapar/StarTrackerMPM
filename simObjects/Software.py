import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import pandas as pd

from dataclasses import dataclass

import matplotlib.pyplot as plt
from alive_progress import alive_bar
from Parameter import Parameter, UniformParameter

import constants as c
from json import load as jsonload

@dataclass
class Projection:
    def __init__(self, Centroid_Deviation:Parameter,* ,cameraFP:str=c.ALVIUM_CAM,numStars:int=7)->None:

        self.dev = Centroid_Deviation

        self.numStars = numStars
        self.imWidth, self.imHeight, self.focal = self.__read_camera(cameraFP)

        self.image_x = UniformParameter(0, self.imWidth, 'IMAGE_X', units='px')
        self.image_y = UniformParameter(0, self.imHeight, 'IMAGE_Y', units='px')

        self.alpha = UniformParameter(0, 360, 'ALPHA_ANG', units=c.DEG)
        self.beta = UniformParameter(0, 360, 'BETA_ANG', units=c.DEG)
        self.gamma = UniformParameter(0, 360, 'GAMMA_ANG', units=c.DEG)

        self.randomize()
        return

    def __repr__(self)->str:
        name = 'PROJECTION'
        return name
    
    def randomize(self, num:int=None)->None:
        if num is None:
            num = self.numStars
        
        self.C = self.__calc_rot_matr()

        frame = self.__generate_px_position(num)
        frame['DEV_X'] = [self.dev.modulate() for _ in range(num)]
        frame['DEV_Y'] = [self.dev.modulate() for _ in range(num)]
        frame['CV_REAL'] = frame.apply(self.__px_to_cv, axis=1, args=(False,))
        frame['CV'] = frame.apply(self.__px_to_cv, axis=1, args=(True,))
        frame['ECI'] = frame['CV_REAL'].apply(lambda x: (self.C@x))

        self.frame = frame
        return 

    def __generate_px_position(self, num:int)->pd.DataFrame:
        imx = np.empty(num, dtype=float)
        imy = np.empty(num, dtype=float)

        for i in range(num):
            imx[i] = self.image_x.modulate()
            imy[i] = self.image_y.modulate()
        
        df = pd.DataFrame({'IMAGE_X':imx, 'IMAGE_Y':imy})
        return df

    def __read_camera(self, fp:str)->tuple[float]:
        with open(fp) as fp_open:
            file = jsonload(fp_open)
        
        imX = file['IMAGE_WIDTH']
        imY = file['IMAGE_HEIGHT']
        f = file['FOCAL_LENGTH_IDEAL']/file['PIXEL_HEIGHT']
        return imX, imY, f
    
    def __px_to_cv(self, row:pd.Series, devFlag:bool=False)->np.ndarray:
        x = row['IMAGE_X']
        y = row['IMAGE_Y']
        
        if devFlag:
            x += row['DEV_X']
            y += row['DEV_Y']

        cvx = x - self.imWidth/2
        cvy = y - self.imHeight/2
        v = np.array([cvx, cvy, self.focal])
        return v/np.linalg.norm(v)

    def __calc_rot_matr(self)->np.ndarray:
        alpha = np.deg2rad(self.alpha.modulate())
        beta = np.deg2rad(self.beta.modulate())
        gamma = np.deg2rad(self.gamma.modulate())

        Rz = lambda theta: np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0],[0,0,1]])
        Ry = lambda theta: np.array([[np.cos(theta), 0, np.sin(theta)],[0,1,0],[-np.sin(theta), 0, np.cos(theta)]])
        Rx = lambda theta: np.array([[1,0,0],[0, np.cos(theta), -np.sin(theta)],[0, np.sin(theta), np.cos(theta)]])
        R = Rz(gamma) @ Ry(beta) @ Rx(alpha)

        return R

class QUEST(Projection):

    def __init__(self)->None:
        X = Parameter(0, 0.1/3, 0, name='Centroid_DEV', units='px')
        super().__init__(X)
        return
    
    def __repr__(self)->str:
        name = 'QUEST SOLVER:\n{}\n'.format(self.frame)
        return name

    def get_attitude(self)->np.ndarray:
        B = self.__calc_B()
        k_22 = np.trace(B)
        K_12 = self.__calc_K12(B)
        S = B + np.transpose(B)

        # get optimal lam
        a = k_22**2 - np.trace(self.__calc_adjoint(S))
        b = k_22**2 + K_12.T @ K_12
        c = np.linalg.det(S) + K_12.T @ S @ K_12
        d = K_12.T @ S @ S @ K_12

        lam = self.__calc_optimal_lambda(a, b, c, d, k_22)

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

    def quat_to_ra_dec(self, q:np.ndarray)->np.ndarray:
        q1 = q[0]
        q2 = q[1]
        q3 = q[2]
        q4 = q[3]

        ra = np.arctan2(q2*q3 - q1*q4, q1*q3 + q2*q4)
        dec = np.arcsin(-q1**2*np.sqrt(q2**2 + q3**2 + q4**2))
        roll = np.arctan2(q2*q3 + q1*q4, -q1*q3 + q2*q4)

        angs = [ra, dec, roll]
        angDecs = [np.rad2deg(ang) for ang in angs]

        return angDecs

    def randomize(self, num:int=7)->None:
        super().randomize(num)
        self.realQuat = self.rot_to_quat(self.C.T)
        return

    def calc_diff(self, Q_real:np.ndarray, Q_calc:np.ndarray)->float:

        realRA = self.quat_to_ra_dec(Q_real)
        calcRA = self.quat_to_ra_dec(Q_calc)
        diff = [(a-b)*3600 for (a,b) in zip(realRA, calcRA)]

        return np.linalg.norm(diff)

    def __calc_B(self, weights:np.ndarray=None)->np.ndarray:
        
        eci = self.frame['ECI'].to_numpy()
        cv = self.frame['CV'].to_numpy()

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

        # print(f'iters: {i}')
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

    Q = QUEST()
    
    diff = np.empty(1_000, dtype=float)

    with alive_bar(1_000) as bar:
        for i in range(1_000):
            Q.randomize(7)
            q = Q.get_attitude()
            diff[i] = Q.calc_diff(Q.realQuat, q)
            bar()
    
    print(np.mean(diff))
    print(np.std(diff))

    fig = plt.figure()
    ax= fig.add_subplot()
    ax.hist(diff, bins=100)

    plt.show()
