import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
import pandas as pd

from Parameter import Parameter, UniformParameter

import constants as c
from json import load as jsonload


from alive_progress import alive_bar
import matplotlib.pyplot as plt


class Projection:
    def __init__(self, Centroid_Deviation:Parameter=None, * ,
                centroidFP:str=c.SIMPLE_CENTROID, cameraFP:str=c.ALVIUM_CAM, 
                numStars:Parameter=Parameter(7, 2, 0, name="NUM_STARS", units="", retVal=lambda x: np.max([1,int(np.round(x))])))->None:

        self.dev = self.__set_parameter(Centroid_Deviation, centroidFP)

        self.numStars = numStars
        self.imWidth, self.imHeight, self.focal = self.__read_camera(cameraFP)

        self.image_x = UniformParameter(0, self.imWidth, 'IMAGE_X', units='px')
        self.image_y = UniformParameter(0, self.imHeight, 'IMAGE_Y', units='px')

        self.randomize()
        return

    def __repr__(self)->str:
        name = 'PROJECTION @ {}'.format(self.quat_real)
        return name
    
    def randomize(self, num:int=None)->None:
        if num is None:
            num = self.numStars.modulate()
        
        self.quat_real:np.ndarray = self.__random_quat()

        frame:pd.DataFrame = self.__generate_px_position(num)
        frame['DEV_X'] = [self.dev.modulate() for _ in range(num)]
        frame['DEV_Y'] = [self.dev.modulate() for _ in range(num)]
        frame['CV_REAL'] = frame.apply(self.__px_to_cv, axis=1, args=(False,))
        frame['CV_EST'] = frame.apply(self.__px_to_cv, axis=1, args=(True,))
        frame['ECI_REAL'] = frame['CV_REAL'].apply(self.__quat_mult)

        self.frame = frame
        return 

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
    
    def calc_diff(self, Q_calc:np.ndarray)->float:

        realRA = self.quat_to_ra_dec(self.quat_real)
        calcRA = self.quat_to_ra_dec(Q_calc)
        diff = [(a-b)*3600 for (a,b) in zip(realRA, calcRA)]

        return np.linalg.norm(diff)

    def __set_parameter(self, param:Parameter, paramFP:str)->Parameter:
        if param is not None:
            return param
        
        with open(paramFP) as fp_open:
            file = jsonload(fp_open)
        ideal = file['IDEAL']
        mean = file['MEAN']
        std = file['STDDEV']
        units = file['UNITS']

        return Parameter(ideal, std, mean, name="CENTROID_ACCURACY", units=units)

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
    
    def __px_to_cv(self, row:pd.Series, devFlag:bool)->np.ndarray:
        x = row['IMAGE_X']
        y = row['IMAGE_Y']
        
        if devFlag:
            x += row['DEV_X']
            y += row['DEV_Y']

        cvx = x - self.imWidth/2
        cvy = y - self.imHeight/2
        v = np.array([cvx, cvy, self.focal])
        return v/np.linalg.norm(v)

    def __random_quat(self)->np.ndarray:
        q = np.random.uniform(0, 1, 4)
        return q/np.linalg.norm(q)

    def __quat_mult(self, x:np.ndarray)->np.ndarray:
        e = -1*self.quat_real[:3]
        n = self.quat_real[3]
        
        Ca = (2*n**2 - 1) * np.identity(3)
        Cb = 2*np.outer(e, e)
        Cc = -2*n*self.__skew(e)

        return (Ca + Cb + Cc)@x
    
    def __skew(self, n:np.ndarray)->np.ndarray:
        return np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

class QUEST(Projection):

    def __init__(self)->None:
        super().__init__()
        return
    
    def __repr__(self)->str:
        name = 'QUEST SOLVER, {}:\n{}\n'.format(self.quat_real,self.frame)
        return name

    def get_attitude(self, *, weights:np.ndarray=None)->np.ndarray:
        # extract eci and cv vectors
        eci_vecs = self.frame['ECI_REAL'].to_numpy()
        cv_vecs = self.frame['CV_EST'].to_numpy()
        if weights is None:
            weights = np.ones(len(eci_vecs))/len(eci_vecs)

        # calc lam params
        B = self.__calc_B(eci_vecs, cv_vecs, weights)
        K_12, K_22 = self.__calc_K(B)
        S = B + np.transpose(B)
        a, b, c, d = self.__calc_lam_params(K_12, K_22, S)
        
        # get optimal lam
        lam = self.__calc_optimal_lambda(a, b, c, d, K_22)

        # calc quat params
        alpha, beta, gamma = self.__calc_quat_params(lam, K_22, S)

        X = (alpha*np.identity(3) + beta*S + (S@S))@K_12
        f = np.sqrt(gamma**2 + X.T @ X)
        factor = 1/f

        quatE = factor * X
        quatN = factor * gamma
        
        quat = np.array([*quatE, quatN])
        return quat/np.linalg.norm(quat)

    def __calc_B(self, eci:np.ndarray, cv:np.ndarray, weights:np.ndarray)->np.ndarray:
        B = np.zeros((3,3))        
        for i in range(len(eci)):
            B += weights[i] * np.outer(eci[i], cv[i])
        return np.transpose(B)

    def __calc_K(self, B:np.ndarray)->tuple[np.ndarray]:
        k_12 = np.array([B[1][2] - B[2][1],
                         B[2][0] - B[0][2],
                         B[0][1] - B[1][0]])

        return (np.transpose(k_12), np.trace(B))

    def __calc_lam_params(self, K_12:np.ndarray, K_22:float, S:np.ndarray)->tuple[float]:
        a = K_22**2 - np.trace(self.__calc_adjoint(S))
        b = K_22**2 + K_12.T @ K_12
        c = np.linalg.det(S) + K_12.T @ S @ K_12
        d = K_12.T @ S @ S @ K_12
        return (a, b, c, d)

    def __calc_quat_params(self, lam:float, K_22:float, S:np.ndarray)->tuple[float]:
        alpha = lam**2 - K_22**2 + np.trace(self.__calc_adjoint(S))
        beta = lam - K_22
        gamma = (lam + K_22)*alpha - np.linalg.det(S)
        return (alpha, beta, gamma)

    def __calc_adjoint(self, B:np.ndarray)->np.ndarray:
        deter = np.linalg.det(B)
        cofactor = np.linalg.inv(B).T * deter

        return np.transpose(cofactor)

    def __calc_optimal_lambda(self, a:float, b:float, c:float, d:float, k:float, lam0:float=1.0, *,eps:float=1e-12, max_iters:int=1000):

        e0 = lam0
        e1 = lam0 - self.__optimal_lambda_f(e0, a, b, c, d, k)/self.__optimal_lambda_fp(e0, a, b, c)
        err = np.abs(e1 - e0)

        # newton solver
        i = 0
        while err > eps and i < max_iters:
            e0 = e1
            e1 = e0 - self.__optimal_lambda_f(e0, a, b, c, d, k)/self.__optimal_lambda_fp(e0, a, b, c)
            err = np.abs(e1 - e0)
            i += 1
        
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
