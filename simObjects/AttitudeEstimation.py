import sys

sys.path.append(sys.path[0] + '/..')

from json import load as jsonload

import numpy as np
import pandas as pd
from .Parameter import Parameter, UniformParameter

import constants as c
import logging

logger = logging.getLogger(__name__)

class Projection:

    def __init__(self, sim_row:pd.Series):

        self.state = sim_row

        self.quat_real, self.C = self.__set_real_rotation()

        self.frame = self.__create_star_frame()

        return

    def __repr__(self)->str:
        name = 'PROJECTION @ {}'.format(self.quat_real)
        return name
    













        


        
        

  




            # cvx -= sim_row.PRINCIPAL_POINT_ACCURACY
            # cvy -= sim_row.PRINCIPAL_POINT_ACCURACY

        v = np.array([cvx, cvy, f])
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

    def __init__(self, Centroid_Deviation_X: Parameter = None,
                       Centroid_Deviation_Y: Parameter = None,
                       img_width: float = None,
                       img_height: float = None,
                       img_focal: float = None, 
                       sim_row:pd.Series=None, *,
                       centroidFP: str = c.SIMPLE_CENTROID,
                       cameraFP: str = c.ALVIUM_CAM,
                       numStars: Parameter = Parameter(7, 2, 0, name="NUM_STARS", units="", retVal=lambda x: np.max([1, int(np.round(x))]))) -> None:
        
        super().__init__(Centroid_Deviation_X, Centroid_Deviation_Y, img_width, img_height, img_focal, sim_row=sim_row,
                         centroidFP=centroidFP, cameraFP=cameraFP, numStars=numStars)

    def __repr__(self)->str:
        name = 'QUEST SOLVER, {}:\n{}\n'.format(self.frame)
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

