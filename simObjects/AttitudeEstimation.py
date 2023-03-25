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
    
    def __create_star_frame(self):
        """
        IMAGE_X => WHERE LIGHT HIT SENSOR (random)
        IMAGE_Y => WHERE LIGHT HIT SENSOR (random)
        PX_X => REAL X-POS OF CV    (coupled to image x, state)
        PX_Y => REAL Y-POS OF CV    (coupled to image y, state)
        PX_Z => REAL Z-POS OF CV    (coupled to image z (0), state)
        """

        # create dataframe
        frame = pd.DataFrame()

        # set perceived information
        frame['IMAGE_X'] = np.random.uniform(0, c.SENSOR_WIDTH, int(self.state.NUM_STARS_SENSOR))
        frame['IMAGE_Y'] = np.random.uniform(0, c.SENSOR_HEIGHT, int(self.state.NUM_STARS_SENSOR))

        frame['IMAGE_DEV_X'] = frame['IMAGE_X'] + np.random.uniform(-1*np.abs(self.state.BASE_DEV_X), np.abs(self.state.BASE_DEV_X), len(frame.index))
        frame['IMAGE_DEV_Y'] = frame['IMAGE_Y'] + np.random.uniform(-1*np.abs(self.state.BASE_DEV_Y), np.abs(self.state.BASE_DEV_Y), len(frame.index))

        f_len = self.state.FOCAL_LENGTH - self.state.D_FOCAL_LENGTH

        # create camera vectors
        frame['CV_EST'] = frame[['IMAGE_DEV_X', 'IMAGE_DEV_Y']].apply(self.__px_to_cv, axis=1, args=(f_len, ))

        # set real information
        frame['CV_REAL'] = frame[['IMAGE_X', 'IMAGE_Y']].apply(self.__set_real_px, axis=1)

        
        # set ECI vector
        rot = lambda x: self.C @ x
        frame['ECI_REAL'] = frame['CV_REAL'].apply(rot)

        return frame

    def __px_to_cv(self, row:pd.Series, f_len:float)->np.ndarray:
        x = row[0]
        y = row[1]
        z = f_len

        cvx = x - c.SENSOR_WIDTH/2
        cvy = y - c.SENSOR_HEIGHT/2
        cvz = z

        v = np.array([cvx, cvy, cvz])
        return v/np.linalg.norm(v)

    def __set_real_rotation(self)->None:
        
        # quaternion representation
        q = np.random.uniform(-1,1,4)
        q = q/np.linalg.norm(q)
        
        # rotation matrix
        
        skew = lambda n: np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        e = -1*q[:3]
        n = q[3]
        
        Ca = (2*n**2 - 1) * np.identity(3)
        Cb = 2*np.outer(e, e)
        Cc = -2*n*skew(e)

        C = Ca + Cb + Cc

        return q, C
    
    def __set_real_px(self, img:pd.Series)->pd.Series:

        cvx = img[0] - c.SENSOR_WIDTH/2 # delta_x from ppt
        cvy = img[1] - c.SENSOR_HEIGHT/2    # delta_y from ppt

        """
        Focal Length Deviation
        """
        cvz = self.state.FOCAL_LENGTH   # this is the unobservable Focal Length

        """
        Principal Point Deviation (affects distortion and inclination calc)
        """
        ppt_dev = np.random.uniform(-1, 1, 2)   # random movement within some bounded circle about 0
        ppt_dev = self.state.PRINCIPAL_POINT_ACCURACY * ppt_dev/np.linalg.norm(ppt_dev)
        cvx += ppt_dev[0]
        cvy += ppt_dev[1]

        """
        Lens Distortion (expressed as a percent)
        """
        D = self.state.DISTORTION/100
        R = np.linalg.norm([cvx, cvy])
        Rp = R * (1 - D)
        cvx = Rp * cvx/R
        cvy = Rp * cvy/R
        
        """
        Focal Array Inclination
        """
        inc = np.deg2rad(self.state.FOCAL_ARRAY_INCLINATION)
        del_x = np.abs(cvx) - np.abs(cvx * np.cos(inc))
        del_z = np.abs(cvx * np.sin(inc))
        
        cvx = np.sign(cvx) * (np.abs(cvx) - del_x)    # tilting always brings x-coordinates towards center

        if np.sign(self.state.FOCAL_ARRAY_INCLINATION) == np.sign(cvx):
            cvz -= del_z    # focal length shortens if on "uphill"
        else:
            cvz += del_z    # focal length increases if on "downhill"

        """
        Convert to Unit Vector
        """
        v = np.array([cvx, cvy, cvz])

        return v/np.linalg.norm(v)

  
class QUEST:

    def __init__(self, eci_real:np.ndarray, cv_real:np.ndarray, cv_est:np.ndarray, ident:float, q:np.ndarray=None):

        self.eci_real = eci_real
        self.cv_real = cv_real
        self.cv_est = cv_est
        self.id = ident
        self.q_real = q

        return

    def __repr__(self)->str:
        name = 'QUEST SOLVER, {}:\n{}\n'.format(self.frame)
        return name
    
    def calc_acc(self)->float:

        real_quat = self.__get_attitude(self.eci_real, self.cv_real, truth=True)

        est_quat = self.__get_attitude(self.eci_real, self.cv_est, truth=False)
        q_diff = self.__quat_accuracy(real_quat, est_quat)

        return q_diff

    def __remove_stars(self, eci_real:np.ndarray, cv_est:np.ndarray)->np.ndarray:

        # artificially remove rows from starframe
        ident_chance = np.random.uniform(0, 1, len(eci_real))

        del_rows = np.argwhere(ident_chance > self.id)
    
        eci = np.delete(eci_real, del_rows, axis=0)
        cv = np.delete(cv_est, del_rows, axis=0)

        return eci, cv

    def __get_attitude(self, eci_real:np.ndarray, cv:np.ndarray, truth:bool=True)->np.ndarray:
        
        # extract eci and cv vectors
        eci_vecs = eci_real
        cv_vecs = cv

        if not truth:
            eci_vecs, cv_vecs = self.__remove_stars(eci_vecs, cv_vecs)
            
            if len(eci_vecs) == 0:
                return -1

        weights = np.ones(len(eci_vecs))/len(eci_vecs)

        # calc lam params
        B = self.__calc_B(eci_vecs, cv_vecs, weights)
        K_12, K_22 = self.__calc_K(B)
        S = B + np.transpose(B)
        a, b, c_lam, d = self.__calc_lam_params(K_12, K_22, S)
        
        # get optimal lam
        lam = self.__calc_optimal_lambda(a, b, c_lam, d, K_22)

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

    def __quat_accuracy(self, q_real:np.ndarray, q_est:np.ndarray)->float:

        conj_Q_calc = np.array([*(q_est[:3] * -1), q_est[3]])
        q_true = q_real
        
        q_err_e = q_true[3]*conj_Q_calc[:3] + conj_Q_calc[3]*q_true[:3] + np.cross(q_true[:3], conj_Q_calc[:3])
        q_err_n = q_true[3]*conj_Q_calc[3] - q_true[:3] @ conj_Q_calc[:3]
        q_err = np.array([*q_err_e, q_err_n])

        theta = 2 * np.arctan2(np.linalg.norm(q_err[:3]), q_err[3])
        theta_deg = np.rad2deg(theta)
        
        return theta_deg * 3600
