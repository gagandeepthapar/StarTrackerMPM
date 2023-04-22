import sys

sys.path.append(sys.path[0] + '/..')

from json import load as jsonload

import numpy as np
import pandas as pd
from .Parameter import Parameter, UniformParameter
from .generateProjection import generate_projection, eci_to_cv_rotation
import matplotlib.pyplot as plt

import constants as c
import logging

logger = logging.getLogger(__name__)


class Projection:

    def __init__(self, sim_row:pd.Series):
        
        self.state = sim_row
        self.frame:pd.DataFrame = None

        # ROTATIONS DESCRIBE ECI -> CAMERA VECTOR
        self.quat_real:np.ndarray = None
        self.C:np.ndarray = None

        return
    
    def skew_sym(self, n:np.ndarray)->np.ndarray:

        return np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

    def quat_to_rotm(self, quat:np.ndarray)->np.ndarray: 

        e = quat[:3]
        n = quat[3]
        
        Ca = (2*n**2 - 1) * np.identity(3)
        Cb = 2*np.outer(e, e)
        Cc = -2*n*self.skew_sym(e)

        C = Ca + Cb + Cc

        return C
    
    def rotm_to_quat(self, C:np.ndarray)->np.ndarray:
        
        q4 = (np.trace(C) + 1) ** 0.5 / 2
        q1 = (C[1][2] - C[2][1]) / (4 * q4)
        q2 = (C[2][0] - C[0][2]) / (4 * q4)
        q3 = (C[0][1] - C[1][0]) / (4 * q4)

        return np.array([q1, q2, q3, q4])

    def PHI_S(self, cv_true:np.ndarray)->np.ndarray:

        phi = np.deg2rad(self.state.F_ARR_PHI)
        theta = np.deg2rad(self.state.F_ARR_THETA)
        psi = np.deg2rad(self.state.F_ARR_PSI)
        C_gamma_pi = c.Rx(phi) @ c.Ry(theta) @ c.Rz(psi)
        
        r_F_pi_gamma = np.array([self.state.F_ARR_EPS_X, 
                            self.state.F_ARR_EPS_Y,
                            self.state.F_ARR_EPS_Z + self.state.FOCAL_LENGTH])
        
        r_F_pi_pi = C_gamma_pi @ r_F_pi_gamma
        r_S_F_pi = C_gamma_pi @ cv_true
        lam_star = (-1 * r_F_pi_pi[2]) / (r_S_F_pi[2])
        P_star = lam_star * r_S_F_pi + r_F_pi_pi
        P_star[0] += self.state.BASE_DEV_X
        P_star[1] += self.state.BASE_DEV_Y

        s_hat = np.array([-P_star[0], -P_star[1], self.state.FOCAL_LENGTH])

        return s_hat / np.linalg.norm(s_hat)

class RandomProjection(Projection):

    def __init__(self, sim_row:pd.Series):

        super().__init__(sim_row)

        self.quat_real = self.__set_real_rotation()
        self.C = self.quat_to_rotm(self.quat_real)
        self.frame = self.__create_star_frame()

        # INVERSE ROTATIONS TO GET ECI -> CV DESCRIPTION
        self.quat_real = np.array([*(-1*self.quat_real[:3]), self.quat_real[3]])
        self.C = self.quat_to_rotm(self.quat_real)
        # self.C = self.C.T
        # self.quat_real = self.rotm_to_quat(self.C)

        return

    def __repr__(self)->str:
        name = 'PROJECTION @ {}'.format(self.quat_real)
        return name
    
    def __create_star_frame(self):

        # create dataframe
        frame = pd.DataFrame()

        frame['IDX'] = np.array([i for i in range(int(self.state.NUM_STARS_SENSOR))])
        frame['CV_TRUE'] = frame.apply(lambda v: self.__s_hat_in_fov(), axis=1)
        frame['ECI_TRUE'] = frame['CV_TRUE'].apply(lambda v: self.C @ v)
        frame['CV_MEAS'] = frame['CV_TRUE'].apply(self.PHI_S)

        return frame

    def __s_hat_in_fov(self)->np.ndarray:
        fov = np.arctan2(c.SENSOR_HEIGHT/2,self.state.FOCAL_LENGTH)
        phi = np.random.uniform(-fov, fov)  
        theta = np.random.uniform(-np.pi, np.pi)
        return np.array([np.sin(phi)*np.cos(theta), np.sin(phi)*np.sin(theta), np.cos(phi)])

    def __set_real_rotation(self)->None:
        
        # quaternion representation
        q = np.random.uniform(-1,1,4)
        q = q/np.linalg.norm(q)
        
        return q

class StarProjection(Projection):

    def __init__(self, sim_row:pd.Series, catalog:pd.DataFrame):
        super().__init__(sim_row)

        fov = 2 * np.arctan2(c.SENSOR_HEIGHT/2, sim_row.FOCAL_LENGTH)
        self.frame = generate_projection(starlist=catalog,
                                         ra=self.state.RIGHT_ASCENSION,
                                         dec=self.state.DECLINATION,
                                         roll=self.state.ROLL,
                                         camera_fov=fov,
                                         max_magnitude=self.state.MAX_MAGNITUDE)

        ra = self.state.RIGHT_ASCENSION
        dec = self.state.DECLINATION

        boresight = np.array([np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)])
        self.C = eci_to_cv_rotation(self.state.ROLL, boresight)
        self.quat_real = self.rotm_to_quat(self.C)

        self.__add_measurement_noise()

    def __add_measurement_noise(self)->None:
        self.frame['CV_MEAS'] = self.frame['CV_TRUE'].apply(self.PHI_S)
        return


class QUEST:

    def __init__(self, eci_real:np.ndarray, cv_real:np.ndarray, cv_est:np.ndarray, q:np.ndarray=None):

        self.eci_real = eci_real
        self.cv_real = cv_real
        self.cv_est = cv_est
        self.q_real = q

        return

    def __repr__(self)->str:
        name = 'QUEST SOLVER, {}:\n{}\n'.format(self.frame)
        return name
    
    def calc_acc(self)->float:

        real_quat = self.__get_attitude(self.eci_real, self.cv_real, truth=True)
        est_quat = self.__get_attitude(self.eci_real, self.cv_est, truth=False)
        
        q_diff_A = self.__quat_accuracy(self.q_real, est_quat)
        q_diff_B = self.__quat_accuracy(-1*self.q_real, est_quat)

        min_diff = np.mod(min(q_diff_A, q_diff_B), 3600*360)

        if min_diff > 3600:
            # print(self.q_real, est_quat, min_diff)
            print('{}THEORETICAL REAL:\n\t{}\nCALC REAL:\n\t{}\nCALC EST:\n\t{}\nMIN DIFF:\n\t{}{}'.format(c.GREEN, self.q_real, real_quat, est_quat, min_diff, c.DEFAULT))

            # fig = plt.figure()
            # ax = fig.add_subplot(projection='3d')

            # ax.scatter(*self.q_real[:3], color='green', label='Theoretical Real')
            # ax.scatter(*real_quat[:3], color='blue', label='Calc Real Quat')
            # ax.scatter(*est_quat[:3], color='red', label='Est Quat')

            # ax.plot([0, self.q_real[0]], [0, self.q_real[1]], [0, self.q_real[2]], color='green')
            # ax.plot([0, est_quat[0]], [0, est_quat[1]], [0, est_quat[2]], color='red')
            # ax.plot([0, real_quat[0]], [0, real_quat[1]], [0, real_quat[2]], color='blue')

            # ax.legend()
            # ax.axis('equal')

            # plt.show()
            # return -1
            raise ValueError

        return min(q_diff_A, q_diff_B)

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

        # if not truth:
        #     eci_vecs, cv_vecs = self.__remove_stars(eci_vecs, cv_vecs)
            
        #     if len(eci_vecs) <= 1:
        #         return -1

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
        return quat
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

        # SRC: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6864719/
        #https://math.stackexchange.com/questions/3572459/how-to-compute-the-orientation-error-between-two-3d-coordinate-frames

        if type(q_est) is int:
            logger.critical('{}Failed Ident{}'.format(c.RED,c.DEFAULT))
            return -1
        
        conj_Q_calc = np.array([*(q_est[:3] * -1), q_est[3]])
        q_true = q_real

        # if q_true.dot(conj_Q_calc) < 0:
        #     conj_Q_calc *= -1
        
        q_err_e = q_true[3]*conj_Q_calc[:3] + conj_Q_calc[3]*q_true[:3] - np.cross(conj_Q_calc[:3], q_true[:3])
        q_err_n = q_true[3]*conj_Q_calc[3] - q_true[:3] @ conj_Q_calc[:3]
        q_err = np.array([*q_err_e, q_err_n])

        theta = np.arctan2(np.linalg.norm(q_err[:3]), q_err[3])
        theta_deg = np.rad2deg(theta)
        
        return theta_deg * 3600
