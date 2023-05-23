import sys
sys.path.append(sys.path[0] + '/..')

import numpy as np
from numpy.matlib import repmat
import pandas as pd
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
from .generateProjection import generate_projection, eci_to_cv_rotation

import constants as c
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)

class Projection:

    def __init__(self, sim_row:pd.Series, type:str):
        
        self.state = sim_row
        self.frame:pd.DataFrame = None
        self.type = type
        self.temp:float=25

        # self.centroid_model = lambda m: 0.00246601*m**2 + 0.00687582*m + 0.0211441
        self.centroid_model = lambda m:0.00333124*np.exp(0.406843*m)-0.00153084
        # ROTATIONS DESCRIBE ECI -> CAMERA VECTOR
        self.quat_real:np.ndarray = None
        self.C:np.ndarray = None


        return
    
    def skew_sym(self, n:np.ndarray)->np.ndarray:

        return np.array([[0, -n[2], n[1]],
                         [n[2], 0, -n[0]],
                         [-n[1], n[0], 0]])

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

    def PHI_S(self, sim_row:pd.Series)->np.ndarray:

        cv_true = sim_row.CV_TRUE
        # x_f = np.random.normal(7e-6, 3.9e-6)
        # fZ = (self.state.FOCAL_LENGTH * (self.temp-25)*x_f) + self.state.F_ARR_EPS_Z
        fZ = self.state.F_ARR_EPS_Z
        # logger.critical(fZ)


        phi = np.deg2rad(self.state.F_ARR_PHI)
        theta = np.deg2rad(self.state.F_ARR_THETA)
        psi = np.deg2rad(self.state.F_ARR_PSI)
        
        C_gamma_pi = c.Rx(phi) @ c.Ry(theta) @ c.Rz(psi)
        
        r_F_pi_gamma = np.array([self.state.F_ARR_EPS_X, 
                            self.state.F_ARR_EPS_Y,
                            fZ + self.state.FOCAL_LENGTH])
        
        r_F_pi_pi = C_gamma_pi @ r_F_pi_gamma
        r_S_F_pi = C_gamma_pi @ cv_true
        lam_star = (-r_F_pi_pi[2]) / (r_S_F_pi[2])
        P_star = lam_star * r_S_F_pi + r_F_pi_pi

        # add centroiding error
        if np.abs(self.state.BASE_DEV_X) > 0:
            if self.type == 'MC':
                cdiff = self.centroid_model(sim_row.v_magnitude)/np.sqrt(2)
                # cdiff = self.state.BASE_DEF_X
                # logger.critical(cdiff)
                P_star[0] += np.random.normal(0, cdiff)
                P_star[1] += np.random.normal(0, cdiff)
            
            else:
                t = np.random.uniform(0, 2*np.pi)
                P_star[0] += self.state.BASE_DEV_X * np.cos(t)
                P_star[1] += self.state.BASE_DEV_Y * np.sin(t)

        s_hat = np.array([-P_star[0], -P_star[1], self.state.FOCAL_LENGTH])

        sim_row['IMG_X'] = P_star[0]
        sim_row['IMG_Y'] = P_star[1]
        sim_row['CV_MEAS'] = s_hat / np.linalg.norm(s_hat)

        return sim_row
        
class NoiseSim:

    def __init__(self, frame:pd.DataFrame, state:pd.Series, temp:float=25):

        # INPUTS
        self.frame = frame
        self.state = state

        # STAR TRACKER PROPERTIES
        self.qe = 0.6
        self.t_i = 0.2
        self.window_size = 21
        self.sigma = 5
        self.star_counter = np.array([])
        self.prnu_factor = .01
        self.pA = (3.5e-6)**2   # mm2
        self.temp = temp+ 273.15
        # self.temp = np.random.uniform(0, 60)+273.15
        self.darknoise_dn = 0.1# * 1.5
        self.max_e = 10_000
        self.eg0 = 1.1557
        self.alpha = 7.021e-4
        self.beta = 1108
        self.asn = 5e-6
        self.reset_factor = 0.2
        self.v_ref = 3.3
        self.bits = 12

        # CLASS PROPERTIES
        self.img_data = np.zeros((c.SENSOR_HEIGHT, c.SENSOR_WIDTH))
        self.dark_noise = np.zeros(self.img_data.shape)
        self.signal = np.zeros(self.img_data.shape)

        """ 
        START NOISE SIMULATION
        """
        # Add photons to sensor
        for _, row in self.frame.iterrows():
            x = int(np.round(row.IMG_X + c.SENSOR_HEIGHT/2))
            y = int(np.round(row.IMG_Y + c.SENSOR_HEIGHT/2))
            u_in = self.__calc_photons(row.v_magnitude)
            self.__add_star_to_image(x, y, u_in)

        self.frame['STAR_POWER'] = self.star_counter

        # Shot Noise (Signal)
        self.img_data = np.random.poisson(self.img_data)
        
        # QE (Signal)
        self.img_data = self.img_data * self.qe

        # PRNU (Signal)
        prnu = self.__get_prnu_signal()
        self.img_data = np.multiply(self.img_data, 1 + (prnu * self.prnu_factor))

        # Dark Current
        pacm = self.pA * 1e4
        eg = self.eg0 - (self.alpha*self.temp**2) / (self.beta + self.temp)
        # eg = 2.1
        dE = self.t_i * 2.55e15 * pacm * self.temp**1.5 * np.exp(-eg / (2*self.temp * 8.6173e-5))
        
        self.dark_noise = dE * np.ones(self.dark_noise.shape)

        # Shot Noise (Dark)
        self.dark_noise = np.random.poisson(self.dark_noise)

        # FPN (Dark)
        fpn = self.__get_prnu_signal()
        self.dark_noise = np.multiply(self.dark_noise, 1 + (self.darknoise_dn * 1*fpn))
        # logger.critical('DARK: {} +/- {}'.format(self.dark_noise.mean(), self.dark_noise.std()))
        
        # Combine noise and signal; check bounds
        # self.img_data = self.img_data*5
        self.signal = (self.img_data + self.dark_noise)
        self.signal[self.signal < 0] = 0
        self.signal[self.signal > self.max_e] = self.max_e

        # round down
        self.signal = np.floor(self.signal)

        # CONV to VOLTAGE
        sn_cap = 1.6022e-19 / self.asn
        vmin = 1.6022e-19 * self.asn / sn_cap
        vmax = self.max_e * 1.6022e-19 / sn_cap

        reset_noise_sigma = np.sqrt((1.3807e-23)*(self.temp)/(sn_cap))
        reset_noise = np.exp(reset_noise_sigma * np.random.normal(0, 1, self.img_data.shape)) - 1
        self.signal_voltage = (self.v_ref + self.reset_factor*reset_noise) - (self.signal * self.asn)

        # Column FPN
        column_noise = np.random.normal(0, 1, c.SENSOR_HEIGHT)
        cds_noise = repmat(column_noise, c.SENSOR_HEIGHT, 1)
        self.signal_voltage = np.multiply(self.signal_voltage, 1 + cds_noise*(vmax * 5e-4))
        
        # CONV to DN
        n_max = 2**self.bits
        adc_gain= n_max / (vmax - vmin)
        self.signal_dn = np.round(adc_gain * (self.v_ref - self.signal_voltage))

        # HARDWARE LIMITS
        self.signal_dn[self.signal_dn < 0] = 0
        self.signal_dn[self.signal_dn > n_max] = n_max

        # SOFTWARE FILTER
        self.pre_filter = deepcopy(self.signal_dn)
        thresh = self.signal_dn.mean() + 5*self.signal_dn.std()
        self.signal_dn[self.signal_dn <= thresh] = 0
        self.frame = self.frame[self.frame.STAR_POWER > thresh]

        self.__plot_analysis(len(frame.index))
        raise ValueError
        return
    
    def __plot_analysis(self, tot_star_count:int):
        plt.rcParams['text.usetex'] = True
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        raw = ax.imshow(self.pre_filter)
        ax.set_title('{} / {} Stars Visible'.format(len(self.frame.index), tot_star_count), fontsize=15)
        cbar = fig.colorbar(raw, ax=ax)
        
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Brightness Intensity', rotation=90, fontsize=12)
        # fig = plt.figure()
        ax = fig.add_subplot(2,1,2)
        new = ax.imshow(self.signal_dn)
        
        cbar = fig.colorbar(new, ax=ax)

        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel('Brightness Intensity', rotation=90, fontsize=12)

        plt.show()
        return

    def __calc_photons(self, mag):
        return 19_100/self.qe * 1/(2.5**mag) * self.t_i * np.pi * (12.25/(2*1.4))**2

    def __add_star_to_image(self, x:int, y:int, electron_count:float=500_000)->None:

        roi = np.zeros((self.window_size, self.window_size))
        roi[self.window_size//2][self.window_size//2] = electron_count
        roi = gaussian_filter(roi, sigma=self.sigma)
        roi= 5*roi
        for i, ic in enumerate(range(x - self.window_size//2 -1, x + self.window_size//2, 1)):
            for j, jc in enumerate(range(y-self.window_size//2 -1, y+self.window_size//2, 1)):
                
                if ic < 0 or ic >= c.SENSOR_WIDTH:
                    continue
                if jc < 0 or jc >= c.SENSOR_WIDTH:
                    continue
                self.img_data[ic, jc] += roi[i, j] 

        # append star center to keep track for threshholding
        self.star_counter = np.array([*self.star_counter, roi[self.window_size//2][self.window_size//2]])

        return

    def __get_prnu_signal(self):
        return np.random.normal(0, 1, self.img_data.shape)

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

    def __init__(self, sim_row:pd.Series, catalog:pd.DataFrame, type:str):
        super().__init__(sim_row, type)

        fov = 2 * np.arctan2(c.SENSOR_HEIGHT/2, sim_row.FOCAL_LENGTH)
        self.frame = generate_projection(starlist=catalog,
                                         ra=self.state.RIGHT_ASCENSION,
                                         dec=self.state.DECLINATION,
                                         roll=self.state.ROLL,
                                         camera_fov=fov,
                                         max_magnitude=self.state.MAX_MAGNITUDE)

        # logger.critical(f'{c.GREEN}NUM STARS: {len(self.frame.index)}{c.DEFAULT}')
        # self.temp = np.random.uniform(0, 60)
        if len(self.frame.index) > 1:
            
            # if self.state.MAX_MAGNITUDE < 10:
                # logger.critical('yo')
            self.frame = self.frame[self.frame.v_magnitude <= self.state.MAX_MAGNITUDE]
                

            self.C = eci_to_cv_rotation(self.state.RIGHT_ASCENSION, self.state.DECLINATION,self.state.ROLL)
            self.quat_real = self.rotm_to_quat(self.C)

            self.frame = self.frame.apply(self.PHI_S, axis=1)
            # self.frame['TEMP'] = self.temp*np.ones(self.frame.R)
        

            # if self.state.MAX_MAGNITUDE < 10:
            # self.noise_model = NoiseSim(self.frame, self.state, self.temp)
            #     # self.pre_maxmag = self.frame.v_magnitude.max()
            #     # self.pre_numstar = len(self.frame.index)
            #     # logger.critical(len(self.noise_model.frame.index))

                
            # #     # logger.critical(self.noise_model.frame.v_magnitude.mean())
            # #     # logger.critical(self.noise_model.frame.v_magnitude.std())
            # #     # logger.critical(self.frame.v_magnitude.max())
            # #     # logger.critical(self.noise_model.frame.v_magnitude.max())
            # #     # logger.critical(self.noise_model.)
            # #     # raise ValueError
            # self.frame = self.noise_model.frame


