import sys

import numpy as np
import pandas as pd

sys.path.append(sys.path[0] + '/..')
import json

import constants as c

from .Parameter import Parameter, UniformParameter
import logging

logger = logging.getLogger(__name__)
AVG_NUM_STARS = 7
STD_NUM_STARS = 2

class LensThermalEffect:
    def __init__(self):
        
        # 1/f df/dt from "Thermal effects in optical systems", Jamieson
        self.x_f =   np.array([10.77,
                19.73,
                27.50,
                0.99,
                8.29,
                1.11,
                8.92,
                3.28,
                -5.03,
                0.98,
                -1.72,
                3.54,
                -0.53,
                4.24,
                -2.38,
                -9.27,
                0.79,
                -10.37,
                2.68,
                -0.23,
                -11.99,
                -1.82,
                2.24,
                -2.80,
                10.09,
                0.99,
                -1.93,
                -3.30,
                5.36,
                -1.94,
                0.83,
                4.05,
                8.44,
                5.42,
                -1.34,
                0.32,
                -3.65,
                2.09,
                0.89,
                -2.18,
                4.46,
                3.90,
                2.52,
                0.68,
                -0.37,
                6.65,
                2.32,
                -9.26,
                5.00,
                -5.67,
                3.41,
                -4.30,
                1.85,
                -3.13,
                0.55,
                -10.61,
                4.24,
                -6.85,
                14.63,
                7.66,
                20.94,
                -0.71,
                1.66,
                -2.89,
                -4.73,
                16.83,
                -64.10,
                -85.19,
                -28.24,
                92.09,
                227.87,
                137.17,
                132.04]) * 1e-6

        self.temp_param = Parameter(self.x_f.mean(),
                         self.x_f.std(),
                         0,
                         name='FOCAL_THERMAL_COEFFICIENT')
        return 

class StarVisibility:

    def __init__(self, magnitude:float=5):

        self.cat_data:pd.DataFrame = pd.read_pickle(c.BSC5PKL)
        self.star_vis = self.__create_vis_param(magnitude)

        return
    
    def __create_vis_param(self, mag:float, trials:int=1_000_000):

        mod_df = self.cat_data[self.cat_data['v_magnitude'] <= mag]
        

        return

class StarTracker:

    def __init__(self, principal_point_accuracy:Parameter=None,
                       focal_length:Parameter=None,
                       array_tilt:Parameter=None,
                       distortion:Parameter=None,
                       sensor:Parameter=None,
                       cam_name:str='IDEAL_CAM',
                       cam_json:str=c.IDEAL_CAM)->None:

        # open cam property file
        self.camJSON = cam_json
        self.cam_name = self.__set_name(cam_name)

        # set properties of camera
        # self.eps_x = Parameter(0, 0, name='F_ARR_EPS_X')
        # self.eps_y = Parameter(0, 0, name='F_ARR_EPS_Y')
        # self.eps_z = Parameter(0, 0, name='F_ARR_EPS_Z')

        # self.phi = Parameter(0, 0, name='F_ARR_PHI')
        # self.theta = Parameter(0, 0, name='F_ARR_THETA')
        # self.psi = Parameter(0, 0, name='F_ARR_PSI')

        self.eps_x = Parameter(0, .1, name='F_ARR_EPS_X')
        self.eps_y = Parameter(0, .1, name='F_ARR_EPS_Y')
        self.eps_z = Parameter(0, 1, name='F_ARR_EPS_Z')

        self.phi = Parameter(0, .01, name='F_ARR_PHI')
        self.theta = Parameter(0, .01, name='F_ARR_THETA')
        self.psi = Parameter(0, .01, name='F_ARR_PSI')


        self.ppt_acc = self.__set_parameter(principal_point_accuracy, "PRINCIPAL_POINT_ACCURACY")
        self.array_tilt = self.__set_parameter(array_tilt, "FOCAL_ARRAY_INCLINATION")
        self.distortion = self.__set_parameter(distortion, "DISTORTION")
        
        self.sensor = Parameter(AVG_NUM_STARS, STD_NUM_STARS, 0, name="NUM_STARS_SENSOR", units="", retVal=lambda x: np.max([1, int(x)]))
        
        f_len_mm = self.__set_parameter(focal_length, "FOCAL_LENGTH")
        f_len_mean, f_len_std = f_len_mm.get_prob_distribution()
        self._fov = self.__set_img_fov(f_len_mm=f_len_mm)

        f_len_px = f_len_mm.ideal / self._pixelX
        f_len_mean = f_len_mean / self._pixelX
        f_len_std = f_len_std / self._pixelX

        self.f_len = Parameter(ideal=f_len_px, stddev=f_len_std, mean=f_len_mean, name=f_len_mm.name, units='px')
        self.f_len_dtemp = LensThermalEffect().temp_param

        self.params = {param.name:param for param in self.__all_params()}

        self.data = self.randomize()
        self.reset_params()

        return

    def __repr__(self):
        cname = "Camera: {}"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}\n".format(self.cam_name,
                            repr(self.f_len),
                            repr(self.ppt_acc),
                            repr(self.array_tilt),
                            repr(self.distortion))

        return cname

    def __set_name(self, cam_name:str)->str:
        if cam_name is not None:
            return cam_name
        
        js = json.load(open(self.camJSON))
        return js['CAMERA_NAME']

    def __set_parameter(self, parameter:Parameter, name:str)->Parameter:
        if parameter is not None:
            return parameter

        return Parameter.init_from_json(self.camJSON, name)

    def __set_img_fov(self, f_len_mm)->float:

        f_len = f_len_mm

        cam = json.load(open(self.camJSON))

        self._imgX = cam['IMAGE_WIDTH']
        self._imgY = cam['IMAGE_HEIGHT']
        self._pixelX = cam['PIXEL_WIDTH']
        self._pixelY = cam['PIXEL_HEIGHT']

        wd = self._imgX * self._pixelX
        ht = self._imgY * self._pixelY
        x = np.sqrt(wd**2 + ht**2)

        fov = np.rad2deg(2*np.arctan(x/(2*f_len.value)))

        return fov

    def randomize(self, mod_param:list=None, num:int=1_000)->pd.DataFrame:

        df = pd.DataFrame()

        if mod_param is None:
            mod_param = list(self.params.keys())

        for param_name in self.params:
            if param_name in mod_param:
                df[param_name] = self.params[param_name].modulate(num)
            else:
                df[param_name] = self.params[param_name].ideal * np.ones(num)

        # only parameter that is not 0 mean; need to know the delta 
        df['D_FOCAL_LENGTH'] = df['FOCAL_LENGTH'] - self.f_len.ideal
        self.data = df

        return df

    def ideal(self, num:int=1_000)->pd.DataFrame:

        df = pd.DataFrame()

        for param_name in self.params:
            df[param_name] = self.params[param_name].reset(num)

        # only parameter that is not 0 mean; need to know the delta 
        df['D_FOCAL_LENGTH'] = np.zeros(len(df.index))
        self.data = df

        return df

    def reset_params(self)->None:
        for param in self.__all_params():
            param.reset()
        return
    
    def __all_params(self)->tuple[Parameter]:
        return [self.f_len, self.f_len_dtemp, self.sensor, self.array_tilt, self.distortion, self.ppt_acc, self.eps_x, self.eps_y, self.eps_z, self.phi, self.theta, self.psi]
            