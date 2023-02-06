import sys

import numpy as np
import pandas as pd

sys.path.append(sys.path[0] + '/..')
import json

import constants as c

from .Parameter import Parameter


class StarTracker:

    def __init__(self, principal_point_accuracy:Parameter=None,
                       focal_length:Parameter=None,
                       array_tilt:Parameter=None,
                       distortion:Parameter=None,
                       cam_name:str='IDEAL_CAM',
                       cam_json:str=c.IDEAL_CAM)->None:

        # open cam property file
        self.camJSON = cam_json
        self.cam_name = self.__set_name(cam_name)

        # set properties of camera
        self.ppt_acc = self.__set_parameter(principal_point_accuracy, "PRINCIPAL_POINT_ACCURACY")
        self.array_tilt = self.__set_parameter(array_tilt, "FOCAL_ARRAY_INCLINATION")
        self.distortion = self.__set_parameter(distortion, "DISTORTION")

        f_len_mm = self.__set_parameter(focal_length, "FOCAL_LENGTH")
        self._fov = self.__set_img_fov(f_len_mm=f_len_mm)

        f_len_px = f_len_mm.ideal / self._pixelX
        self.f_len = Parameter(ideal=f_len_px, stddev=f_len_px*0.001, mean=0, name=f_len_mm.name, units='px')

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

    def reset_params(self)->None:
        self.f_len.reset()
        self.ppt_acc.reset()
        self.array_tilt.reset()
        self.distortion.reset()
        return
    
    def all_params(self)->tuple[Parameter]:
        return [self.f_len, self.array_tilt, self.distortion, self.ppt_acc]