import numpy as np

import sys
sys.path.append(sys.path[0] + '/..')
import constants as c

import json

from .Parameter import Parameter


class StarTracker:

    def __init__(self, centroid_accuracy:Parameter=None,
                       principal_point_accuracy:Parameter=None,
                       focal_length:Parameter=None,
                       array_tilt:Parameter=None,
                       distortion:Parameter=None,
                       cam_name:str=None,
                       cam_json:str=c.IDEAL_CAM)->None:

        # open cam property file
        self.camJSON = cam_json
        self.cam_name = self._set_name(cam_name)

        # set properties of camera
        self.ctr_acc = self._set_parameter(centroid_accuracy, "CENTROID_ACCURACY")
        self.ppt_acc = self._set_parameter(principal_point_accuracy, "PRINCIPAL_POINT_ACCURACY")
        self.array_tilt = self._set_parameter(array_tilt, "FOCAL_ARRAY_INCLINATION")
        self.distortion = self._set_parameter(distortion, "DISTORTION")

        f_len_mm = self._set_parameter(focal_length, "FOCAL_LENGTH")
        self._fov = self._set_img_fov(f_len_mm=f_len_mm)

        f_len_px = f_len_mm.ideal / self._pixelX
        self.f_len = Parameter(ideal=f_len_px, stddev=f_len_mm._err_stddev, mean=f_len_mm._err_mean, name=f_len_mm.name, units='px')

        self._num_stars = self.__set_est_num_stars()
        
        self.reset_params()

        return

    def __repr__(self):
        cname = "Camera: {} ({} Stars per frame)"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}\n".format(self.cam_name,
                            self._num_stars,
                            repr(self.f_len),
                            repr(self.ctr_acc),
                            repr(self.ppt_acc),
                            repr(self.array_tilt),
                            repr(self.distortion))

        return cname

    def _set_name(self, cam_name:str)->str:
        if cam_name is not None:
            return cam_name
        
        js = json.load(open(self.camJSON))
        return js['CAMERA_NAME']

    def _set_parameter(self, parameter:Parameter, name:str)->Parameter:
        if parameter is not None:
            return parameter

        return Parameter._init_from_json(self.camJSON, name)

    def _set_img_fov(self, f_len_mm)->float:

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

    def __set_est_num_stars(self)->float:
        return 4

    def reset_params(self)->None:
        self.f_len.reset()
        self.ctr_acc.reset()
        self.ppt_acc.reset()
        self.array_tilt.reset()
        self.distortion.reset()
        return
    
    def all_params(self)->tuple[Parameter]:
        return [self.f_len, self.array_tilt, self.ctr_acc, self.distortion, self.ppt_acc]