import numpy as np
import constants as c
import json

class camera:
    """_summary_

    Returns:
        _type_: _description_
    """

    _centroid_std_dev = 0.1/3
    _principal_pt_std_dev = 4.5/3
    _f_len_std_dev = 0.6/3
    _f_array_inc_std_dev = 0.075/3
    _distortion_std_dev = 0.1/3

    def __init__(self, inc_angle:float=None,
                       centroid_acc:float=None,
                       principal_pt_acc:float=None,
                       f_len:float=None,
                       f_array_inc_angle:float=None,
                       distortion:float=None,
                       cam_json:str=c.simCameraSunEtal):

        # open cam property file
        self.camJSON = json.load(open(cam_json))

        # set incident angle of star
        # self._true_incident_angle = inc_angle

        # set misc camera properties
        self._true_incident_angle = self._set_true_value(inc_angle, "INCIDENT_ANGLE", self.camJSON)
        self._true_f_len = self._set_true_value(f_len, "FOCAL_LENGTH", self.camJSON)
        self._true_centroid_acc = self._set_true_value(centroid_acc, "CENTROID_ACCURACY", self.camJSON)
        self._true_principal_pt_acc = self._set_true_value(principal_pt_acc, "PRINCIPAL_POINT_ACCURACY", self.camJSON)
        self._true_f_array_inc_angle = self._set_true_value(f_array_inc_angle, "FOCAL_ARRAY_INCLINATION", self.camJSON)
        self._true_distortion = self._set_true_value(distortion, "DISTORTION", self.camJSON)

        # preallocate space for properties; defaulted to equal to true value
        self.f_len = self._true_f_len
        self.centroid_acc = self._true_centroid_acc
        self.principal_pt_acc = self._true_principal_pt_acc
        self.f_array_inc_angle = self._true_f_array_inc_angle
        self.distortion = self._true_distortion

        # set FOV
        # self._true_fov = self._get_fov(self._true_f_len)
        # self.fov = self._get_fov(self.f_len)

        return

    def __repr__(self):
        cname = f"Camera:" \
                f"\n\tFocal Length: {self.f_len} mm"

        return cname

    def reset_params(self)->None:

        self.f_len = self._true_f_len
        self.centroid_acc = self._true_centroid_acc
        self.principal_pt_acc = self._true_principal_pt_acc
        self.f_array_inc_angle = self._true_f_array_inc_angle
        self.distortion = self._true_distortion

        return

    def modulate_params(self, f_len:bool=False, centroid:bool=False, principal:bool=False, f_array:bool=False, distortion:bool=False)->None:

        if f_len:
            self.f_len = self._set_single_value(self._true_f_len, error_stddev=self._f_len_std_dev)
        
        if centroid:
            self.centroid_acc = self._set_single_value(self._true_centroid_acc, error_stddev=self._centroid_std_dev)
        
        if principal:
            self.principal_pt_acc = self._set_single_value(self._true_principal_pt_acc, error_stddev=self._principal_pt_std_dev)
        
        if f_array:
            self.f_array_inc_angle = self._set_single_value(self._true_f_array_inc_angle, error_stddev=self._f_array_inc_std_dev)
        
        if distortion:
            self.distortion = self._set_single_value(self._true_distortion, error_stddev=self._distortion_std_dev)
        
        return

    def _set_true_value(self, param_val:float, param_str:str, true_source:dict)->float:

        if param_val is None:
            return true_source[param_str]

        return param_val

    def _set_single_value(self, true_param:float, error_mean:float=0, error_stddev:float=0)->float:
        """Sets parameter based on true/target value and expected error

        Args:
            true_param (float): true/target value
            error_mean (float, optional): mean of error. Defaults to 0.
            error_stddev (float, optional): std dev of error. Defaults to 0.

        Returns:
            float: randomized parameter
        """

        param = true_param + np.random.normal(loc=error_mean, scale=error_stddev)

        return param
    
    def _get_fov(self, f_len:float, img_height:int, pixel_height:float)->float:

        H = img_height*pixel_height
        fov = 2* np.arctan(H/(2*f_len))

        return np.rad2deg(fov)
