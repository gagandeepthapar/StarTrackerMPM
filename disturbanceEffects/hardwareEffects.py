import numpy as np

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

import json

class Parameter:

    def __init__(self, ideal:float, stddev:float, mean:float=0, name:str=None)->None:

        self.ideal = ideal
        self.name = name

        self._err_mean = mean
        self._err_stddev = stddev
    
        self.range = self._err_mean + (3*self._err_stddev)
        self.minRange = self.ideal - self.range
        self.maxRange = self.ideal + self.range

        self.modulate()

        return
    
    def __repr__(self)->str:
        pname = '{}: {} [{}(\u03BC) +/- {}(3\u03C3)]'.format(self.name, np.round(self.value,3), self.ideal, 3*self._err_stddev)
        return pname 

    def modulate(self)->float:
        self.value = self.ideal + (np.random.normal(loc=self._err_mean, scale=self._err_stddev))
        return self.value
    
    def reset(self)->float:
        self.value = self.ideal
        return self.value

    def get_ideal_param(self)->None:
        return Parameter(ideal=self.ideal, stddev=0, mean=0, name="IDEAL_"+self.name)

    def _init_from_json(fp:str,name:str)->None:

        camDict = json.load(open(fp))

        ideal = camDict[name+"_IDEAL"]
        mean = camDict[name+"_MEAN"]
        stddev = camDict[name+"_STDDEV"]/3

        return Parameter(ideal=ideal, stddev=stddev, mean=mean, name=name)

class Camera:

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
        self.f_len = self._set_parameter(focal_length, "FOCAL_LENGTH")
        self.array_tilt = self._set_parameter(array_tilt, "FOCAL_ARRAY_INCLINATION")
        self.distortion = self._set_parameter(distortion, "DISTORTION")

        self._fov = self._set_img_fov()
        self._num_stars = self._set_est_num_stars()

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

    def _set_img_fov(self)->float:

        cam = json.load(open(self.camJSON))

        self._imgX = cam['IMAGE_WIDTH']
        self._imgY = cam['IMAGE_HEIGHT']
        self._pixelX = cam['PIXEL_WIDTH']
        self._pixelY = cam['PIXEL_HEIGHT']

        wd = self._imgX * self._pixelX
        ht = self._imgY * self._pixelY
        x = np.sqrt(wd**2 + ht**2)

        fov = np.rad2deg(2*np.arctan(x/(2*self.f_len.value)))

        return fov

    def _set_est_num_stars(self)->float:
        return 4

    def modulate_centroid(self)->float:
        return self.ctr_acc.modulate()
    
    def modulate_principal_point_accuracy(self)->float:
        return self.ppt_acc.modulate()
    
    def modulate_focal_length(self)->float:
        return self.f_len.modulate()
    
    def modulate_array_tilt(self)->float:
        return self.array_tilt.modulate()
    
    def modulate_distortion(self)->float:
        return self.distortion.modulate()

    def reset_params(self)->None:
        self.f_len.reset()
        self.ctr_acc.reset()
        self.ppt_acc.reset()
        self.array_tilt.reset()
        self.distortion.reset()
        return
    


