import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append(sys.path[0] + '/..')
from dataclasses import dataclass

from Parameter import Parameter

import constants as c
from json import load as jsonload


@dataclass
class Material:

    alpha:Parameter=None
    emi:Parameter=None
    area:Parameter=None
    
    name:str="GENERIC"
    jsonFp:str=c.CUBESAT1U

    def __repr__(self)->str:
        name = "MATERIAL: {}".format(self.name)        
        return name

    def __post_init__(self)->None:
        self.alpha = self.__set_parameter(self.alpha, "ABSORPTIVITY")
        self.emi = self.__set_parameter(self.emi, "EMISSIVITY")
        self.area = Parameter(c.C1U, 0, 0, name="1U_SIDE")

        self.faceVector:np.ndarray = None
        return

    def randomize(self)->None:
        self.alpha.modulate()
        self.emi.modulate()
        self.area.modulate()
        return

    def __set_parameter(self, param:Parameter, name:str)->Parameter:
        if param is not None:
            return param
        
        with open(self.jsonFp) as fp_open:
            file = jsonload(fp_open)
        
        ideal = file[name+"_IDEAL"]
        stddev = file[name+"_STDDEV"]/3
        mean = file[name+"_MEAN"]

        return Parameter(ideal,stddev, mean, name=name, units='-')
    
class SatNode:
    def __init__(self, faceA:Material, faceB:Material, faceC:Material, heatCap:float=961)->None:
        self.faceA = faceA
        self.faceB = faceB
        self.faceC = faceC

        self.heatCap = heatCap
        self.effArea = self.__calc_eff_area()

        return

    def randomize(self)->None:
        self.faceA.randomize()
        self.faceB.randomize()
        self.faceC.randomize()
        return

    def calc_all_Q(self, state:np.ndarray, solarVec:np.ndarray)->np.ndarray[float]:
        
        attitude = self.__get_attitude(state)

        Qsol = self.__calc_Q_solar(solarVec, attitude)
        Qalb = self.__calc_Q_alb()
        Qir = self.__calc_Q_ir(state[:3])

        return np.array([Qsol, Qalb, Qir])

    def __calc_eff_area(self)->float:
        a = 0
        for face in [self.faceA, self.faceB, self.faceC]:
            a += face.area.value * face.alpha.value * 2
        return a

    def __calc_Q_solar(self, solarUnitVec:np.ndarray, attitude:np.ndarray)->float:

        Qsol = 0

        for unit, face in zip(attitude, [self.faceA, self.faceB, self.faceC]):
            cosb = np.abs(np.dot(unit, solarUnitVec))
            Qsol += face.alpha.value * face.area.value * cosb * c.SOLARFLUX

        return Qsol

    def __calc_Q_alb(self)->float:

        #TODO: implement view factor calculations instead of average view factor
        F = 0.4
        gamma = 0.273

        Qalb = 0
        for face in [self.faceA, self.faceB, self.faceC]:
            Qalb += 2 * face.alpha.value * face.area.value * gamma * c.EARTHFLUX * F

        return Qalb

    def __calc_Q_ir(self, position:np.ndarray)->float:

        h = np.linalg.norm(position)/c.EARTHRAD
        
        Fa = 1/(h**2)
        Fb = -np.sqrt(h**2-1)/(np.pi*h**2) + 1/np.pi * np.arctan(1/(np.sqrt(h**2-1)))

        QirA = 2 * self.faceA.emi.value * self.faceA.area.value * c.EARTHFLUX * Fa
        QirB = 2 * self.faceB.emi.value * self.faceB.area.value * c.EARTHFLUX * Fb
        QirC = 2 * self.faceC.emi.value * self.faceC.area.value * c.EARTHFLUX * Fb

        return QirA + QirB + QirC

    def __get_attitude(self, state:np.ndarray)->tuple[np.ndarray]:
        position = state[:3]/np.linalg.norm(state[:3])
        velocity = state[3:]/np.linalg.norm(state[3:])

        z = -1*position/np.linalg.norm(position)
        
        h = np.cross(position, velocity)
        x = h/np.linalg.norm(h)

        y = np.cross(z, x)
        return (x, y, z)

if __name__ == '__main__':
    al = Material()
    sat = SatNode(al, al, al)