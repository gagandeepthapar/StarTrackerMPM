import numpy as np
import pandas as pd

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

from dataclasses import dataclass
from json import load as jsonload

from alive_progress import alive_bar
from Parameter import Parameter

@dataclass
class StateVector:
    rx:float
    ry:float
    rz:float
    vx:float
    vy:float
    vz:float
    a:float=None
    mu:float=398600

    def __post_init__(self)->None:
        self.position = np.array([self.rx, self.ry, self.rz]).reshape((1,3))[0]
        self.velocity = np.array([self.vx, self.vy, self.vz]).reshape((1,3))[0]
        self.state = np.array([*self.position, *self.velocity])
        if self.a is None:
            self.a = self.__calc_semimajor()
        return

    def __repr__(self)->str:
        name = "State Vector:"\
                "\n\tRX:\t{}\t[km]"\
                "\n\tRY:\t{}\t[km]"\
                "\n\tRZ:\t{}\t[km]"\
                "\n\tVX:\t{}\t[km/s]"\
                "\n\tVY:\t{}\t[km/s]"\
                "\n\tVZ:\t{}\t[km/s]".format(self.rx, self.ry, self.rz, self.vx, self.vy, self.vz)
        return name
    
    def __calc_semimajor(self)->float:
        R = self.position
        V = self.velocity

        print(R)
        print(V)
        
        r = np.linalg.norm(R)
        v = np.linalg.norm(V)
        v_r = np.dot(R,V)/r

        h_bar = np.cross(R,V)
        h = np.linalg.norm(h_bar)

        ecc_bar = 1/self.mu * ((v**2 - self.mu/r)*R - r*v_r*V)
        ecc = np.linalg.norm(ecc_bar)
        print(ecc)

        rp = h**2/self.mu * 1/(1 + ecc)
        ra = h**2/self.mu * 1/(1 - ecc)

        return 0.5*(ra + rp)

@dataclass
class TLE:

    inc:float
    raan:float
    ecc:float
    arg:float
    mean_anomaly:float
    mean_motion:float

    name:str=None
    id:str=None

    mu:float=c.EARTHMU

    def __post_init__(self)->None:
        self.a = self.__get_semimajor()
        return
    
    def __repr__(self)->str:
        name = f"TLE {self.name}:"\
                f"\n\tinc:\t{self.inc}\t[deg]"\
                f"\n\tecc:\t{self.ecc}\t[~]"\
                f"\n\traan:\t{self.raan}\t[deg]"\
                f"\n\targ:\t{self.arg}\t[deg]"
        return name
    
    def from_dict(info:dict)->None:

        inc = info['INCLINATION']
        raan = info['RA_OF_ASC_NODE']
        ecc = info['ECCENTRICITY']
        arg = info['ARG_OF_PERICENTER']
        mean_anom = info['MEAN_ANOMALY']
        mean_motion = info['MEAN_MOTION']
        name = info['OBJECT_NAME']
        id = info['OBJECT_ID']

        return TLE(inc, raan, ecc, arg, mean_anom, mean_motion, name, id)
    
    def tle_to_state(self)->StateVector:
        Me = np.deg2rad(self.mean_anomaly)

        if Me < np.pi:
            E_0 = Me - self.ecc
        else:
            E_0 = Me + self.ecc
        
        f = lambda E: Me - E + self.ecc * np.sin(E)
        fp = lambda E: -1 + self.ecc * np.sin(E)

        E_1 = E_0 - (f(E_0)/fp(E_0))
        err = np.abs(E_1 - E_0)

        while err > 1e-8:
            E_0 = E_1
            E_1 = E_0 - (f(E_0)/fp(E_0))
            err = np.abs(E_1 - E_0)
        
        TA = 2*c.atand((np.sqrt((1+self.ecc)/(1-self.ecc)) * np.tan(E_1/2)))
        theta = np.mod(TA, 360)

        T = 1/self.mean_motion * 24 * 3600
        a = (T*np.sqrt(self.mu)/(2*np.pi))**(2/3)
        h = np.sqrt(a*self.mu*(1-self.ecc**2))

        return self.coes_to_state(h=h, ecc=self.ecc, inc=self.inc, raan=self.raan, arg=self.arg, theta=theta,a=a, mu=self.mu)

    def coes_to_state(self, ecc:float, inc:float, raan:float, arg:float, theta:float,*, h:float=None,a:float=None, mu:float=398600)->StateVector:

        if h is None:
            h = np.sqrt(a * mu * (1 - ecc**2))

        if a is None:
            rp = h**2/self.mu * 1/(1 + ecc)
            ra = h**2/self.mu * 1/(1 - ecc)
            a = 0.5*(ra+rp)   

        peri_r = h**2 / mu * (1/(1 + ecc*c.cosd(theta))) * np.array([[c.cosd(theta)],[c.sind(theta)], [0]])
        peri_v = mu / h * np.array([[-c.sind(theta)], [ecc + c.cosd(theta)], [0]])

        Q_bar = self.__R3(arg) @ self.__R1(inc) @ self.__R3(raan)

        r = np.transpose(Q_bar) @ peri_r
        v = np.transpose(Q_bar) @ peri_v

        return StateVector(*r, *v, a=a)

    def __get_semimajor(self)->float:
        T = 1/self.mean_motion * 24 * 3600
        a = (T*np.sqrt(self.mu)/(2*np.pi))**(2/3)
        return a

    def __R1(self, theta:float)->np.ndarray:
        R = np.array([[1, 0, 0], [0, c.cosd(theta), c.sind(theta)], [0, -c.sind(theta), c.cosd(theta)]])
        return R
    
    def __R3(self, theta:float)->np.ndarray:
        R = np.array([[c.cosd(theta), c.sind(theta), 0], [-c.sind(theta), c.cosd(theta), 0], [0, 0, 1]])
        return R

class Orbit:
    def __init__(self, semiMajorParam:Parameter=None,
                       eccParam:Parameter=None,
                       incParam:Parameter=None, 
                       raanParam:Parameter=None, 
                       argParam:Parameter=None,
                       *, 
                       tleDF:pd.DataFrame=None,
                       orbitName:str=None, 
                       sat_TLE_fp:str=c.CUBESATS)->None:

        if tleDF is None:
            self.TLES = self.__read_TLE_to_df(sat_TLE_fp)
        else:
            self.TLES = tleDF

        print('done')

        self.semi = self.__set_parameter(semiMajorParam, "SEMI")
        self.ecc = self.__set_parameter(eccParam, "ECC")
        self.inc = self.__set_parameter(incParam, "INC")
        self.raan = self.__set_parameter(raanParam, "RAAN")
        self.arg = self.__set_parameter(argParam, "ARG")

        self.name = orbitName

        self.semi.modulate()
        self.ecc.modulate()
        self.inc.modulate()
        self.raan.modulate()
        self.arg.modulate()

        return

    def __repr__(self)->str:
        name = "Orbit: {}"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}\n".format(self.name,
                            repr(self.semi),
                            repr(self.ecc),
                            repr(self.inc),
                            repr(self.raan),
                            repr(self.arg))

        return name

    def __read_TLE_to_df(self, fp:str)->pd.DataFrame:
        with open(fp) as fp_open:
            sat_tles = jsonload(fp_open)
        
        filelen = len(sat_tles)

        ecc = np.empty(filelen, dtype=float)
        inc = np.empty(filelen, dtype=float)
        raan = np.empty(filelen, dtype=float)
        arg = np.empty(filelen, dtype=float)
        semi = np.empty(filelen, dtype=float)
        mean_mot = np.empty(filelen, dtype=float)

        with alive_bar(filelen, title='Reading in TLE Data') as bar:
            for i in range(filelen):
                tle = TLE.from_dict(sat_tles[i])
                
                ecc[i] = tle.ecc
                inc[i] = tle.inc 
                raan[i] = tle.raan
                arg[i] = tle.arg
                semi[i] = tle.a
                mean_mot[i] = tle.mean_motion

                bar()

        tledict = {
                    "ECC":ecc,
                    "INC":inc,
                    "RAAN":raan,
                    "ARG":arg,
                    "SEMI":semi,
                    "MEAN_MOTION":mean_mot
                    }
                
        return pd.DataFrame(tledict)
    
    def __set_parameter(self, param:Parameter, name:str)->Parameter:
        if param is not None:
            return param 

        mean = np.mean(self.TLES[name])
        std = np.std(self.TLES[name])

        return Parameter(0, std, mean, name)

    def __solar_position(self, position:StateVector)->np.ndarray:
        # TODO: implement (again)
        return

