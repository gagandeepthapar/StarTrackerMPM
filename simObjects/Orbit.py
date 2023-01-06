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
from Parameter import Parameter, UniformParameter
from MaterialProperty import Material
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

@dataclass
class StateVector:
    rx:float
    ry:float
    rz:float
    vx:float
    vy:float
    vz:float
    a:float = None
    mu:int=c.EARTHMU

    def __post_init__(self)->None:
        self.position = np.array([self.rx, self.ry, self.rz]).reshape((1,3))[0]
        self.velocity = np.array([self.vx, self.vy, self.vz]).reshape((1,3))[0]
        self.state = np.array([*self.position, *self.velocity])
        self.a = self.__calc_semimajor()
        self.T = self.__calc_period()
        return

    def __repr__(self)->str:
        return f'State Vector: {np.round(self.a,3)}km @ {np.round(self.T/60,3)}min'
    
    def __calc_semimajor(self)->float:
        R = self.position
        V = self.velocity

        r = np.linalg.norm(R)
        v = np.linalg.norm(V)
        v_r = np.dot(R,V)/r

        h_bar = np.cross(R,V)
        h = np.linalg.norm(h_bar)

        ecc_bar = 1/self.mu * ((v**2 - self.mu/r)*R - r*v_r*V)
        ecc = np.linalg.norm(ecc_bar)

        rp = h**2/self.mu * 1/(1 + ecc)
        ra = h**2/self.mu * 1/(1 - ecc)

        return 0.5*(ra + rp)

    def __calc_period(self)->float:
        return self.a**(3/2) * (2*np.pi) / np.sqrt(self.mu)

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
        return f"TLE: {self.name}" 
    
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

    def coes_to_state(self, ecc:float, inc:float, raan:float, arg:float, theta:float, *, h:float=None, a:float=None, mu:float=398600)->StateVector:

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

class OrbitData:

    mu = c.EARTHMU
    
    def __init__(self,*, tlePD:pd.DataFrame=None, tleJSONFP:str=c.CUBESATS)->None:

        if tlePD is None:
            tlePD = self.__read_TLE_to_df(tleJSONFP)
        
        self.TLES = tlePD

        self.params = self.__create_params()

        return
    
    def __repr__(self)->str:
        name = 'ORBIT DATA:\n{}'.format(self.TLES)
        return name

    def add_TLE(self, tle:TLE)->None:
        tledict = {
                    "ECC":[tle.ecc],
                    "INC":[tle.inc],
                    "RAAN":[tle.raan],
                    "ARG":[tle.arg],
                    "SEMI":[tle.a],
                    "MEAN_MOTION":[tle.mean_motion],
                    }

        self.TLES = self.TLES.append(pd.DataFrame(tledict), ignore_index=True)
        self.params = self.__create_params()
        return

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
        tleList = np.empty(filelen, dtype=TLE)

        with alive_bar(filelen, title='Reading in TLE Data') as bar:
            for i in range(filelen):
                tle = TLE.from_dict(sat_tles[i])
                
                ecc[i] = tle.ecc
                inc[i] = tle.inc 
                raan[i] = tle.raan
                arg[i] = tle.arg
                semi[i] = tle.a
                mean_mot[i] = tle.mean_motion
                tleList[i] = tle

                bar()

        tledict = {
                    "ECC":ecc,
                    "INC":inc,
                    "RAAN":raan,
                    "ARG":arg,
                    "SEMI":semi,
                    "MEAN_MOTION":mean_mot,
                    }
        
        df = pd.DataFrame(tledict)

        return df
    
    def __create_params(self)->dict:

        param_dict = {}
        for param in self.TLES.columns:
            mean = np.mean(self.TLES[param])
            std = np.mean(self.TLES[param])
            param_dict[param] = Parameter(mean, std, mean, param)

        # skew = self.TLES[name].skew()
        # if np.abs(skew) > 1:
        #     norm_data = self.__normalize_data(name)
        #     name = 'TRANSFORMED_'+name
        #     self.TLES[name] = norm_data

        return param_dict

    def __get_random_value(self, param:Parameter)->float:
        
        val = param.modulate()

        if "TRANSFORMED_" not in param.name:
            return val
        
        paramname = param.name[12:]
        skew = self.TLES[paramname].skew()

        if skew > 0:
            return np.e**val    # undo left skew tfr

        return np.real(val**(1/5))   # undo right skew tfr

    def __normalize_data(self, name:str)->pd.DataFrame:
        skew = self.TLES[name].skew()
        if skew > 0:
            return self.TLES[name].apply(lambda x: np.log(x))   # handle left skew
        return self.TLES[name].apply(lambda x: x**5)    # handle right skew

class Orbit:

    mu:int = c.EARTHMU

    def __init__(self, incParam:Parameter=None,
                       eccParam:Parameter=None,
                       semiParam:Parameter=None,
                       *,
                       orbitData:OrbitData=None)->None:

        self.__orbitData = orbitData

        self.inc:Parameter = self.__set_parameter(incParam, "INC")
        self.ecc:Parameter = self.__set_parameter(eccParam, "ECC")
        self.semi:Parameter = self.__set_parameter(semiParam, "SEMI")

        self.arg = UniformParameter(0, 360, 'ARG', c.DEG)
        self.raan = UniformParameter(0, 360, 'RAAN', c.DEG)
        self.theta = UniformParameter(0, 360, 'TA', c.DEG)
        self.jd = UniformParameter(0, 365.25, 'JULIAN_DATE', 'days')

        self.randomize()
        return
    
    def randomize(self)->None:

        self.inc.modulate()
        self.ecc.modulate()
        self.semi.modulate()
        
        self.arg.modulate()
        self.raan.modulate()
        self.theta.modulate()
        self.jd.modulate()

        self.T = self.__calc_period()

        self.state = self.__calc_state()
        self.path = self.__propagate_orbit()

        return

    def calc_temperature(self)->float:

        return

    def __calc_state(self)->StateVector:
        
        inc = self.inc.value
        ecc = self.ecc.value
        theta = self.theta.value
        arg = self.arg.value
        raan = self.raan.value
        a = self.semi.value
        
        __R1 = lambda theta: np.array([[1, 0, 0], [0, c.cosd(theta), c.sind(theta)], [0, -c.sind(theta), c.cosd(theta)]])
        __R3 = lambda theta: np.array([[c.cosd(theta), c.sind(theta), 0], [-c.sind(theta), c.cosd(theta), 0], [0, 0, 1]]) 

        h = np.sqrt(a * self.mu * (1 - ecc**2))

        peri_r = h**2 / self.mu * (1/(1 + ecc*c.cosd(theta))) * np.array([[c.cosd(theta)],[c.sind(theta)], [0]])
        peri_v = self.mu / h * np.array([[-c.sind(theta)], [ecc + c.cosd(theta)], [0]])

        Q_bar = __R3(arg) @ __R1(inc) @ __R3(raan)

        r = np.transpose(Q_bar) @ peri_r
        v = np.transpose(Q_bar) @ peri_v

        return StateVector(*r, *v, a=a)

    def __set_parameter(self, param:Parameter, name:str)->Parameter:
        if param is not None:
            return param
        
        if self.__orbitData is None:
            print('{}Generating Default Orbit Data{}'.format(c.YELLOW, c.DEFAULT))
            self.__orbitData = OrbitData()
        
        print('{}Generating Parameter: {}{}'.format(c.YELLOW,c.DEFAULT,name))
        return self.__orbitData.params[name]
        
    def __calc_period(self)->float:
        a = self.semi.value
        T = a**(3/2) * (2*np.pi) / np.sqrt(self.mu)
        return T

    def __propagate_orbit(self, tspan:tuple[float]=None, tstep=1)->pd.DataFrame:
        if tspan is None:
            tspan = (0, int(self.T)+1)
        
        t_eval = np.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0])/tstep)+1)
        
        sol = solve_ivp(self.__two_body, tspan, self.state.state, t_eval=t_eval)
        
        traj = pd.DataFrame({
                                "TIME": sol['t'],
                                "RX": sol['y'][0],
                                "RY": sol['y'][1],
                                "RZ": sol['y'][2],
                                "VX": sol['y'][3],
                                "VY": sol['y'][4],
                                "VZ": sol['y'][5],
                            })

        return traj

    def __two_body(self, t:float, state:np.ndarray, mu=None)->np.ndarray:
        if mu is None:
            mu = self.mu

        r = state[:3]
        R = np.linalg.norm(r)
        v:np.ndarray = state[3:]

        a = -mu * r / (R**3)
        return np.append(v, a)

    def __get_Q_direct(self, position:StateVector, julianDate:float)->float:
        """Calculate Heat Flux from Sun (Direct)

        Args:
            position (StateVector): Position of satellite
            julianDate (float): julianDate

        Returns:
            float: Heat Flux from Sun (direct)
        """
        
        # If not in direct LOS of Sun, Q_direct = 0

        # TODO: implement from Garzon et al.

        return
    
    def __get_Q_albedo(self)->float:
        # TODO: implement from Garzon et al.
        return
    
    def __get_Q_ir(self)->float:
        # TODO: implement from Garzon et al.
        return

    def __solar_position(self)->np.ndarray:
        """Adapted from "Orbital Mechanics for Engineering Students", Curtis et al.

        Calculate position of the Sun relative to Earth based on julian date

        Args:
            julianDate (float): julian date for day in question

        Returns:
            np.ndarray: non-normalized position of Sun wrt Earth
        """

        jd = self.jd.value
        
        #...Julian days since J2000:
        n     = jd - 2451545

        #...Mean anomaly (deg{:
        M     = 357.528 + 0.9856003*n
        M     = np.mod(M,360)

        #...Mean longitude (deg):
        L     = 280.460 + 0.98564736*n
        L     = np.mod(L,360)

        #...Apparent ecliptic longitude (deg):
        lamda = L + 1.915*c.sind(M) + 0.020*c.sind(2*M)
        lamda = np.mod(lamda,360)

        #...Obliquity of the ecliptic (deg):
        eps   = 23.439 - 0.0000004*n

        #...Unit vector from earth to sun:
        u     = np.array([c.cosd(lamda), c.sind(lamda)*c.cosd(eps), c.sind(lamda)*c.sind(eps)])

        #...Distance from earth to sun (km):
        rS    = (1.00014 - 0.01671*c.cosd(M) - 0.000140*c.cosd(2*M))*c.AU

        #...Geocentric position vector (km):
        r_S   = rS*u

        return r_S

    def __solar_line_of_sight(self)->bool:

        r_earth_sun = self.__solar_position()
        r_earth_sc = self.state.position

        theta = np.arccos(np.dot(r_earth_sun, r_earth_sc)/(np.linalg.norm(r_earth_sun)*np.linalg.norm(r_earth_sc)))
        thetaA = np.arccos(c.EARTHRAD/(np.linalg.norm(r_earth_sc)))
        thetaB = np.arccos(c.EARTHRAD/(np.linalg.norm(r_earth_sun)))

        return theta > (thetaA + thetaB)

if __name__ == '__main__':
    o = Orbit()