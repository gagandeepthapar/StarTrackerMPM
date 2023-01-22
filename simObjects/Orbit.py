import sys

import numpy as np
import pandas as pd

sys.path.append(sys.path[0] + '/..')
from dataclasses import dataclass
from json import load as jsonload

import matplotlib.pyplot as plt
from alive_progress import alive_bar
from MaterialProperty import Material
from Parameter import Parameter, UniformParameter
from scipy.integrate import solve_ivp
from scipy.fft import dst, dct

import constants as c

import time
from MaterialProperty import Material, SatNode


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

    def __get_semimajor(self)->float:
        T = 1/self.mean_motion * 24 * 3600
        a = (T*np.sqrt(self.mu)/(2*np.pi))**(2/3)
        return a

class OrbitData:

    mu = c.EARTHMU
    
    def __init__(self,*, tlePD:pd.DataFrame=None, tleJSONFP:str=c.CUBESATS)->None:

        if tlePD is None:
            tlePD = self.__read_TLE_to_df(tleJSONFP)
        
        self.TLES = tlePD

        self.params:dict = self.__create_params()
        self.params["TEMP"] = Parameter(20, 5, 0, name="TEMP", units='C')

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
                    "MEAN_ANOMALY":[tle.mean_anomaly],
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
        mean_anom = np.empty(filelen, dtype=float)

        with alive_bar(filelen, title='Reading in TLE Data') as bar:
            for i in range(filelen):
                tle = TLE.from_dict(sat_tles[i])
                
                ecc[i] = tle.ecc
                inc[i] = tle.inc 
                raan[i] = tle.raan
                arg[i] = tle.arg
                semi[i] = tle.a
                mean_mot[i] = tle.mean_motion
                mean_anom[i] = tle.mean_anomaly

                bar()

        tledict = {
                    "ECC":ecc,
                    "INC":inc,
                    "RAAN":raan,
                    "ARG":arg,
                    "SEMI":semi,
                    "MEAN_ANOMALY":mean_anom,
                    "MEAN_MOTION":mean_mot,
                    }
        
        df = pd.DataFrame(tledict)

        return df

    def __create_params(self)->dict:
        param_dict = {}

        for param in self.TLES.columns:

            data = self.TLES[param]

            if np.abs(data.skew()) > 1:
                param_dict[param] = self.__normalize_data(param)
            
            else:
                mean = np.mean(self.TLES[param])
                std = np.std(self.TLES[param])
                param_dict[param] = Parameter(mean, std, 0, name=param)

        return param_dict

    def __normalize_data(self, name:str)->pd.DataFrame:
        skew = self.TLES[name].skew()
        if skew > 0:
            data = self.TLES[name].apply(lambda x: np.log(x))   # handle left skew
            retVal = lambda x: np.e**x
        else:
            data = self.TLES[name].apply(lambda x: x**5)    # handle right skew
            retVal = lambda x: np.real(np.abs(x)**(1/5))
        
        mean = np.mean(data)
        std = np.std(data)

        return Parameter(mean, std, 0, name=name, retVal=retVal)

    def gen_heat_flux(self, satellite:SatNode)->None:
        iters = len(self.path.index)

        Qsol = np.empty(iters, dtype=float)
        Qalb = np.empty(iters, dtype=float)
        Qir = np.empty(iters, dtype=float)
        Qtot = np.empty(iters, dtype=float)

        ti = time.perf_counter()
        for i in range(iters):
            row = self.path.iloc[i]
            satState = row[['RX', 'RY', 'RZ', 'VX', 'VY', 'VZ']].to_numpy()
            
            sunState = self.__solar_position(self.jd.value + row['TIME']/(24*3600))

            sunUnitState = sunState/np.linalg.norm(sunState)

            Qs = satellite.calc_all_Q(satState, sunUnitState)

            if not self.__solar_line_of_sight(self.jd.value + row['TIME']/(24*3600), satState[:3]):
                Qs[0] = 0

            Qsol[i] = Qs[0]
            Qalb[i] = Qs[1]
            Qir[i] = Qs[2]
            Qtot[i] = np.sum(Qs)
        
        print('loop gen: {}'.format(time.perf_counter() - ti))

        s = time.perf_counter()
        self.heatFlux['Q_SOLAR'] = Qsol
        self.heatFlux['Q_ALBEDO'] = Qalb
        self.heatFlux['Q_IR'] = Qir
        self.heatFlux['Q_TOTAL'] = Qtot
        print('calc gen: {}'.format(time.perf_counter() - s))
        return

    def calc_temperature(self, satellite:SatNode, initial_temp:float=np.random.normal(20, 5))->None:
        
        tspan = (self.heatFlux['TIME'].iloc[0], self.heatFlux['TIME'].iloc[-1])
        teval = self.heatFlux['TIME'].to_numpy()

        sol = solve_ivp(self.__sat_temp_diffeq, tspan, [273.15+initial_temp], t_eval=teval, rtol=1e-8, atol=1e-8, args=(satellite.effArea, satellite.heatCap, ))
        self.heatFlux['SAT_TEMP'] = sol['y'][0] - 273.15

        return

    def __sat_temp_diffeq(self, t:float, state:np.ndarray[float], effArea:float, heatCapacity:float)->float:

        sigma = c.STEFBOLTZ
        
        # find nearest neighbors in the set
        tminF = lambda t: np.abs(o.heatFlux['TIME'] - t).argmin()
        tminIdx = tminF(t)
        tmaxIdx = tminIdx + 1

        if tminIdx >= len(self.heatFlux.index):
            tminIdx = int(self.heatFlux['TIME'].iloc[-2])
            tmaxIdx = tminIdx + 1

        if tmaxIdx >= len(self.heatFlux.index):
            tmaxIdx = tminIdx
            tminIdx = tminIdx-1

        tmin = self.heatFlux['TIME'].iloc[tminIdx]
        tmax = self.heatFlux['TIME'].iloc[tmaxIdx]
        Qmin = self.heatFlux['Q_TOTAL'].iloc[tminIdx]
        Qmax = self.heatFlux['Q_TOTAL'].iloc[tmaxIdx]

        realQ = Qmin + (t - tmin)*(Qmax - Qmin)/(tmax-tmin)

        dt = (-sigma * effArea * state**4 + realQ)/heatCapacity
        return dt 

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
        
    def __calc_period(self)->float:
        a = self.semi.value
        T = a**(3/2) * (2*np.pi) / np.sqrt(self.mu)
        return T

    def __propagate_orbit(self, tspan:tuple[float]=None, tstep:float=1)->pd.DataFrame:
        if tspan is None:
            tspan = (0, int(self.T)+1)
        
        t_eval = np.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0])/tstep)+1)
        
        sol = solve_ivp(self.__two_body, tspan, self.state.state, t_eval=t_eval, rtol=1e-8, atol=1e-8)
        
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
        v = state[3:]

        a = -mu * r / (R**3)
        return np.append(v, a)

    def __solar_position(self, julianDate:float=None)->np.ndarray:
        """Adapted from "Orbital Mechanics for Engineering Students", Curtis et al.

        Calculate position of the Sun relative to Earth based on julian date

        Args:
            julianDate (float): julian date for day in question

        Returns:
            np.ndarray: non-normalized position of Sun wrt Earth
        """

        jd = julianDate
        if julianDate is None:
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

    def __solar_line_of_sight(self, julianDate:float=None, position:np.ndarray=None)->bool:

        r_earth_sun = self.__solar_position(julianDate)

        r_earth_sc = position
        if position is None:
            r_earth_sc = self.state.position

        theta = np.arccos(np.dot(r_earth_sun, r_earth_sc)/(np.linalg.norm(r_earth_sun)*np.linalg.norm(r_earth_sc)))
        thetaA = np.arccos(c.EARTHRAD/(np.linalg.norm(r_earth_sun)))
        thetaB = np.arccos(c.EARTHRAD/(np.linalg.norm(r_earth_sc)))

        return theta <= (thetaA + thetaB)

class Orbit:

    mu:int = c.EARTHMU
    rad:int = c.EARTHRAD

    def __init__(self, incParam:Parameter=None,
                       eccParam:Parameter=None,
                       semiParam:Parameter=None,
                       tempParam:Parameter=None,
                       *,
                       name:str = None,
                       orbitData:OrbitData=None)->None:

        self.orbitData:OrbitData = orbitData

        self.inc:Parameter = self.__set_parameter(incParam, "INC")
        self.ecc:Parameter = self.__set_parameter(eccParam, "ECC")
        self.semi:Parameter = self.__set_parameter(semiParam, "SEMI")
        self.temp:Parameter = self.__set_parameter(semiParam, "TEMP")

        self.arg = UniformParameter(0, 360, 'ARG', c.DEG)
        self.raan = UniformParameter(0, 360, 'RAAN', c.DEG)
        self.theta = UniformParameter(0, 360, 'THETA', c.DEG)
        self.jd = UniformParameter(c.J2000, c.J2000+365.25, 'JULIAN_DATE', 'days')

        self.all_params = [self.inc, self.ecc, self.semi, self.arg, self.raan, self.theta, self.jd]

        self.name = name
        self.randomize()

        return
    
    def __repr__(self)->str:
        name = 'Orbit: {}'\
                '\n\t{}'\
                '\n\t{}'\
                '\n\t{}'.format(self.name,self.inc, self.ecc, self.semi)
        return name

    def randomize(self, params:list=None, num:int=1)->None:

        if params is None:
            params = self.all_params

        f = {}
        for param in self.all_params:
            f[param.name] = np.zeros((num))

        for i in range(num):
            for param in self.all_params:
                if param in params:
                    param.modulate()
            
                f[param.name][i] = param.value

        data = pd.DataFrame(f)        

        return data
  
    def __set_parameter(self, param:Parameter, name:str)->Parameter:
        if param is not None:
            return param
        
        if self.orbitData is None:
            print('{}Generating Default Orbit Data{}'.format(c.YELLOW, c.DEFAULT))
            self.orbitData = OrbitData()
        
        print('{}Generating Parameter: {}{}'.format(c.YELLOW,c.DEFAULT,name))
        return self.orbitData.params[name]

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
        
    def __calc_period(self)->float:
        a = self.semi.value
        T = a**(3/2) * (2*np.pi) / np.sqrt(self.mu)
        return T

    def __propagate_orbit(self)->pd.DataFrame:
        
        tspan = (0, int(self.T)+1)
        tstep = 1
        
        t_eval = np.linspace(tspan[0], tspan[1], int((tspan[1] - tspan[0])/tstep)+1)
        
        sol = solve_ivp(self.__two_body, tspan, self.state.state, t_eval=t_eval, rtol=1e-8, atol=1e-8)
        
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
        v = state[3:]

        a = -mu * r / (R**3)
        return np.append(v, a)

    def solar_position(self, julianDate:float=None)->np.ndarray:
        """Adapted from "Orbital Mechanics for Engineering Students", Curtis et al.

        Calculate position of the Sun relative to Earth based on julian date

        Args:
            julianDate (float): julian date for day in question

        Returns:
            np.ndarray: non-normalized position of Sun wrt Earth
        """

        jd = julianDate
        if julianDate is None:
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

    def solar_line_of_sight(self, julianDate:float=None, position:np.ndarray=None)->bool:

        r_earth_sun = self.__solar_position(julianDate)

        r_earth_sc = position
        if position is None:
            r_earth_sc = self.state.position

        theta = np.arccos(np.dot(r_earth_sun, r_earth_sc)/(np.linalg.norm(r_earth_sun)*np.linalg.norm(r_earth_sc)))
        thetaA = np.arccos(c.EARTHRAD/(np.linalg.norm(r_earth_sun)))
        thetaB = np.arccos(c.EARTHRAD/(np.linalg.norm(r_earth_sc)))

        return theta <= (thetaA + thetaB)


""" """

""" """

# def calc_state(inc, ecc, theta, arg, raan, semi, mu:int=c.EARTHMU)->StateVector:
def calc_state(row, mu:int=c.EARTHMU):
    ecc = row['ECC']
    inc = row['INC']
    a = row['SEMI']
    raan = row['RAAN']
    theta = row['THETA']
    arg = row['ARG']
    
    __R1 = lambda theta: np.array([[1, 0, 0], [0, c.cosd(theta), c.sind(theta)], [0, -c.sind(theta), c.cosd(theta)]])
    __R3 = lambda theta: np.array([[c.cosd(theta), c.sind(theta), 0], [-c.sind(theta), c.cosd(theta), 0], [0, 0, 1]]) 

    h = np.sqrt(a * mu * (1 - ecc**2))

    peri_r = h**2 / mu * (1/(1 + ecc*c.cosd(theta))) * np.array([[c.cosd(theta)],[c.sind(theta)], [0]])

    Q_bar = __R3(arg) @ __R1(inc) @ __R3(raan)

    r = np.transpose(Q_bar) @ peri_r
    return r

def solar_line_of_sight(row)->bool:

    julianDate = row['JULIAN_DATE']
    position = row['POS']

    r_earth_sun = spos(julianDate)
    r_earth_sc = position

    theta = np.arccos(np.dot(r_earth_sun, r_earth_sc)/(np.linalg.norm(r_earth_sun)*np.linalg.norm(r_earth_sc)))
    thetaA = np.arccos(c.EARTHRAD/(np.linalg.norm(r_earth_sun)))
    thetaB = np.arccos(c.EARTHRAD/(np.linalg.norm(r_earth_sc)))

    return theta <= (thetaA + thetaB)

def spos(julianDate:float=None)->np.ndarray:
    """Adapted from "Orbital Mechanics for Engineering Students", Curtis et al.

    Calculate position of the Sun relative to Earth based on julian date

    Args:
        julianDate (float): julian date for day in question

    Returns:
        np.ndarray: non-normalized position of Sun wrt Earth
    """
    jd = julianDate
    
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

    return u

    #...Distance from earth to sun (km):
    rS    = (1.00014 - 0.01671*c.cosd(M) - 0.000140*c.cosd(2*M))*c.AU

    #...Geocentric position vector (km):
    r_S   = rS*u

    return r_S

def vec2radec(vec:np.ndarray)->np.ndarray:

    dec = np.arcsin(vec[2])
    ra = np.arcsin(vec[1]/np.cos(dec))

    return np.array([ra, dec])

def beta_ang(radec:np.ndarray, raan:float, inc:float)->float:
    ra = radec[0]
    dec = radec[1]

    bA = np.cos(dec)*np.sin(inc)*np.sin(raan - ra)
    bB = np.sin(dec)*np.cos(inc)
    beta = np.arcsin(bA + bB)

    return beta

def eclipse_time(beta, t, pos)->float:

    r = pos
    num = 1 - (c.EARTHRAD/r)**2
    den = np.cos(beta)

    delT = np.arccos(np.sqrt(num)/den)*t/np.pi

    return delT

def test_temp(row):

    T = row['PERIOD']
    eT = row['EC_TIME']

    tA = (T-eT)/2
    tB = tA + eT

    def __temp_diffeq(t, state):

        if t <= tA:
            Q = row['Q_TOT']
        
        elif t <= tB:
            Q = row['Q_IR'] + row['Q_ALB']
        
        else:
            Q = row['Q_TOT']
        
        return (-1*c.STEFBOLTZ*row['AREA']*row['EMI']*state**4 + Q)/961

    sol = solve_ivp(__temp_diffeq, (0, T), [273.15+row['INITIAL_TEMP']], t_eval=np.linspace(0, T, 100), rtol=1e-6, atol=1e-6)
    temps = sol['y'][0] - 273.15

    return temps
if __name__ == '__main__':
    N = 100_000    
    o = Orbit()
    helpers = pd.DataFrame()
    m = Material()
    F = 0.4
    gamma = 0.273

    df:pd.DataFrame = o.randomize(num=N)
    
    start = time.perf_counter()
    df['PERIOD'] = df['SEMI']**1.5 * 2*np.pi /np.sqrt(c.EARTHMU)

    df['INITIAL_TEMP'] = np.random.normal(20, 5, len(df.index))
    df['ALPHA'] = np.random.normal(0.5, 0.1, len(df.index))
    df['EMI'] = np.random.normal(0.5, 0.1, len(df.index))
    df['AREA'] = 6*np.random.normal(0.01, 0.0001, len(df.index))

    df['Q_ALB'] = df['ALPHA']*df['AREA']*gamma*c.EARTHFLUX*F
    helpers['POS'] = df['SEMI'] * (1-df['ECC']**2)/(1+df['ECC']*np.cos(np.deg2rad(df['THETA'])))
    helpers['Q_H'] = helpers['POS']/c.EARTHRAD
    helpers['Fa'] = 1/(helpers['Q_H']**2)
    helpers['Fb'] = -np.sqrt(helpers['Q_H']**2 - 1)/(np.pi*helpers['Q_H']**2) + 1/np.pi * np.arctan(1/(np.sqrt(helpers['Q_H']**2-1)))
    df['Q_IR'] = (1/3*df['EMI']*df['AREA']*c.EARTHFLUX*helpers['Fa']) \
                + (2/3*df['EMI']*df['AREA']*c.EARTHFLUX*helpers['Fb'])

    helpers['SVECX'], helpers['SVECY'], helpers['SVECZ'] = spos(df['JULIAN_DATE'].values)
    helpers['RA'], helpers['DEC'] = vec2radec([helpers['SVECX'], helpers['SVECY'], helpers['SVECZ']])
    
    helpers['BETA'] = beta_ang([helpers['RA'], helpers['DEC']], df['RAAN'], df['INC'])
    
    helpers['SEMI'] = df['SEMI']
    helpers['T'] = df['PERIOD']
    helpers['EC_TIME'] = 0
    helpers['EC_FLAG'] = np.abs(np.sin(helpers['BETA'])) < c.EARTHRAD/helpers['POS']
    print('pre')
    helpers.loc[helpers['EC_FLAG'], 'EC_TIME'] = eclipse_time(helpers['BETA'], helpers['T'], helpers['POS'])
    print('post')
    df['EC_TIME'] = helpers['EC_TIME']
    df['EC_FRAC'] = df['EC_TIME']/df['PERIOD']
    df['Q_SOL'] = df['ALPHA'] * 0.4*df['AREA'] * c.SOLARFLUX * (1-df['EC_FRAC'])
    df['Q_TOT'] = df['Q_ALB'] + df['Q_IR'] + df['Q_SOL']
    
    end = time.perf_counter() - start

    print(helpers)
    print(df)
    print('\nTIME: {}\n'.format(end))

    print('MEAN:')
    print(df.mean())
    print('\nSTD:')
    print(df.std())