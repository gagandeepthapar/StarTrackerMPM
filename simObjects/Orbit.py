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
        # self.TLES['THETA'] = self.TLES.apply(self.__calc_theta, axis=1)

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
    
    def __calc_theta(self,x):
        mean_anom = x['MEAN_ANOMALY']
        ecc = x['ECC']
        Me = np.deg2rad(mean_anom)
        
        E_0 = Me + ecc
        if Me < np.pi:
            E_0 = Me - ecc

        f = lambda E: Me - E + ecc*np.sin(E)
        fp = lambda E: -1 + ecc*np.sin(E)
        newt = lambda E: E - f(E)/fp(E)

        E_1 = newt(E_0)
        err = np.abs(E_1 - E_0)

        while err > 1e-8:
            E_0 = E_1
            E_1 = newt(E_0)
            err = np.abs(E_1 - E_0)
        
        TA = 2*c.atand((np.sqrt((1+ecc)/(1-ecc)) * np.tan(E_1/2)))
        theta = np.mod(TA, 360)
        # print(theta)
        return theta

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

class Orbit:

    mu:int = c.EARTHMU
    rad:int = c.EARTHRAD

    def __init__(self, incParam:Parameter=None,
                       eccParam:Parameter=None,
                       semiParam:Parameter=None,
                       *,
                       name:str = None,
                       orbitData:OrbitData=None)->None:

        self.orbitData:OrbitData = orbitData

        self.inc:Parameter = self.__set_parameter(incParam, "INC")
        self.ecc:Parameter = self.__set_parameter(eccParam, "ECC")
        self.semi:Parameter = self.__set_parameter(semiParam, "SEMI")

        self.arg = UniformParameter(0, 360, 'ARG', c.DEG)
        self.raan = UniformParameter(0, 360, 'RAAN', c.DEG)
        self.theta = UniformParameter(0, 360, 'THETA', c.DEG)
        self.jd = UniformParameter(c.J2000, c.J2000+365.25, 'JULIAN_DATE', 'days')

        self.T:float=None
        self.state:StateVector=None
        self.path:pd.DataFrame=None
        self.heatFlux = pd.DataFrame()

        self.name = name
        self.randomize()
        return
    
    def __repr__(self)->str:
        name = 'Orbit: {}'\
                '\n\t{}'\
                '\n\t{}'\
                '\n\t{}'.format(self.name,self.inc, self.ecc, self.semi)
        return name

    def randomize(self, params:list=None)->None:

        if params is None:
            params = [self.inc, self.ecc, self.semi, self.arg, self.raan, self.theta, self.jd]

        for param in params:
            param.modulate()

        self.T = self.__calc_period()

        self.state = self.__calc_state()
        self.path = self.__propagate_orbit()

        self.heatFlux = pd.DataFrame()
        self.heatFlux['TIME'] = self.path['TIME']

        return

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

    def __set_parameter(self, param:Parameter, name:str)->Parameter:
        if param is not None:
            return param
        
        if self.orbitData is None:
            print('{}Generating Default Orbit Data{}'.format(c.YELLOW, c.DEFAULT))
            self.orbitData = OrbitData()
        
        print('{}Generating Parameter: {}{}'.format(c.YELLOW,c.DEFAULT,name))
        return self.orbitData.params[name]
        
    def __calc_period(self)->float:
        a = self.semi.value
        T = a**(3/2) * (2*np.pi) / np.sqrt(self.mu)
        return T

    def __propagate_orbit(self, tspan:tuple[float]=None, tstep:float=60)->pd.DataFrame:
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

    def plot_system(self):

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        # fig.add_axes(ax)
        
        # al = Material()
        # sat = SatNode(al, al, al)
        # att = sat.get_attitude()

        # sat pos and path
        ax.plot(self.path['RX'],self.path['RY'], self.path['RZ'], 'y--', label='orbit')
        ax.scatter(*self.state.position, c='r', label='satellite')
        # ax.quiver(*self.state.position, *(att[0]), length=1500, color='red', label=r'$Sat_{+X}$')
        # ax.quiver(*self.state.position, *(att[1]), length=1500, color='darkviolet',  label=r'$Sat_{+Y}$')
        # ax.quiver(*self.state.position, *(att[2]), length=1500, color='blue',  label=r'$Sat_{+Z}$')

        # ax.quiver(*self.state.position, *(-1*att[0]), length=1500, color='red', linestyle='--', label=r'$Sat_{-X}$')
        # ax.quiver(*self.state.position, *(-1*att[1]), length=1500, color='darkviolet', linestyle='--',  label=r'$Sat_{-Y}$')
        # ax.quiver(*self.state.position, *(-1*att[2]), length=1500, color='blue', linestyle='--',  label=r'$Sat_{-Z}$')

        # earth
        x, y, z = self.__earth_model()
        ax.plot_surface(x, y, z, alpha=0.2)

        # dir to sun
        rS = self.__solar_position()
        rS = rS/np.linalg.norm(rS)

        col = 'red'
        los = 'Not Found'
        if self.__solar_line_of_sight():
            col = 'green'
            los = 'Acquired'

        ax.quiver(0,0,0, *rS, length=6000, color=col, label='Dir to Sun')

        title = 'Simulated Orbit on JD {}\nSolar Line-of-Sight {}'.format(self.jd.value, los)

        ax.legend()
        ax.axis('equal')
        ax.set_title(title)
        ax.set_xlabel('X [km]')
        ax.set_ylabel('Y [km]')
        ax.set_zlabel('Z [km]')

        return
    
    def __earth_model(self)->tuple[float]:

        u, v = np.mgrid[0:2*np.pi:200j, 0:np.pi:100j]
        x = c.EARTHRAD*np.cos(u)*np.sin(v)
        y = c.EARTHRAD*np.sin(u)*np.sin(v)
        z = c.EARTHRAD*np.cos(v)
    
        return x, y, z

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

def eclipse_time(xrow)->float:
    beta = xrow['BETA']
    t = xrow['T']
    r = xrow['SEMI']

    if np.abs(np.sin(beta)) >= c.EARTHRAD/r:
        return 0

    num = 1 - (c.EARTHRAD/r)**2
    den = np.cos(beta)

    delT = np.arccos(np.sqrt(num)/den)*t/np.pi

    return delT

def t(a):
    return 2*np.pi*a/np.sqrt(c.EARTHMU/a)


if __name__ == '__main__':
    N = 100_000    
    o = Orbit()
    sat = SatNode(Material(), Material(), Material())

    o.randomize()
    o.gen_heat_flux(sat)
    o.calc_temperature(sat)

    o.heatFlux[['Q_TOTAL', 'SAT_TEMP']].plot(subplots=True, layout=(2,1))
    plt.show()

    # print(t(6878))

    # raans = np.random.uniform(0, 2*np.pi, N)
    # incs = np.random.uniform(0, np.pi/4, N)
    # semis = np.random.uniform(6778, 7078, N)


    # sol = pd.DataFrame()
    # sol['RAAN'] = raans
    # sol['INC'] = incs
    # sol['SEMI'] = semis
    # sol['T'] = t(sol['SEMI'])
    # sol['JD'] = np.random.uniform(c.J2000, c.J2000+c.JYEAR, N)

    # start = time.perf_counter()

    # sol['SVECX'], sol['SVECY'], sol['SVECZ'] = spos(sol['JD'].values)
    # sol['RA'], sol['DEC'] = vec2radec([sol['SVECX'], sol['SVECY'], sol['SVECZ']])
    # sol['BETA'] = beta_ang([sol['RA'], sol['DEC']], sol['RAAN'], sol['INC'])
    # sol['EC_TIME'] = sol[['BETA', 'T', 'SEMI']].apply(eclipse_time, axis=1)
    # sol['EC_FRAC'] = sol['EC_TIME']/sol['T']

    # end = time.perf_counter() - start
    # print(sol)
    # print('MAX TIME:\n{}'.format(sol[sol['EC_TIME'] == np.max(sol['EC_TIME'])]))
    # print('\n{} Calcs: {}\n'.format(N, end))
    

    # fig = plt.figure()
    # ax = fig.add_subplot()
    # ax.scatter(sol['SEMI'], sol['EC_FRAC'])
    # plt.show()

    # print(o.T, o.semi.value)
    # print(sv, radec, beta, tec)

    # solvecX, solvecY, solvecZ = np.apply_along_axis(spos, 0, jdspan)

    # print(solvecs[0][0], solvecs[1][0], solvecs[2][0])
    # print(solvecX[0], solvecY[0], solvecZ[0])
    

    

    # # o.randomize()
    # # o.calc_temperature(sat)

    
    # # ax = fig.add_subplot(211)
    # # ax2 = fig.add_subplot(212)
    
    # N = 10
    # start = time.perf_counter()
    # for i in range(N):
    #     fig = plt.figure()
    #     ax1 = fig.add_subplot(2,1,1)
    #     ax2 = fig.add_subplot(2,1,2)
    #     # ax3 = fig.add_subplot(3,1,3, projection='3d')

    #     loop = time.perf_counter()
    #     print('\nLOOP: {}'.format(i+1))
        
    #     o.randomize()
    #     rand = time.perf_counter() - loop
    #     print('Rand Gen: {}'.format(rand))

    #     o.gen_heat_flux(sat)
    #     heat = time.perf_counter() - loop - rand
    #     print('Heat Gen: {}'.format(heat))

    #     o.calc_temperature(sat)
    #     temp = time.perf_counter() - loop - heat
    #     print('Temp Gen: {}'.format(temp))

    #     ax1.set_title('{}{} x {}km'.format(np.round(o.inc.value, 2), c.DEG, np.round(o.semi.value)))
    #     ax1.plot(o.heatFlux['TIME'], o.heatFlux['SAT_TEMP'])
    #     ax2.plot(o.heatFlux['TIME'], o.heatFlux['Q_TOTAL'])
    #     # ax3.plot(o.path['RX'], o.path['RY'], o.path['RZ'], marker='--')
        
        

    # end = time.perf_counter() - start
    # print('\n\nTotal Time: {}/run'.format(end/N))

    # o.heatFlux[['SAT_TEMP', 'Q_TOTAL']].plot(subplots=True, layout=(2,1))
    # plt.show()