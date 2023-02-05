import sys

import numpy as np
import pandas as pd

sys.path.append(sys.path[0] + '/..')
from dataclasses import dataclass
from json import load as jsonload

from Parameter import Parameter, UniformParameter
from scipy.integrate import solve_ivp

import constants as c


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
    
    def __init__(self, tlePD:pd.DataFrame=None, tleJSONFP:str=c.CUBESATS)->None:

        if tlePD is None:
            tlePD = self.__read_TLE_to_df(tleJSONFP)
        
        self.TLES = tlePD

        self.params:dict = self.__create_params()
        self.params['THETA'] = UniformParameter(0, 360, name='THETA', units='deg')
        self.params['INITIAL_TEMP'] = Parameter(20, 5, 0, name='INITIAL_TEMP', units='C')
        self.params['JULIAN_DATE'] = UniformParameter(c.J2000, c.J2000 + c.JYEAR, name='JULIAN_DATE', units='Day')

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

        for i in range(filelen):
            tle = TLE.from_dict(sat_tles[i])
            
            ecc[i] = tle.ecc
            inc[i] = tle.inc 
            raan[i] = tle.raan
            arg[i] = tle.arg
            semi[i] = tle.a
            mean_mot[i] = tle.mean_motion
            mean_anom[i] = tle.mean_anomaly

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

class TempData:

    def __init__(self, Absorptivity:Parameter=Parameter(0.5, 0.1, name='Absorptivity', units='-'),
                       Emissivity:Parameter=Parameter(0.5, 0.1, name='Emissivity', units='-'),
                       Area:Parameter=Parameter(6*c.C1U, 0.0003, name='1U', units='m'), *, temp_pkl:str=c.TEMP_DATA, orbitData = OrbitData())->None:

        self.orbitdata = orbitData
        self.abs = Absorptivity
        self.emi = Emissivity
        self.area = Area


        if temp_pkl is None:
            self.temp_data, self.helper_data = self.recalc_temp()

        else:
            self.temp_data = pd.read_pickle(temp_pkl)

        self.temp_param = self.__set_parameter()
        self.params = list(self.temp_data.columns)
        return

    def __repr__(self)->str:
        return 'Heat Data Class:\n{}'.format(self.temp_data)

    def recalc_temp(self, mod_params:list=None, num_runs:int=10_000, *, min_alt:float=300)->pd.DataFrame:

        if mod_params is None:
            mod_params = list(self.orbitdata.params.keys())
        
        helper_df = pd.DataFrame()

        # ORBIT + SAT PARAMS
        temp_df = self.__modulate_params(mod_params, num_runs)
        temp_df = temp_df.drop(['ARG', 'MEAN_ANOMALY', 'MEAN_MOTION'], axis=1)
        
        temp_df = temp_df[temp_df['SEMI'] * (1 - temp_df['ECC']) > (c.EARTHRAD + min_alt)]
        temp_df = temp_df.reset_index(drop=True)
        
        num_runs = len(temp_df.index)   # update number of runs to number of feasible orbits generated

        temp_df['PERIOD'] = 2*np.pi / np.sqrt(c.EARTHMU) * temp_df['SEMI']**1.5
        temp_df['AREA'] = self.area.modulate(num_runs)
        temp_df['ALPHA'] = self.abs.modulate(num_runs)
        temp_df['EMI'] = self.emi.modulate(num_runs)

        # Q_ALBEDO
        # TODO: FIX VIEW FACTOR -> NOT 0 EVER; NEED TO TAKE INTEGRAL OF FUNCTION AND GET AVERAGE Q_ALB OVER PERIOD
        helper_df['VF'] = self.__calc_view_factor(num_runs)
        temp_df['Q_ALB'] = self.__calc_Q_alb(temp_df['AREA'], temp_df['ALPHA'], helper_df['VF'])

        # Q_IR
        helper_df['Q_H'] = self.__calc_Qh(temp_df['SEMI'], temp_df['ECC'], temp_df['THETA'])
        helper_df['Fa'], helper_df['Fb'] = self.__calc_face_ir_view(helper_df['Q_H'])
        temp_df['Q_IR'] = self.__calc_Q_ir(temp_df['EMI'], temp_df['AREA'], helper_df['Fa'], helper_df['Fb'])

        # Q_SOL
        helper_df['Sun_pos'] = temp_df['JULIAN_DATE'].apply(self.__solar_position)
        radec = helper_df['Sun_pos'].apply(self.__vec_to_ra_dec)
        helper_df['RA'] = radec.apply(lambda x: x[0])
        helper_df['DEC'] = radec.apply(lambda x: x[1])
        helper_df['BETA_ANG'] = self.__calc_beta_ang(helper_df['RA'], helper_df['DEC'], temp_df['RAAN'], temp_df['INC'])
        temp_df['ECLIPSE_TIME'] = 0
        ecl_flag =  np.abs(np.sin(helper_df['BETA_ANG'])) < 1/helper_df['Q_H']
        temp_df['ECLIPSE_TIME'].loc[ecl_flag] = self.__calc_eclipse_time(helper_df['Q_H'], helper_df['BETA_ANG'], temp_df['PERIOD'])
        temp_df['ECLIPSE_FRAC'] = temp_df['ECLIPSE_TIME']/temp_df['PERIOD']
        temp_df['Q_SOL'] = self.__calc_Q_sol(temp_df['ALPHA'], temp_df['AREA'], temp_df['ECLIPSE_FRAC'])

        # TEMP CALCS
        temp_df = temp_df[['PERIOD', 'ECLIPSE_TIME', 'Q_IR', 'Q_ALB', 'Q_SOL', 'AREA', 'EMI', 'INITIAL_TEMP']].apply(self.__calc_temp, axis=1)
        
        return temp_df, helper_df

    def save_frame(self, path:str=c.TEMP_DATA)->None:
        self.temp_data.to_pickle(path)
        return

    """ 
    UTILITY 
    """
    def __modulate_params(self, mod_params:list, num_runs:int)->pd.DataFrame:

        df = pd.DataFrame()

        for param_name in self.orbitdata.params:
            param_cls = self.orbitdata.params[param_name]
            if param_name in mod_params:
                df[param_name] = param_cls.modulate(num_runs)
            else:
                df[param_name] = np.ones(num_runs) * param_cls.ideal

        return df

    def __set_parameter(self)->Parameter:

        data = self.temp_data[['MEAN_TEMP', 'TEMP_STD']]

        mean = np.mean(data['MEAN_TEMP'])
        std = np.mean(data['TEMP_STD'])

        return Parameter(mean, std, 0, name='TEMP', units='C')
    
    """ 
    Q ALBEDO
    """
    def __calc_view_factor(self, num_runs:int, mean:float=0.5, stddev:float=0.15, max:float=0.8)->float:
        R = max
        S = stddev
        M = mean
        cst = 1/(S*np.sqrt(2*np.pi)) 

        to_norm = lambda x: np.exp(-0.5*((x - M)/S)**2) * cst
        to_vf = lambda x: R - np.minimum(R, x)

        x = np.random.uniform(0, 1, num_runs)
        vf = np.apply_along_axis(to_vf, 0, np.apply_along_axis(to_norm, 0, x))

        return vf

    def __calc_Q_alb(self, area:float, alpha:float, vf:float)->float:
        return alpha * area * c.TEMP_GAMMA * c.EARTHFLUX * vf

    """
    Q INFRARED
    """
    def __calc_Qh(self, semi_major:float, ecc:float, theta:float)->float:
        pos = semi_major * (1 - ecc**2)/(1 + ecc*np.cos(np.deg2rad(theta)))
        return pos/c.EARTHRAD

    def __calc_face_ir_view(self, h:float)->float:

        Fa = 1/h**2
        Fb = -np.sqrt(h**2 -1)/(np.pi * h**2) + 1/np.pi * np.arctan(1/np.sqrt(h**2-1))

        return Fa, Fb

    def __calc_Q_ir(self, emi:float, area:float, face_view_A:float, face_view_B:float)->None:
        faceA = 1/3 * emi * area * c.EARTHFLUX * face_view_A
        faceB = 2/3 * emi * area * c.EARTHFLUX * face_view_B
        return faceA + faceB

    """ 
    Q SOLAR
    """
    def __solar_position(self, julianDate:float)->np.ndarray:
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

        #...Distance from earth to sun (km):
        rS    = (1.00014 - 0.01671*c.cosd(M) - 0.000140*c.cosd(2*M))*c.AU

        #...Geocentric position vector (km):
        r_S   = rS*u

        return r_S

    def __vec_to_ra_dec(self, vector:np.ndarray)->float:
        
        vec = np.linalg.norm(vector)
        v = vector/vec

        dec = np.arcsin(v[2])
        ra = np.arcsin(v[1]/np.cos(dec)) 
        return ra, dec

    def __calc_beta_ang(self, ra:float, dec:float, raan:float, inc:float)->float:

        bA = np.cos(dec)*np.sin(inc)*np.sin(raan - ra)
        bB = np.sin(dec)*np.cos(inc)
        beta = np.arcsin(bA + bB)

        return beta

    def __calc_eclipse_time(self, q_h:float, beta:float, period:float)->float:

        num = 1 - (1/q_h)**2
        den = np.cos(beta)

        delT = np.arccos(np.sqrt(num)/den)*period/np.pi

        return delT

    def __calc_Q_sol(self, alpha:float, area:float, ec_frac:float)->float:
        return alpha * 0.4 * area * c.SOLARFLUX * (1 - ec_frac)
    

    """ 
    TEMP ODE
    """
    def __calc_temp(self, row:pd.Series)->pd.Series:

        T = row.PERIOD
        eT = row.ECLIPSE_TIME

        tA = (T-eT)/2
        tB = tA + eT

        def __temp_diffeq(t:float, state:float, heat_capacity:float=961)->float:

            Q = row.Q_IR + row.Q_ALB + row.Q_SOL

            if tA < t and t <= tB:
                Q -= row.Q_SOL

            return (-1*c.STEFBOLTZ*row.AREA*row.EMI*state**4 + Q)/heat_capacity

        sol = solve_ivp(__temp_diffeq, (0, T), [273.15+row.INITIAL_TEMP], t_eval=np.linspace(0, T, 100), rtol=1e-5, atol=1e-5)
        temps = np.array(sol['y'][0])

        row['MEAN_TEMP'] = np.mean(temps) - 273.15
        row['TEMP_STD']  = np.std(temps)

        return row

class Orbit:

    mu:int = c.EARTHMU
    rad:int = c.EARTHRAD

    def __init__(self, incParam:Parameter=None,
                       eccParam:Parameter=None,
                       semiParam:Parameter=None,
                       tempParam:Parameter=None,
                       *,
                       name:str = None,
                       orbitData:OrbitData=None,
                       tempData:TempData=None)->None:

        self.orbitData:OrbitData = orbitData
        self.tempData:TempData = tempData

        self.inc:Parameter = self.__set_parameter(incParam, "INC")
        self.ecc:Parameter = self.__set_parameter(eccParam, "ECC")
        self.semi:Parameter = self.__set_parameter(semiParam, "SEMI")
        self.temp:Parameter = self.__set_parameter(tempParam, "TEMP")

        self.arg = UniformParameter(0, 360, 'ARG', c.DEG)
        self.raan = UniformParameter(0, 360, 'RAAN', c.DEG)
        self.theta = UniformParameter(0, 360, 'THETA', c.DEG)
        self.jd = UniformParameter(c.J2000, c.J2000+365.25, 'JULIAN_DATE', 'days')

        self.params = {
                        'INC':self.inc,
                        'ECC':self.ecc,
                        'SEMI':self.semi,
                        'TEMP':self.temp,
                        'ARG':self.arg,
                        'RAAN':self.raan,
                        'THETA':self.theta,
                        'JULIAN_DATE':self.jd
                      }

        self.data = self.randomize()
        self.name = name

        return
    
    def __repr__(self)->str:
        name = 'Orbit: {}'\
                '\n\t{}'\
                '\n\t{}'\
                '\n\t{}'.format(self.name,self.inc, self.ecc, self.semi)
        return name

    def randomize(self, mod_params:list=None, num:int=10_000)->None:
        df = pd.DataFrame()

        if mod_params is None:
            mod_params = list(self.params.keys())

        for param_name in self.params:
            if param_name in mod_params:
                df[param_name] = self.params[param_name].modulate(num)
            else:
                df[param_name] = self.ideal*np.ones(num)

        return df
  
    def __set_parameter(self, param:Parameter, name:str)->Parameter:
        if param is not None:
            return param
        
        if self.orbitData is None:
            print('{}Generating Default Orbit Data{}'.format(c.YELLOW, c.DEFAULT))
            self.orbitData = OrbitData()

        if self.tempData is None:
            print('{}Generating Temp Data{}'.format(c.YELLOW, c.DEFAULT))
            self.tempData = TempData(orbitData=self.orbitData)
        
        if name == 'TEMP':
            return self.tempData.temp_param

        return self.orbitData.params[name]
