import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
from scipy.integrate._ivp.ivp import OdeResult

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

import json

# from .Parameter import Parameter
# from .StarTracker import StarTracker

class COES:
    def __init__(self, ecc:float, inc:float, raan:float, arg:float, theta:float, *, h:float=None, a:float=None)->None:

        if not self.__check_params(h, a):
            raise ValueError('{}Provide either h [km3/s2] or a [km]{}'.format(c.RED, c.DEFAULT)) 

        self.h = h if h is not None else self.__calc_ang_mom(a, ecc)
        self.a = a if a is not None else self.__calc_semimajor(h, ecc)

        self.ecc = ecc
        self.inc = inc
        self.raan = raan
        self.arg = arg
        self.theta = theta
        self.T = self.__calc_period()

        return

    def __repr__(self)->str:
        name = "Orbital Elements:"\
                    "\n\th:\t{}\t[km3/s2]"\
                    "\n\ta:\t{}\t[km]"\
                    "\n\tecc:\t{}\t[~]"\
                    "\n\tinc:\t{}\t[deg]"\
                    "\n\traan:\t{}\t[deg]"\
                    "\n\targ:\t{}\t[deg]"\
                    "\n\ttheta:\t{}\t[deg]".format(self.h, 
                                                self.a,
                                                self.ecc,
                                                self.inc,
                                                self.raan,
                                                self.arg,
                                                self.theta)

        return name

    def createOrbit(self)->None:
        return Orbit.init_from_Element(self)

    def __check_params(self, h:float, a:float)->bool:
        haveH = h is not None
        haveA = a is not None

        if haveH ^ haveA:
            return True

        return False

    def __calc_ang_mom(self, a:float, ecc:float, mu:float=398600)->float:
        return np.sqrt(a * mu * (1 - ecc**2))
    
    def __calc_semimajor(self, h:float, ecc:float, mu:float=398600)->float:
        return h**2 / (mu * (1 - ecc**2))

    def __calc_period(self, mu:float=398600)->float:
        t = 2*np.pi * self.a**1.5 / np.sqrt(mu)
        return t

    def create_random(a_range:tuple[float]=(6778, 6978),
                      ecc_range:tuple[float]=(0, 0.1),
                      inc_range:tuple[float]=(0, 90),
                      raan_range:tuple[float]=(0, 360),
                      arg_range:tuple[float]=(0, 360),
                      theta_range:tuple[float]=(0, 360))->None:

        a = np.random.uniform(a_range[0], a_range[1])
        ecc = np.random.uniform(ecc_range[0], ecc_range[1])
        inc = np.random.uniform(inc_range[0], inc_range[1])
        raan = np.random.uniform(raan_range[0], raan_range[1])
        arg = np.random.uniform(arg_range[0], arg_range[1])
        theta = np.random.uniform(theta_range[0], theta_range[1])
        return COES(ecc, inc, raan, arg, theta, a=a)

class StateVector:
    def __init__(self, rx:float, ry:float, rz:float, vx:float, vy:float, vz:float):
        self.rx = rx
        self.ry = ry
        self.rz = rz
        self.vx = vx
        self.vy = vy
        self.vz = vz

    def __repr__(self)->str:
        name = "State Vector:"\
                "\n\tRx:\t{}\t[km]"\
                "\n\tRy:\t{}\t[km]"\
                "\n\tRz:\t{}\t[km]"\
                "\n\tVx:\t{}\t[km/s]"\
                "\n\tVy:\t{}\t[km/s]"\
                "\n\tVz:\t{}\t[km/s]".format(self.rx,
                                          self.ry,
                                          self.rz,
                                          self.vx,
                                          self.vy,
                                          self.vz)

        return name
    
    def to_array(self)->np.ndarray:
        return np.array([self.rx, self.ry, self.rz, self.vx, self.vy, self.vz])

    def position(self)->np.ndarray:
        return (self.to_array()[0:3].reshape((1,3)))[0]

    def createOrbit(self)->None:
        return Orbit.init_from_State(self)

class Orbit:

    mu = 398600
    rad = 6378

    def __init__(self, *, RV:StateVector=None, Elements:COES=None)->None:

        haveRV = type(RV) == StateVector
        haveCOE = type(Elements) == COES

        # logic to ensure only one parameter was given
        if not haveRV and not haveCOE:
            raise ValueError('{}Provide State Vector or COES{}'.format(c.RED, c.DEFAULT))
        
        if haveRV and haveCOE:
            print('{}State Vector and COES Given; using State Vector for Orbit Det due to potential disparity{}'.format(c.RED, c.DEFAULT))
            Elements = None
        
        # determine state/elements given the other
        self.RV0 = RV if RV is not None else self.__coes_to_state(Elements)
        self.COES = Elements if Elements is not None else self.__state_to_coes(RV)

        self.trajectory = self.propagate_orbit()

        return

    def __repr__(self)->None:
        name = "Orbit:"\
                "\n\t{}"\
                "\n\t{}".format(repr(self.COES), repr(self.RV0))
        return name

    def init_from_State(RV:StateVector=None, fp:str=None)->None:

        if RV is None and fp is None:
            raise ValueError('Provide at least State Vector or Path to State Vector JSON')

        if fp is None:
            return Orbit(RV=RV)
        
        if RV is None:
            with open(fp) as fp_open:
                file = json.load(fp_open)

            rx = file['RX']
            ry = file['RY']
            rz = file['RZ']
            vx = file['VX']
            vy = file['VY']
            vz = file['VZ']
            
            state = StateVector(rx, ry, rz, vx, vy, vz)
            return Orbit(RV=state)
    
    def init_from_Element(Elements:COES=None, fp:str=None)->None:
        if Elements is None and fp is None:
            raise ValueError('Provide at least Orbital Elements or Path to Orbital Elements JSON') 
        
        if fp is None:
            return Orbit(Elements=Elements)
        
        if Elements is None:
            with open(fp) as fp_open:
                file = json.load(fp_open)

            h = file['h']
            ecc = file['ecc']
            inc = file['inc']
            raan = file['raan']
            arg = file['arg']
            theta = file['theta']

            coes = COES(h, ecc, inc, raan, arg, theta)
            return Orbit(Elements=coes)

    def init_from_json(fp:str)->None:

        with open(fp) as fp_open:
            file = json.load(fp_open)

        if file['TYPE'] == 'STATEVECTOR':
            return Orbit.init_from_State(fp=fp)
        
        if file['TYPE'] == 'COES':
            return Orbit.init_from_Element(fp=fp)
        
        raise ValueError(f'{c.RED}Improper JSON Import for Orbit Creation{c.DEFAULT}')

    def get_temp(self)->float:
        return np.random.uniform(273.15, 333.15)

    def plot_alt(self, tspan:tuple=None, t_eval:np.ndarray=None)->None:

        traj_frame = self.trajectory
        if tspan is not None:
            traj_frame = self.propagate_orbit(tspan=tspan,t_eval=t_eval)

        fig = plt.figure()
        ax = fig.add_subplot()

        ax.plot(traj_frame['TIME'], traj_frame['ALT'])

        return

    def __state_to_coes(self, RV:StateVector)->COES:
        R = np.array([RV.rx, RV.ry, RV.rz])
        V = np.array([RV.vx, RV.vy, RV.vz])

        r = np.linalg.norm(R)
        v = np.linalg.norm(V)
        v_r = np.dot(R,V)/r

        h_bar = np.cross(R, V)
        h = np.linalg.norm(h_bar)

        inc = c.acosd(h_bar[2]/h)

        n_bar = np.cross(np.array([0, 0, 1]), h_bar)
        n = np.linalg.norm(n_bar)

        if n != 0:
            raan = c.acosd(n_bar[0]/n)
            if n_bar[1] < 0:
                raan = 360 - raan
        else:
            raan = 0
        
        ecc_bar = 1/self.mu * ((v**2 - self.mu/r)*R - r*v_r*V)
        ecc = np.linalg.norm(ecc_bar)

        if n != 0:
            arg = c.acosd(np.dot(n_bar, ecc_bar)/(n*ecc))
            if ecc_bar[2] < 0:
                arg = 360 - arg
        else:
            arg = 0
        
        theta = c.acosd(np.dot(ecc_bar, R)/(ecc*r))

        if v_r < 0:
            theta = 360 - theta

        return COES(h=h, ecc=ecc, inc=inc, raan=raan, arg=arg, theta=theta)
    
    def __coes_to_state(self, Elements:COES)->StateVector:
        coes = Elements

        peri_r = coes.h**2 / self.mu * (1/(1 + coes.ecc*c.cosd(coes.theta))) * np.array([[c.cosd(coes.theta)],[c.sind(coes.theta)], [0]])
        peri_v = self.mu / coes.h * np.array([[-c.sind(coes.theta)], [coes.ecc + c.cosd(coes.theta)], [0]])

        Q_bar = self.__R3(coes.arg) @ self.__R1(coes.inc) @ self.__R3(coes.raan)

        r = np.transpose(Q_bar) @ peri_r
        v = np.transpose(Q_bar) @ peri_v

        return StateVector(*r, *v) 

    def propagate_orbit(self, tspan:tuple[float]=None, t_eval:tuple=None)->None:
        if tspan is None:
            tspan = (0, self.COES.T)
        
        if t_eval is None:
            t_eval = np.linspace(tspan[0], tspan[1], 1000)

        state0 = self.RV0.to_array().reshape((1,6))[0]
        dstate = solve_ivp(self.__two_body, tspan, state0, t_eval=t_eval)
        
        trajdict = {
                    'TIME': dstate['t'],
                    
                    'POS_X': dstate['y'][0],
                    'POS_Y': dstate['y'][1],
                    'POS_Z': dstate['y'][2],
                    
                    'VE:_X': dstate['y'][3],
                    'VEL_Y': dstate['y'][4],
                    'VEL_Z': dstate['y'][5] 
                    }

        trajframe = pd.DataFrame(trajdict)
        posframe = trajframe[['POS_X', 'POS_Y', 'POS_Z']]
        trajframe['ALT'] = posframe.apply(self.__calc_altitude, axis=1)
        
        return trajframe
    
    def __calc_altitude(self, row)->pd.DataFrame:
        return np.linalg.norm(row) - self.rad

    def __solar_position(self, jd:float)->np.ndarray:
        """Adapted from "Orbital Mechanics for Engineering Students", Curtis et al.

        Calculates vector pointing from earth to sun based on Julian Date

        Args:
            jd (float): julian date

        Returns:
            np.ndarray: vector pointing to sun
        """

        jd_count = jd - c.J2000
        jd_count_cent = jd_count/36525

        mean_anom = 357.528 + 0.9856003*jd_count
        mean_anom = np.mod(mean_anom, 360)

        mean_long = 280.460 + 0.98564736*jd_count
        mean_long = np.mod(mean_long, 360)

        eclip_long = mean_long + 1.915*c.sind(mean_anom) + 0.020*c.sind(2*mean_anom)
        eclip_long = np.mod(eclip_long, 360)

        obliquity = 23.439 - 0.0000004*jd_count

        x = c.cosd(eclip_long)
        y = c.sind(eclip_long)*c.cosd(obliquity)
        z = c.sind(eclip_long)*c.sind(obliquity)

        R = (1.00014 - 0.01671*c.cosd(mean_anom) - 0.000140*c.cosd(2*mean_anom))*c.AU

        return R*np.array([x,y,z])

    def __solar_line_of_sight(self, jd:float)->bool:
        R_earth_sun = self.__solar_position(jd)
        R_earth_sc = self.RV0.position()

        theta = np.arccos(np.dot(R_earth_sun,R_earth_sc)/np.linalg.norm(R_earth_sun)*np.linalg.norm(R_earth_sc))
        thetaA = np.arccos(self.rad/np.linalg.norm(R_earth_sc))
        thetaB = np.arccos(self.rad/np.linalg.norm(R_earth_sun))

        return (theta + thetaA < thetaB)

    def __two_body(self, t:float, state:np.ndarray, mu:float=None)->np.ndarray:
        if mu is None:
            mu = self.mu

        rx = state[0]
        ry = state[1]
        rz = state[2]
        R = np.linalg.norm(np.array([rx, ry, rz]))

        vx = state[3]
        vy = state[4]
        vz = state[5]

        ax = -mu * rx / (R**3)
        ay = -mu * ry / (R**3)
        az = -mu * rz / (R**3)

        return np.array([vx, vy, vz, ax, ay, az])
    
    def __R1(self, theta:float)->np.ndarray:
        R = np.array([[1, 0, 0], [0, c.cosd(theta), c.sind(theta)], [0, -c.sind(theta), c.cosd(theta)]])
        return R
    
    def __R3(self, theta:float)->np.ndarray:
        R = np.array([[c.cosd(theta), c.sind(theta), 0], [-c.sind(theta), c.cosd(theta), 0], [0, 0, 1]])
        return R

