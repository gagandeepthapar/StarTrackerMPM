import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

import json
# from .Parameter import Parameter
# from .StarTracker import StarTracker

@dataclass
class COES:
    h:      float
    ecc:    float
    inc:    float
    raan:   float
    arg:    float
    theta:  float

    def __post_init__(self)->None:
        #TODO: implement semi major and period
        
        self.a:float = self.__calc_semimajor()
        self.T:float = self.__calc_period()
        return

    def __repr__(self)->str:
        name = "Orbital Elements:"\
                    "\n\th:\t{}\t[km3/s2]"\
                    "\n\tecc:\t{}\t[~]"\
                    "\n\tinc:\t{}\t[deg]"\
                    "\n\traan:\t{}\t[deg]"\
                    "\n\targ:\t{}\t[deg]"\
                    "\n\ttheta:\t{}\t[deg]".format(self.h, 
                                                self.ecc,
                                                self.inc,
                                                self.raan,
                                                self.arg,
                                                self.theta)


        return name

    def createOrbit(self)->None:
        return Orbit.init_from_Element(self)
    
    def __calc_semimajor(self, mu:float=398600)->float:
        rp = self.h**2 / mu * 1/(1 + self.ecc)
        ra = self.h**2 / mu * 1/(1 - self.ecc)
        return 0.5*(ra + rp)

    def __calc_period(self, mu:float=398600)->float:
        t = 2*np.pi * self.a**1.5 / np.sqrt(mu)
        return t

@dataclass
class StateVector:
    rx:     float
    ry:     float
    rz:     float
    vx:     float
    vy:     float
    vz:     float

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

        self.__propagate_orbit()

        return

    def __repr__(self)->None:
        name = "Orbit:"\
                "\n\t{}"\
                "\n\t{}".format(repr(self.COES), repr(self.RV0))
        return name

    def init_from_State(RV:StateVector)->None:
        return Orbit(RV=RV)
    
    def init_from_Element(Elements:COES)->None:
        return Orbit(Elements=Elements)

    def init_from_json(fp:str)->None:

        file = json.load(open(fp))

        if file['TYPE'] == 'STATEVECTOR':
            rx = file['RX']
            ry = file['RY']
            rz = file['RZ']
            vx = file['VX']
            vy = file['VY']
            vz = file['VZ']
            
            state = StateVector(rx, ry, rz, vx, vy, vz)
            return Orbit.init_from_State(state)
        
        if file['TYPE'] == 'COES':
            h = file['h']
            ecc = file['ecc']
            inc = file['inc']
            raan = file['raan']
            arg = file['arg']
            theta = file['theta']

            coes = COES(h, ecc, inc, raan, arg, theta)
            return Orbit.init_from_Element(coes)
        
        raise ValueError(f'{c.RED}Improper JSON Import for Orbit Creation{c.DEFAULT}')

    def get_temp(self)->float:
        return np.random.uniform(273.15, 333.15)

    def plot_orbit(self, satellite:np.ndarray=None, earth:bool=False, jd:float=None)->None:
        # check trajectory
        if self._rx_traj is None:
            self.__propagate_orbit()

        # plot setup
        fig = plt.figure()

        ax = plt.axes(projection='3d')
        ax.axis('equal')

        plt.locator_params(axis='x', nbins=5)
        plt.locator_params(axis='y', nbins=5)
        plt.locator_params(axis='z', nbins=5)

        xlbl = 'x [km]'
        ylbl = 'y [km]'
        zlbl = 'z [km]'
        title = 'Orbit Trajectory'

        ax.set_xlabel(xlbl)
        ax.set_ylabel(ylbl)
        ax.set_zlabel(zlbl)
        ax.set_title(title)

        # plot data
        ax.plot(self._rx_traj, self._ry_traj, self._rz_traj, 'b--', label='Orbit Path')

        if earth:
            self.__create_earth_model()
            ax.plot_surface(self.__earth_model_X, self.__earth_model_Y, self.__earth_model_Z, alpha=0.2, label='Earth Model')
        
        if satellite is None:
            theta = np.random.uniform(0, 360)
            randcoes = self.COES
            randcoes.theta = theta

            satellite_pos = self.__coes_to_state(randcoes)
        
        sunpos = self.__solar_position(jd)

        pos = satellite_pos.to_array()
        ax.scatter(pos[0], pos[1], pos[2], c='r', label='Satellite Position')
        ax.quiver(pos[0], pos[1], pos[2], *sunpos, color='g', length=self.COES.a/2, label='Direction to Sun')
        # add legend, show plot
        # ax.legend()
        ax.axis('equal')
        plt.show()

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

    def __propagate_orbit(self, tspan:tuple[float]=None, tstep:float=1)->None:
        if tspan is None:
            tspan = (0, self.COES.T)

        steps = int((tspan[1] - tspan[0])/tstep)

        self._rx_traj = np.empty(steps)
        self._ry_traj = np.empty(steps)
        self._rz_traj = np.empty(steps)

        self._rx_traj[0] = self.RV0.rx
        self._ry_traj[0] = self.RV0.ry
        self._rz_traj[0] = self.RV0.rz

        state = self.RV0.to_array()

        for i in range(1, steps):
            dstate = self.__two_body(state)
            state = state + dstate*tstep

            self._rx_traj[i] = state[0]
            self._ry_traj[i] = state[1]
            self._rz_traj[i] = state[2]

        return
    
    def __solar_position(self, jd:float)->bool:
        #TODO: implement solar position

        a = np.random.randint(-1, 1)
        b = np.random.randint(-1, 1)
        c = np.random.randint(-1, 1)

        return np.array([a,b,c])

    def __two_body(self, state:np.ndarray, mu:float=None)->np.ndarray:
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
    
    def __create_earth_model(self, R:float=None)->None:
        if R is None:
            R = self.rad

        u = np.linspace(0, 2*np.pi, 100)
        v = np.linspace(0, np.pi, 100)

        U, V = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]

        X = R * np.cos(U) * np.sin(V)
        Y = R * np.sin(U) * np.sin(V)
        Z = R * np.cos(V)

        self.__earth_model_X = X
        self.__earth_model_Y = Y
        self.__earth_model_Z = Z

        return

    def __R1(self, theta:float)->np.ndarray:
        R = np.array([[1, 0, 0], [0, c.cosd(theta), c.sind(theta)], [0, -c.sind(theta), c.cosd(theta)]])
        return R
    
    def __R3(self, theta:float)->np.ndarray:
        R = np.array([[c.cosd(theta), c.sind(theta), 0], [-c.sind(theta), c.cosd(theta), 0], [0, 0, 1]])
        return R
