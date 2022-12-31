import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

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
    
    def __calc_semimajor(self)->float:
        return 0

    def __calc_period(self)->float:
        return 100*60

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

    def ambient_temp(self)->float:
        return np.random.normal(loc=312, scale=30)

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

            satellite_pos = self._coes_to_state(randcoes)
        
        sunpos = self.__solar_position(jd)

        
        ax.scatter(*satellite_pos, c='r', label='Satellite Position')
        ax.quiver(*satellite_pos, *sunpos, color='g', length=3000, label='Direction to Sun')

        # add legend, show plot
        # print(legenditems)
        ax.legend()
        ax.axis('equal')
        plt.show()

        return

    def __state_to_coes(self, RV:StateVector)->COES:
        #TODO: implement 
        return 5
    
    def __coes_to_state(self, Elements:COES)->StateVector:
        #TODO: implement
        return 0

    def __propagate_orbit(self, tspan:tuple[float]=None, tstep:float=1, *, J2:bool=False, Drag:bool=False, SRP:bool=False, NBody:bool=False)->None:
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
            dstate = self.__two_body(state, J2=J2, Drag=Drag, SRP=SRP, NBody=NBody)
            state = state + dstate*tstep

            self._rx_traj[i] = state[0]
            self._ry_traj[i] = state[1]
            self._rz_traj[i] = state[2]

        return
    
    def __solar_position(self, jd:float)->bool:
        #TODO: implement solar position

        a = np.random.uniform(-1, 1)
        b = np.random.uniform(-1, 1)
        c = np.random.uniform(-1, 1)

        return np.array([a,b,c])

    def __two_body(self, state:np.ndarray, mu:float=None, *, J2:bool=False, Drag:bool=False, SRP:bool=False, NBody:bool=False)->np.ndarray:
        if mu is None:
            mu = self.mu

        rx = state[0]
        ry = state[1]
        rz = state[2]
        R = np.linalg.norm(np.array([rx, ry, rz]))

        vx = state[3]
        vy = state[4]
        vz = state[5]

        # calc perturbations
        aPert = np.array([0,0,0])
        
        if J2:
            aPert += self.__calcJ2()
        
        if Drag:
            aPert += self.__calcDrag()
        
        if SRP:
            aPert += self.__calcSRP()

        if NBody:
            aPert += self.__calcNBody()

        ax = -mu * rx / (R**3) + aPert[0]
        ay = -mu * ry / (R**3) + aPert[1]
        az = -mu * rz / (R**3) + aPert[2]

        return np.array([vx, vy, vz, ax, ay, az])
    
    def __calcJ2(self)->np.ndarray:
        return np.array([0,0,0])

    def __calcDrag(self)->np.ndarray:
        return np.array([0,0,0])
    
    def __calcSRP(self)->np.ndarray:
        return np.array([0,0,0])

    def __calcNBody(self)->np.ndarray:
        return np.array([0,0,0])

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

