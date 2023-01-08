import numpy as np

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

from Parameter import Parameter
from copy import deepcopy

class Material:

    def __init__(self, Absorptivity:Parameter=None,
                       Emissivity:Parameter=None,
                       Area:float=None,
                       *,
                       name:str="Generic")->None:

        self.abs = self.__set_parameter(Absorptivity, "Absorptivity")
        self.emi = self.__set_parameter(Emissivity, "Emissivity")
        self.area = self.__set_parameter(Area, "Area")
        self.name = name

        self.faceVector = None

        return
    
    def __repr__(self)->str:
        name = "Material: {}"\
                "\n\t{}"\
                "\n\t{}"\
                "\n\t{}".format(self.name, self.abs, self.emi, self.area)
        return name

    def __set_parameter(self, param:Parameter, name:str)->Parameter:
        if param is not None:
            return param

        return Parameter(0.5, 0.05, 0, name=name)    # random parameter; will be updated to be based on known materials eg Aluminum

class SatNode:

    def __init__(self, faceA:Material, faceB:Material, faceC:Material, state:np.ndarray)->None:
        
        print(faceA is faceB)

        self.faces = [faceA, deepcopy(faceA), faceB, deepcopy(faceB), faceC, deepcopy(faceC)]
        self.state = state

        # self.__set_attitude()

        return
    
    def __repr__(self)->str:
        return 'SAT Node'

    def get_attitude(self)->tuple[np.ndarray]:
        position = self.state[:3]
        velocity = self.state[3:]

        z = -1*position/np.linalg.norm(position)
        
        h = np.cross(position, velocity)
        x = h/np.linalg.norm(h)

        y = np.cross(z, x)
        return (x, y, z)

    def __set_attitude(self)->None:

        faceVecs = self.get_attitude()
        print(faceVecs)

        for i in range(3):
            print(2*i)
            print(2*i + 1)
            print(faceVecs[i])
            self.faces[2*i].faceVector = faceVecs[i]
            self.faces[2*i + 1].faceVector = -1*(faceVecs[i])

        return

if __name__ == '__main__':
    al = Material(name='a')
    bl = Material(name='b')
    cl = Material(name='cl')
    p = SatNode(al, al, cl, np.array([1, 2, 3, 4, 5, 6]))

    for i in range(6):
        print(p.faces[i].faceVector)

    # p.get_attitude(rv)