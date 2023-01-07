import numpy as np

try:
    from StarTrackerMPM import constants as c
except:
    import sys
    sys.path.append(sys.path[0] + '/..')
    import constants as c

from Parameter import Parameter

class Material:

    def __init__(self, Absorptivity:Parameter=None,
                       Emissivity:Parameter=None,
                       Area:float=None,
                       *,
                       matName:str="Generic")->None:

        self.abs = self.__set_parameter(Absorptivity, "Absorptivity")
        self.emi = self.__set_parameter(Emissivity, "Emissivity")
        self.area = self.__set_parameter(Area, "Area")
        self.name = matName

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

        return Parameter(0.5, 0.05, 0, name)    # random parameter; will be updated to be based on known materials eg Aluminum

class SatNode:

    def __init__(self, sideA:Material, sideB:Material, sideC: Material)->None:

        return

    def get_attitude(self)->tuple[np.ndarray]:

        return    