import sys

import numpy as np
import pandas as pd

sys.path.append(sys.path[0] + '/..')

import matplotlib.pyplot as plt
from alive_progress import alive_bar
from Parameter import Parameter

import constants as c

class QUEST:

    def __init__(self)->None:

        return
    
    def __repr__(self)->str:
        name = 'QUEST SOLVER'
        return name

    def get_attitude(self)->np.ndarray:

        return
    
    def __f(self, x, a, b, c, d)->float:

        return
    
    def __fp(self, x, a, b, c, d)->float:

        return

    
if __name__ == '__main__':
    q = QUEST()

    print(q)