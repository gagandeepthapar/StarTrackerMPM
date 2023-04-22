import pandas as pd
import numpy as np
from copy import deepcopy
import time

MAX_MAG = 5

RA = np.pi/4
DEC = np.pi/4
FOV = np.deg2rad(15)

dfA = pd.read_pickle('BSC5PKL.pkl')

st = time.perf_counter()
for i in range(1000):
    df = deepcopy(dfA)
    df = df[df['v_magnitude'] <= MAX_MAG]
    df = df[df['right_ascension'] <= RA + FOV/2]
    df = df[df['right_ascension'] >= RA - FOV/2]
    df = df[df['declination'] <= DEC + FOV/2]
    df = df[df['declination'] >= DEC - FOV/2]

end = time.perf_counter() - st

print(df)
print(len(df.index))
print(end)
