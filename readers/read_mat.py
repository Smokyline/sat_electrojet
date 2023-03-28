import scipy
import numpy as np
from settings import DATA_DIR

def load_vortex(filename):
    mat = scipy.io.loadmat(DATA_DIR + '/%s.mat'%filename)
    LON = np.array(mat['LON'])
    LAT = np.array(mat['LAT'])
    vY = np.array(mat['vY'])
    vX = np.array(mat['vX'])
    return [LON, LAT, vX, vY]