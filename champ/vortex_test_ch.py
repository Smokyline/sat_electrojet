import datetime

import numpy as np
import scipy.io
from settings import DATA_DIR, RESULT_DIR
import matplotlib.pyplot as plt
import math

def get_solar_coord(date, lon):
    lon_to360 = lambda x: (x - 180) % 360 - 180  # -180 180 to 0 360
    lon_to180 = lambda x: (x + 180) % 360 - 180  # 0 360 to -180 180
    midnight_lat, midnight_lon = sun_pos(date)

    """#midnight_lon = get_lon_from_LT(date)
    solar_coord_lon = midnight_lon + lon_to360(lon)
    if solar_coord_lon < 0:
        solar_coord_lon = 360 + solar_coord_lon
    elif solar_coord_lon >= 360:
        solar_coord_lon = solar_coord_lon - 360"""
    return midnight_lon

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def sph_to_cart(lon, lat):
    r = 6371.2
    az, el = math.radians(lon), math.radians(lat)  # lon, lat
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

mat = scipy.io.loadmat(DATA_DIR + '/vortex_11ut_paper.mat')
#date = datetime.datetime(2007, 12, 10, 11, 00)
dk_idx = 0

LON = np.array(mat['LON'])
LAT = np.array(mat['LAT'])
vY = np.array(mat['vY'])
vX = np.array(mat['vX'])

LON_deg = np.rad2deg(LON)
LAT_deg = np.rad2deg(LAT)


def example_1():
    # полярная проекция
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    vX2 = np.cos(LON)*(1*vX[:, :, dk_idx]) + np.sin(LON)*(1*vY[:, :, dk_idx])
    vY2 = np.sin(LON)*(1*vX[:, :, dk_idx]) - np.cos(LON)*(1*vY[:, :, dk_idx])

    ax.quiver(LON, np.pi/2-LAT, -vY2, vX2,  width=0.001)
    plt.show()

example_1()