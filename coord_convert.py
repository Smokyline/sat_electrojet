import numpy as np
from spacepy import coordinates as coord
from spacepy.time import Ticktock
import math
from math import radians, cos, sin
import datetime

"""
if units is None and carsph == 'car':
# use standard units
            self.units = ['Re', 'Re', 'Re']
elif units is None and carsph == 'sph':
            self.units = ['Re', 'deg', 'deg']
else:
            self.units = units
if dtype == 'GDZ' and carsph == 'sph':
            self.units = ['km', 'deg', 'deg']

"""


def earth_radius_in_meters(latitude_radians):
    # latitudeRadians is geodetic, i.e. that reported by GPS.
    # http:#en.wikipedia.org/wiki/Earth_radius
    a = 6378137.0 # equatorial radius in meters
    b = 6356752.3 # polar radius in meters
    cos = math.cos(latitude_radians)
    sin = math.sin(latitude_radians)
    t1 = a * a * cos
    t2 = b * b * sin
    t3 = a * cos
    t4 = b * sin
    return math.sqrt((t1*t1 + t2*t2) / (t3*t3 + t4*t4))


def geodedic_to_sattelite(init_coords, dt_array):
    # coordinate init_coords = [km above sea, lat, lon]
    cvals = coord.Coords(init_coords, 'GDZ', 'sph')

    cvals.ticks = Ticktock(dt_array, 'UTC')

    newcoord = np.array(cvals.convert('GEO', 'sph').data)  # return radius, latitude, longitude
    r, lat, lon = newcoord[:, 0], newcoord[:, 1], newcoord[:, 2]
    print('lat:%s r_meters:%s' % (lat, earth_radius_in_meters(lat)))
    r = r*earth_radius_in_meters(lat)/1000
    return lat, lon, r

def sattelite_to_geodedic(init_coords, dt_array):
    # coordinate init_coords = [km from center of earth, lat, lon]
    earth_r = earth_radius_in_meters(init_coords[1])
    init_coords[0] = init_coords[0] * 1000 / earth_r
    cvals = coord.Coords(init_coords, 'GEO', 'sph')
    cvals.ticks = Ticktock(dt_array, 'UTC')
    newcoord = np.array(cvals.convert('GDZ', 'sph').data)  # return radius, latitude, longitude
    r, lat, lon = newcoord[:, 0], newcoord[:, 1], newcoord[:, 2]
    return lat, lon, r

def sattelite_to_Geomagnetic(init_coords, dt_array):
    # coordinate init_coords = [km from center of earth, lat, lon]
    earth_r = earth_radius_in_meters(init_coords[1])
    init_coords[0] = init_coords[0] * 1000 / earth_r
    cvals = coord.Coords(init_coords, 'GEO', 'sph')
    cvals.ticks = Ticktock(dt_array, 'UTC')
    newcoord = np.array(cvals.convert('MAG', 'sph').data)  # return radius, latitude, longitude
    r, lat, lon = newcoord[:, 0], newcoord[:, 1], newcoord[:, 2]
    r = r * earth_r/1000
    return lat, lon, r

def sattelite_to_Geocentric_Solar_Magnetospheric(init_coords, dt_array):
    # coordinate init_coords = [km from center of earth, lat, lon]
    earth_r = earth_radius_in_meters(init_coords[1])
    init_coords[0] = init_coords[0] * 1000 / earth_r
    cvals = coord.Coords(init_coords, 'GEO', 'sph')
    cvals.ticks = Ticktock(dt_array, 'UTC')
    newcoord = np.array(cvals.convert('GSM', 'sph').data)  # return radius, latitude, longitude
    r, lat, lon = newcoord[:, 0], newcoord[:, 1], newcoord[:, 2]
    r = r * earth_r/1000
    return lat, lon, r

def geocentric_Solar_Magnetospheric_to_sattelite(init_coords, dt_array):
    # coordinate init_coords = [km from center of earth, lat, lon]
    earth_r = earth_radius_in_meters(init_coords[1])
    init_coords[0] = init_coords[0] * 1000 / earth_r
    cvals = coord.Coords(init_coords, 'GSM', 'sph')
    cvals.ticks = Ticktock(dt_array, 'UTC')
    newcoord = np.array(cvals.convert('GEO', 'sph').data)  # return radius, latitude, longitude
    r, lat, lon = newcoord[:, 0], newcoord[:, 1], newcoord[:, 2]
    r = r * earth_r/1000
    return lat, lon, r

def sattelite_to_Solar_Magnetic(init_coords, dt_array):
    # coordinate init_coords = [km from center of earth, lat, lon]
    earth_r = earth_radius_in_meters(init_coords[1])
    init_coords[0] = init_coords[0] * 1000 / earth_r
    cvals = coord.Coords(init_coords, 'GEO', 'sph')
    cvals.ticks = Ticktock(dt_array, 'UTC')
    newcoord = np.array(cvals.convert('SM', 'sph').data)  # return radius, latitude, longitude
    r, lat, lon = newcoord[:, 0], newcoord[:, 1], newcoord[:, 2]
    r = r * earth_r/1000
    return lat, lon, r

def solar_Magnetic_to_sattelite(init_coords, dt_array):
    # coordinate init_coords = [km from center of earth, lat, lon]
    earth_r = earth_radius_in_meters(init_coords[1])
    init_coords[0] = init_coords[0] * 1000 / earth_r
    cvals = coord.Coords(init_coords, 'SM', 'sph')
    cvals.ticks = Ticktock(dt_array, 'UTC')
    newcoord = np.array(cvals.convert('GEO', 'sph').data)  # return radius, latitude, longitude
    r, lat, lon = newcoord[:, 0], newcoord[:, 1], newcoord[:, 2]
    r = r * earth_r/1000
    return lat, lon, r

# coordinate init_coords = [Z, Y, X] or [rad, lat, lon]
#init_coords = np.array([6700, 45.48, -114.48]).T
init_coords = np.array([6700, 62.48, -105.031]).T

dt_array = [datetime.datetime(2007, 12, 10, 7, 38)]

#lat, lon, r = geodedic_to_sattelite(init_coords, dt_array)
#lat, lon, r = sattelite_to_geodedic(init_coords, dt_array)
#lat, lon, r = sattelite_to_Geomagnetic(init_coords, dt_array)
lat, lon, r = sattelite_to_Solar_Magnetic(init_coords = [6367, 62.48, -105.031], dt_array=dt_array)
#print('lat:%s lon:%s r:%s' % (lat, lon, r))
#lat, lon, r = solar_Magnetic_to_sattelite(init_coords = [300, 62.48, 180], dt_array=dt_array)
print('lat:%s lon:%s r:%s' % (lat, lon, r))
