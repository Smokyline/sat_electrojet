import numpy as np
from spacepy import coordinates as coord
import aacgmv2
from spacepy.time import Ticktock
import datetime
import time
import pandas as pd
from astropy.time import Time, TimeDelta
from astropy import units as u
from spacepy import pycdf
import math
from math import cos, radians, sin, sqrt
from pyproj import Proj, transform, CRS
from shapely.geometry.polygon import Polygon
from shapely.geometry import Point



def decode_str_dt_param(str_dt):
    date, time = str_dt.split('T')
    dt = datetime.datetime.strptime(
        '%s %s' %
        (date, time), '%Y-%m-%d %H:%M:%S')
    return dt

def mjd2000_to_datetime(times):
    t = (Time(2000, format='jyear')+ times - TimeDelta(datetime.timedelta(hours=12), format='datetime')).datetime
    return t

def datetime_to_mjd2000(times):
    time = pycdf.Library().v_datetime_to_epoch(times)  # CDF epoch format
    time = time / (1e3 * 3600 * 24) - 730485  # time in modified Julian date 2000
    return time

def decode_str_time_param(str_time):
    dt_time = datetime.datetime.strptime(str_time, '%H:%M:%S')
    return dt_time

def get_local_time(dt, lon):
    UTtime_h, UTtime_m, UTtime_s = dt.hour, dt.minute, dt.second
    #loct = np.rint(lon / 15) + UTtime_h + (UTtime_m / 60) + (UTtime_s / 3600)
    loct = int(lon / 15) + UTtime_h + (UTtime_m / 60) + (UTtime_s / 3600)
    if loct < 0:
        loct = 24 + loct
    if loct >= 24:
        loct = loct - 24
    local_time = datetime.datetime.strptime('%s %s' %
                                                ('%i-%i-%i' % (dt.year, dt.month, dt.day),
                                                 '%i:%i:%i' % (loct, UTtime_m, UTtime_s)),
                                            '%Y-%m-%d %H:%M:%S')
    local_time = local_time.time()
    return local_time

def get_lon_from_LT(dt):
    """print('time:', dt.time())
    lon_lt = (dt.hour*15)
    try:
        lon_lt += (dt.minute/60*15)
    except:
        print('minute is 0')
    try:
        lon_lt += (dt.second/60*15)
    except:
        print('second is 0')
    #return 360-lon_lt
    return -lon_lt
    #return lon_lt
    #return 2*np.pi *(dt.hour/1440)*360"""

    seconds_per_day = 24 * 60 * 60.0
    sec_since_midnight = (dt - datetime.datetime(dt.year, dt.month, dt.day)).seconds
    lng = -(sec_since_midnight / seconds_per_day) * 360
    #lng = (lng + 180) % 360 - 180  # 0 360 to -180 180
    lng = (lng + 180) % 360 - 180  # 0 360 to -180 180

    return lng


# shapely polygons
def points_in_poly(poly_points, swarm_pos, shapely_convert):
    if shapely_convert:
        polygon = Polygon([(lon, lat) for lon, lat in poly_points])
    else:
        polygon = poly_points
    p_containts = [polygon.contains(Point(lat, lon)) for lon, lat in swarm_pos]
    return p_containts, polygon
def get_points_in_poly(swarm_pos, swarm_poly_loc, proj_type, shapely_convert_to_poly=True):
    if proj_type == 'ortho_n' or proj_type == 'ortho_s':
        p_in_p, poly = points_in_poly(poly_points=swarm_poly_loc, swarm_pos=swarm_pos, shapely_convert=shapely_convert_to_poly)
    elif proj_type == 'miller':
        # idk but work
        p_in_p1, poly = points_in_poly(poly_points=swarm_poly_loc, swarm_pos=swarm_pos, shapely_convert=shapely_convert_to_poly)
        p_in_p2, poly = points_in_poly(poly_points=swarm_poly_loc, swarm_pos=swarm_pos, shapely_convert=shapely_convert_to_poly)
        for i, point in enumerate(p_in_p2):
            if p_in_p1[i] == False:
                p_in_p1[i] = p_in_p2[i]
        p_in_p = p_in_p1
    return p_in_p, poly   # bool array

def get_swarm_poly_loc(point, deg_radius):
    # poly = [ -y, +y, -x, +x]
    #delta = deg_radius * 2
    y1, y2 = point[0] - deg_radius, point[0]+deg_radius
    x1, x2 = point[1] - deg_radius, point[1] + deg_radius
    return [x1, x2, y1, y2]

def swarm_egrf_vector_subtraction(swarm_pos, swarm_values_full, swarm_date):
    #TODO igrf12 to 13
    year = int(swarm_date[0].year)
    #switched_swarm_pos = convert_coord_system(swarm_pos[:, :2], dest_sys='apex')  # convert geo to mag coords
    #switched_swarm_pos = geo2mag(swarm_pos, swarm_date, to_coord='GSM')
    B = []
    idx = 0
    for n, e in swarm_values_full:
        """  
                d, i, h, x, y, z, f = calculate_magnetic_field_intensity(lon=switched_swarm_pos[idx, 1], lat=switched_swarm_pos[idx, 0], alt=switched_swarm_pos[idx, 2], date=date)
                d, i, h, x, y, z, f = variation_of_magnetic_filed_intensity(lon=swarm_pos[idx, 1], lat=swarm_pos[idx, 0], alt=swarm_pos[idx, 2], date=date)
                print('n:%s e:%s' % (n, e))
                print('x:%s y:%s' % (x, y))
                Bb = np.sqrt((n-x)**2+(e-y)**2)
                Bb = np.sqrt((n-x)**2+(e-y)**2)
                print('d:%s dx:%s dy%s' % (dd, dx, dy))
                Bd = dd - d
                print(dd, end='======\n')
                b1, b2 = np.sqrt(n-x), np.sqrt(e-y)

                """
        d, i, h, x, y, z, f = igrf_value(lat=swarm_pos[idx, 0], lon=180.-swarm_pos[idx, 1],  alt=swarm_pos[idx, 2], year=year)
        #d, i, h, x, y, z, f = igrf_value(lat=switched_swarm_pos[idx, 0], lon=switched_swarm_pos[idx, 1], alt=switched_swarm_pos[idx, 2], year=year)
        #print('sw_n:%.2f sw_e:%.2f x:%.2f y:%.2f' % (n, e, x, y))
        dd, dx, dy = magfield_variation(n, e, x, y)
        B.append([dd, dx, dy])
        idx += 1
    return np.array(B)

#####################################################################################################################

def calc_circle_lenght_latlon(latlon1, latlon2):
    """

    :param latlon1: [lat, lon]
    :param latlon2: [lat, lon]
    :return: dist in m
    """

    # pi - число pi, rad - радиус сферы (Земли)
    rad = 6371008

    # координаты двух точек
    llat1, llong1 = latlon1[1], latlon1[0]
    llat2, llong2 = latlon2[1], latlon2[0]

    # в радианах
    lat1 = llat1 * math.pi / 180.
    lat2 = llat2 * math.pi / 180.
    long1 = llong1 * math.pi / 180.
    long2 = llong2 * math.pi / 180.

    # косинусы и синусы широт и разницы долгот
    cl1 = math.cos(lat1)
    cl2 = math.cos(lat2)
    sl1 = math.sin(lat1)
    sl2 = math.sin(lat2)
    delta = long2 - long1
    cdelta = math.cos(delta)
    sdelta = math.sin(delta)

    # вычисления длины большого круга
    y = math.sqrt(math.pow(cl2 * sdelta, 2) + math.pow(cl1 * sl2 - sl1 * cl2 * cdelta, 2))
    x = sl1 * sl2 + cl1 * cl2 * cdelta
    ad = math.atan2(y, x)
    dist = ad * rad

    """# вычисление начального азимута
    x = (cl1 * sl2) - (sl1 * cl2 * cdelta)
    y = sdelta * cl2
    z = math.degrees(math.atan(-y / x))

    if (x < 0):
        z = z + 180.

    z2 = (z + 180.) % 360. - 180.
    z2 = - math.radians(z2)
    anglerad2 = z2 - ((2 * math.pi) * math.floor((z2 / (2 * math.pi))))
    angledeg = (anglerad2 * 180.) / math.pi"""
    return dist


def calc_circle_lenght(latlonR1, latlonR2):
    phi1, r1 = latlonR1[1], latlonR1[2]
    phi2, r2 = latlonR2[1], latlonR2[2]
    mean_r = (r1+r2)/2
    mean_r = (mean_r * 1000) + 6371008  # m to center of Earth
    phi = np.abs(phi1-phi2)

    arclenght = 2 * np.pi * mean_r * (phi/360)
    return arclenght

def gc2gd(gc_xyz):

    wgs84 = Proj('+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs')
    geocentric = Proj('+proj=geocent +datum=WGS84 +units=m +no_defs')

    # same
    #wgs84 = CRS("EPSG:4326")   # (deg)  # Horizontal component of 3D system. Used by the GPS satellite navigation system and for NATO military geodetic surveying.
    #geocentric = CRS("EPSG:4978")    # WGS 84 (geocentric) # X OTHER, Y EAST, Z NORTH,
    x, y, z = gc_xyz[:, 0], gc_xyz[:, 1], gc_xyz[:, 2]
    lon, lat, alt = transform(geocentric, wgs84, x, y, z)
    alt = alt/1000  # m to km
    return np.array([lat, lon, alt]).T


def gc_latlon_to_xyz(lat, lon, R):

    # Convert (lat, lon, elv) to (x, y, z).
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0
    radius = R*1000     # km to m
    geo_centric_lat = geocentric_latitude(lat)

    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)
    cos_lat = math.cos(geo_centric_lat)
    sin_lat = math.sin(geo_centric_lat)
    x = radius * cos_lon * cos_lat
    y = radius * sin_lon * cos_lat
    z = radius * sin_lat

    return x, y, z

def spher_to_cart(lat, lon):

    # Convert (lat, lon) to (x, y, z).
    lat = lat * math.pi / 180.0
    lon = lon * math.pi / 180.0
    radius = 6371008     # R in m WGS84

    cos_lon = math.cos(lon)
    sin_lon = math.sin(lon)
    cos_lat = math.cos(lat)
    sin_lat = math.sin(lat)
    x = radius * cos_lon * cos_lat
    y = radius * sin_lon * cos_lat
    z = radius * sin_lat

    return x, y, z

def geocentric_latitude(lat):
    # Convert geodetic latitude 'lat' to a geocentric latitude 'clat'.
    # Geodetic latitude is the latitude as given by GPS.
    # Geocentric latitude is the angle measured from center of Earth between a point and the equator.
    # https:#en.wikipedia.org/wiki/Latitude#Geocentric_latitude
    e2 = 0.00669437999014
    return math.atan((1.0 - e2) * math.tan(lat))

###################################################################################################################


def to_spher(x, y, z):
    r = math.sqrt(x**2+y**2+z**2)
    theta = math.atan(math.sqrt(x**2+y**2)/z)
    phi = math.atan(y/x)
    return theta, phi, r

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



def geodetic_to_geocentric(latitude, longitude, height):
    """Return geocentric (Cartesian) Coordinates x, y, z corresponding to
    the geodetic coordinates given by latitude and longitude (in
    degrees) and height above ellipsoid. The ellipsoid must be
    specified by a pair (semi-major axis, reciprocal flattening).

    """

    ellipsoid = 6378137, 298.257223563  # WGS84
    #ellipsoid = 6378137, 298.257222100882711    # GRS80

    theta = radians(latitude)
    phi = radians(longitude)
    sin_theta = sin(theta)
    a, rf = ellipsoid           # semi-major axis, reciprocal flattening
    e2 = 1 - (1 - 1 / rf) ** 2  # eccentricity squared
    n = a / sqrt(1 - e2 * sin_theta ** 2) # prime vertical radius
    r = (n + height) * cos(theta)   # perpendicular distance from z axis
    x = r * cos(phi)
    y = r * sin(phi)
    z = (n * (1 - e2) + height) * sin_theta
    return x, y, z



def geo2mag(sw_pos, swarm_date, to_coord='MAG'):
    #input sw_pos format: lat, lon, rad

    # coordinate init_coords = [Z, Y, X] or [km from center of earth, lat, lon]
    init_coords = np.array([sw_pos[:, 2], sw_pos[:, 0], sw_pos[:, 1]]).T
    # alt above sea lvl to rad
    for i, r in enumerate(init_coords[:, 0]):
        earth_r = earth_radius_in_meters(init_coords[i, 1])
        init_coords[i, 0] = (init_coords[i, 0]+earth_r) * 1000 / earth_r
    cvals = coord.Coords(init_coords, 'GEO', 'sph')


    dt_array = []
    for k, sw_date in enumerate(swarm_date):
        dt_array.append(str(sw_date)+'T'+'00:00:00')
    #print(dt_array)
    cvals.ticks = Ticktock(dt_array, 'UTC')


    newcoord = np.array(cvals.convert(to_coord, 'sph').data)  # return radius, latitude, longitude

    for i, r in enumerate(newcoord[:, 0]):
        earth_r = earth_radius_in_meters(init_coords[i, 1])
        newcoord[i, 0] = newcoord[i, 0] * earth_r / 1000
    #lan, lon, r
    return np.array([newcoord[:, 1], newcoord[:, 2], newcoord[:, 0]]).T

def mag2mlt(sw_pos, swarm_date, swarm_time):
    #   magnetic longitude to MLT
    """
             When referencing this package, please cite both the package DOI and the AACGM-v2 journal article:

            Shepherd, S. G. (2014), Altitude‐adjusted corrected geomagnetic coordinates: Definition and functional approximations, Journal of Geophysical Research: Space Physics, 119, 7501–7521, doi:10.1002/2014JA020264.
            """
    convert_mlt = []
    for pos, date, time in zip(sw_pos, swarm_date, swarm_time):
        mlt_coords = np.array(aacgmv2.convert_mlt(pos, datetime.datetime.strptime(date+' '+time, '%Y-%m-%d %H:%M:%S'), m2a=False))
        convert_mlt.append(mlt_coords)

    return np.array(convert_mlt)


def get_mag_coordinate_system_lines(date, geomag_pole):
    # geomag == True -> pole is Geomagnetic, else pole is geocentric Magnetospheric
    lat_coord = [-90.0, 91.]
    lon_coord = [-180.0, 181.]
    lat_step = 15   # step of legend
    lon_step = 15   # step of legend
    r = 6371.2
    mag_lat_lines = []
    mag_lon_lines = []
    annotate_points = []
    for lat_lines_start in np.arange(lat_coord[0], lat_coord[1], lat_step):
        lat_lines = []  # coord lines in geo
        for lon in np.arange(-180, 181, 1):
            lat_lines.append([lat_lines_start, lon, r])
        #   convert to mag\geomag coords
        if geomag_pole:
            #mag_lat_lines.append(apex_coord_system_convert(np.array(lat_lines)[:, :2], source_sys='geo', dest_sys='apex'))
            #mag_lat_lines.append(geo2geomag(np.array(lat_lines)[:, :2]))
            mag_lat_lines.append(geo2mag(np.array(lat_lines), np.full((1, len(lat_lines)), date)[0], to_coord='MAG'))

        else:
            mag_lat_lines.append(geo2mag(np.array(lat_lines), np.full((1, len(lat_lines)), date)[0], to_coord='GSM'))


    for lon_lines_start in np.arange(lon_coord[0], lon_coord[1], lon_step):
        lon_lines = []
        for lat in np.arange(-90, 91, 1):
            lon_lines.append([lat, lon_lines_start, r])

        #   convert to mag\geomag coords
        if geomag_pole:
            #mag_lon_lines.append(apex_coord_system_convert(np.array(lon_lines)[:, :2], source_sys='geo', dest_sys='apex'))
            #mag_lat_lines.append(geo2geomag(np.array(lon_lines)[:, :2]))
            mag_lon_lines.append(geo2mag(np.array(lon_lines), np.full((1, len(lon_lines)), date)[0], to_coord='MAG'))

        else:
            mag_lon_lines.append(geo2mag(np.array(lon_lines), np.full((1, len(lon_lines)), date)[0], to_coord='GSM'))

    for lat in np.arange(-90+lat_step, 91-lat_step, lat_step):
        for lon in np.arange(-180, 181., lon_step):
            if geomag_pole:
                ap = geo2mag(np.array([[lat, lon, r]]), np.full((1, 1), date)[0], to_coord='MAG')[0]
            else:
                ap = geo2mag(np.array([[lat, lon, r]]), np.full((1, 1), date)[0], to_coord='GSM')[0]

            annotate_points.append([ap[0], ap[1], [lat, lon]])
    for lat in [-90, 90]:
        if geomag_pole:
            ap = geo2mag(np.array([[lat, 0, r]]), np.full((1, 1), date)[0], to_coord='MAG')[0]
        else:
            ap = geo2mag(np.array([[lat, 0, r]]), np.full((1, 1), date)[0], to_coord='GSM')[0]
        annotate_points.append([ap[0], ap[1], [lat, 0]])



    return mag_lat_lines, mag_lon_lines, annotate_points

def get_solar_coord(date, lon):
    lon_to360 = lambda x: (x + 180) % 360 - 180  # -180 180 to 0 360
    midnight_lon = get_lon_from_LT(date)
    solar_coord_lon = lon_to360(lon) - midnight_lon
    if solar_coord_lon < 0:
        solar_coord_lon = 360 + solar_coord_lon
    elif solar_coord_lon >= 360:
        solar_coord_lon = solar_coord_lon - 360
    return solar_coord_lon

def get_lt_coordinate_system_lines(date):
    """
    calculate solar coord
    :param date:
    :return:
    """
    lon_to360 = lambda x: (x + 180) % 360 - 180  # -180 180 to 0 360
    #lon_to180 = lambda x: x + 180  #

    lat_coord = [-90.0, 91.]
    lon_coord = [0, 361.]
    lat_step = 15   # step of lines
    lon_step = 15   # step of lines
    annotate_lat_step = 15   # step of legend
    annotate_lon_step = 15   # step of legend
    r = 6371.2
    midnight_lon = get_lon_from_LT(date) # from 0 360
    #midnight_lon = lon_to180(midnight_lon)

    #midnight_lon = get_lon_from_LT(datetime.datetime(2007, 12, 10, 6, 16))  # from 0 360 to -180 180
    #print(midnight_lon)
    lt_lat_lines = []
    lt_lon_lines = []
    annotate_points = []
    for lat_lines_start in np.arange(lat_coord[0], lat_coord[1], lat_step):
        for lon in np.arange(-180, 181, lat_step):
            lt_lat_lines.append([[lat_lines_start, lon, r]])

    for lon_lines_start in np.arange(lon_coord[0], lon_coord[1], lon_step):
        for lat in np.arange(-90, 91, lon_step):
            lt_lon_lines.append([[lat, lon_lines_start, r]])


    for lat in np.arange(-90+annotate_lat_step, 91-annotate_lat_step, annotate_lat_step):
        for lon in np.arange(-180, 181, annotate_lon_step):
            lt_lon_annotate = lon_to360(lon) - midnight_lon
            if lt_lon_annotate < 0:
                    lt_lon_annotate = 360 + lt_lon_annotate
            elif lt_lon_annotate >= 360:
                    lt_lon_annotate = lt_lon_annotate - 360

            annotate_points.append([lat, lon, [lat, lt_lon_annotate]])
    """for lat in [-90, 90]:
        annotate_points.append([lat, 0, [lat, 0]])"""
    return np.array(lt_lat_lines), np.array(lt_lon_lines), np.array(annotate_points)

def latlt2polar(lat, lt, hemisphere):
    """
    Converts an array of latitude and lt points to polar for a top-down dialplot (latitude in degrees, LT in hours)
    i.e. makes latitude the radial quantity and MLT the azimuthal

    get the radial displacement (referenced to down from northern pole if we want to do a top down on the north,
        or up from south pole if visa-versa)
    """
    if hemisphere == 'N':
        r = 90. - lat
    elif hemisphere == 'S':
        r = 90. - (-1 * lat)
    else:
        raise ValueError('%s is not a valid hemisphere, N or S, please!' % (hemisphere))
    # convert lt to theta (azimuthal angle) in radians
    theta = lt / 24. * 2 * np.pi - np.pi / 2

    # the pi/2 rotates the coordinate system from
    # theta=0 at negative y-axis (local time) to
    # theta=0 at positive x axis (traditional polar coordinates)
    return r, theta     # (y, x)