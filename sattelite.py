import datetime

import numpy as np

from tools import *
import cdflib
from settings import DATA_DIR
from chaosmagpy.coordinate_utils import transform_points
import igrf
import tqdm
import codecs
from chaos7_model.chaos_model import CHAOS7

class Sattelite():
    def __init__(self, dt_from, dt_to):
        try:
            self.dt_from = decode_str_dt_param(dt_from)
            self.dt_to = decode_str_dt_param(dt_to)
        except:
            print('WRONG IMPORT DATETIME FORMAT')
            self.dt_from = None
            self.dt_to = None
        self.chaos7 = CHAOS7()

    def get_start_stop_idx(self, time_array, dt_from, dt_to):
        idx = np.arange(len(time_array))[np.logical_and(time_array >= dt_from, time_array <= dt_to)]
        return idx[0], idx[-1]

    def import_ogo6_from_cdf(self, delta=100):
        if self.dt_from.date() != self.dt_to.date():
            dt_array = pd.date_range(start=self.dt_from.date(),end=self.dt_to.date()).to_pydatetime().tolist()
        else:
            dt_array = [self.dt_from]

        sat_data = np.empty((0, 5))
        self.mjd2000_dt = []
        for date in dt_array:

            filepath = DATA_DIR + '\mf-ogo-6-%s%02d%02d.cdf' % (date.strftime('%y'), date.month, date.day)
            cdf_file = cdflib.CDF(filepath, 'r')
            #print(cdf_file.cdf_info())
            #['t', 'r', 'theta', 'phi', 'F']
            t = cdf_file.varget('t')[0]
            r = cdf_file.varget('r')[0]
            theta = 90.-cdf_file.varget('theta')[0]
            phi = cdf_file.varget('phi')[0]
            F = cdf_file.varget('F')[0]
            #self.mjd2000_dt.extend(t / (1e3*3600*24) - 730485)
            self.mjd2000_dt.extend(t)
            dt = mjd2000_to_datetime(t)

            #print(t[0], t[-1], len(t))

            sat_data = np.append(sat_data, np.array([dt, theta, phi, r, F]).T, axis=0)
        sat_data = sat_data[0::delta]
        self.mjd2000_dt = np.array(self.mjd2000_dt[0::delta])

        return sat_data

    def import_CHAMP_from_cdf(self, delta=100):
        if self.dt_from.date() != self.dt_to.date():
            dt_array = pd.date_range(start=self.dt_from.date(), end=self.dt_to.date()).to_pydatetime().tolist()
        else:
            dt_array = [self.dt_from]

        sat_data = np.empty((0, 8))
        self.mjd2000_dt = []
        for date in dt_array:

            #filepath = DATA_DIR + '\CH-ME-2-FGM-SCI+%04d-%02d-%02d.cdf' % (date.year, date.month, date.day)
            filepath = DATA_DIR + '\CH-ME-3-MAG+%04d-%02d-%02d.cdf' % (date.year, date.month, date.day)
            cdf_file = cdflib.CDF(filepath, 'r')
            #print(cdf_file.cdf_info())
            t = cdf_file.varget('EPOCH')
            r = cdf_file.varget('GEO_ALT')
            theta = cdf_file.varget('GEO_LAT')
            phi = cdf_file.varget('GEO_LON')
            NEC = cdf_file.varget('NEC_VEC')
            N, E, C = NEC[:, 0], NEC[:, 1], NEC[:, 2]
            #N = pd.DataFrame(NEC[:, 0]).interpolate().to_numpy().T[0]
           # E = pd.DataFrame(NEC[:, 1]).interpolate().to_numpy().T[0]
            #C = pd.DataFrame(NEC[:, 2]).interpolate().to_numpy().T[0]
            F = np.sqrt((N**2 + E**2 + C**2))

            self.mjd2000_dt.extend(t)
            dt = mjd2000_to_datetime(t / (1e3*3600*24) - 730485)

            #print(t[0], t[-1], len(t))

            sat_data = np.append(sat_data, np.array([dt, theta, phi, r, N, E, C, F]).T, axis=0)
        sat_data = sat_data[0::delta]
        self.mjd2000_dt = np.array(self.mjd2000_dt[0::delta])

        return sat_data

    def import_explorer_plasma(self, filename):
        month = {
            'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
        }
        data = []
        with open(DATA_DIR + "\%s"%filename) as f:
            for line in f.readlines():
                line = line.split()
                minute = int(line[5])
                #print(line)

                if minute==60:
                    minute = 59
                try:
                    dt = datetime.datetime(int(line[0]), int(month[line[2]]), int(line[3]), int(line[4]), minute)
                    ion_density = float(line[7])
                    aberrated_flow_speed = int(line[8])
                except Exception as e:
                    print(e)
                data.append([dt, ion_density, aberrated_flow_speed])
        return np.array(data)

    def import_omni_data(self, filename, start_line):
        """

        Selected parameters:
        3 Scalar B, nT
        4 BX, nT (GSE, GSM)
        5 BY, nT (GSM)
        6 BZ, nT (GSM)
        7 SW Plasma Temperature, K
        8 SW Proton Density, N/cm^3
        9 SW Plasma Speed, km/s
        10 Flow pressure
        11 E elecrtric field
        12 Plasma Beta
        """
        data = []
        error_list = [999.9, 9999999., 99.99, 9999.0, 999.99]
        with open(DATA_DIR + "\%s" % filename) as f:
            for i, line in enumerate(f.readlines()):
                if i >= start_line-1:
                    line = line.split()
                    dt = datetime.datetime.strptime('%i-%iT%i:00:00' % (int(line[0]), int(line[1]), int(line[2])),'%Y-%jT%H:%M:%S')
                    dx = float(line[4])
                    if dx in error_list:
                        dx = np.nan
                    dy = float(line[5])
                    if dy in error_list:
                        dy = np.nan
                    dz = float(line[6])
                    if dz in error_list:
                        dz = np.nan
                    sw_pt = float(line[7])
                    if sw_pt in error_list:
                        sw_pt = np.nan
                    sw_pd = float(line[8])
                    if sw_pd in error_list:
                        sw_pd = np.nan
                    sw_ps = float(line[9])
                    if sw_ps in error_list:
                        sw_ps = np.nan
                    fp = float(line[10])
                    if fp in error_list:
                        fp = np.nan
                    ef = float(line[11])
                    if ef in error_list:
                        ef = np.nan
                    pb = float(line[12])
                    if pb in error_list:
                        pb = np.nan
                    data.append([dt, dx, dy, dz, sw_pt, sw_pd, sw_ps, fp, ef, pb])
        return np.array(data)

    def import_kp(self, filename):
        Kp_indices = {
            '0o': 0., '0+': 0.33, '1-': 0.67, '1o': 1.00, '1+': 1.33, '2-': 1.67, '2o': 2.00, '2+': 2.33, '3-': 2.67,
            '3o': 3.00, '3+': 3.33, '4-': 3.67, '4o': 4.00, '4+': 4.33, '5-': 4.67, '5o': 5.00, '5+': 5.33, '6-': 5.67,
            '6o': 6.00, '6+': 6.33, '7-': 6.67, '7o': 7.00, '7+': 7.33, '8-': 7.67, '8o': 8.00, '8+': 8.33, '9-': 8.67,
            '9o': 9.0}

        clf = pd.read_csv(DATA_DIR + "\%s"%filename, skiprows=35, sep='\s+', )
        np_clf = clf.iloc[:, [0, 1, 3]].to_numpy()
        dt = [datetime.datetime.strptime(str(np_clf[i, 0] + 'T' + np_clf[i, 1]), '%Y-%m-%dT%H:%M:%S.%f') for i in
              np.arange(len(np_clf))]
        kp_array = [Kp_indices[key] for key in np_clf[:, 2]]
        return np.vstack((dt, kp_array)).T  # dt, value

    def import_dst(self, filename):
        clf = pd.read_csv(DATA_DIR + "\%s"%filename, skiprows=24, sep='\s+', )
        np_clf = clf.iloc[:, [0, 1, 3]].to_numpy()
        dt = [datetime.datetime.strptime(str(np_clf[i, 0] + 'T' + np_clf[i, 1]), '%Y-%m-%dT%H:%M:%S.%f') for i in
              np.arange(len(np_clf))]
        return np.vstack((dt, np_clf[:, 2])).T  # dt, value

    def import_ae(self, filename):
        clf = pd.read_csv(DATA_DIR + "\%s"%filename, skiprows=26, sep='\s+', )
        np_clf = clf.iloc[:, [0, 1, 3]].to_numpy()
        dt = [datetime.datetime.strptime(str(np_clf[i, 0] + 'T' + np_clf[i, 1]), '%Y-%m-%dT%H:%M:%S.%f') for i in
              np.arange(len(np_clf))]
        return np.vstack((dt, np_clf[:, 2])).T  # dt, value


    def get_igrf13_model_data(self, date, geodetic_coords):
        """

        Parameters
        ----------

        date: datetime.date or decimal year yyyy.dddd
        glat, glon: geographic Latitude, Longitude
        alt_km: altitude [km] above sea level for itype==1
        isv: 0 for main geomagnetic field
        itype: 1: altitude is above sea level
        """
        igrf_model = np.empty((0, 4))
        for i in tqdm.tqdm(range(len(geodetic_coords)), desc="igrf13 model"):
            glat, glon, alt_km = geodetic_coords[i, 0], geodetic_coords[i, 1], geodetic_coords[i, 2]
            model = igrf.igrf(date[i], glat, glon, alt_km)
            igrf_model = np.append(igrf_model, [[float(model.north), float(model.east), float(model.down),
                                                 float(model.total)]], axis=0)
        return igrf_model

    def geocentric_to_geodetic(self, sat_data, gd_alt=False):
        """
            # geocentric_to_geodetic
            :param sat_data[:, 0]: geocentric lat -90... 90
            :param sat_data[:, 1]: geocentric lon -180... 180
            :param sat_data[:, 2]: alt above center of Earth, km
            :return: [geodetic lat, geodetic lon, alt above geoid]
            """
        geocentric_xyz = np.empty((0, 3))
        for i in tqdm.tqdm(range(len(sat_data)), desc="geocentric coords to geodetic"):
            # lat, lon, r
            geocentric_xyz = np.append(geocentric_xyz, [gc_latlon_to_xyz(sat_data[i, 0], sat_data[i, 1], sat_data[i, 2],)], axis=0)
        geodetic_latlonR = gc2gd(geocentric_xyz)
        if gd_alt:
            geocentric_latlonAlt = np.append(sat_data[:, :2], np.array([geodetic_latlonR[:, 2]]).T, axis=1).astype(float)
            return geocentric_latlonAlt
        else:
            return geodetic_latlonR

    def calc_FAC(self, sat_datetime, sat_pos, sat_value):
        mu = 4*np.pi * 10e-7    # H/m is the magnetic permeability of free space
        #mu = 1    # H/m is the magnetic permeability of free space
        """cart_sat_pos = np.empty((0, 3))
        for latlonR in sat_pos:
            x, y, z = spher_to_cart(latlonR[0], latlonR[1])     # R in m
            cart_sat_pos = np.append(cart_sat_pos, [[x, y, z]], axis=0)"""
        sat_N, sat_E, sat_C = sat_value[:, 0], sat_value[:, 1], sat_value[:, 2]
        chaos_B = self.chaos7.get_chaos_components(dt=sat_datetime, lat=sat_pos[:, 0], lon=sat_pos[:, 1], r=sat_pos[:, 2])      # N, E, C, F

        FAC = np.array([])
        for i in range(len(sat_N)):
            if i < len(sat_N)-1:
                bx, by, bz = sat_N[i], sat_E[i], sat_C[i]
                if np.isnan(bx) or np.isnan(by) or np.isnan(bz) or np.isnan(sat_E[i + 1]):
                    jFAC = np.nan
                else:
                    dx = calc_circle_lenght(sat_pos[i], sat_pos[i+1])
                    jz = (1 / mu) * ((sat_E[i + 1] - by) / dx) / 1000  # deriv   /1000 nano to micro
                    #F = np.sqrt(bx ** 2 + by ** 2 + bz ** 2)
                    sinI = chaos_B[i, 2] / chaos_B[i, 3]     # bz/F
                    jFAC = -(jz / sinI)
            else:
                jFAC = FAC[i-1]
            FAC = np.append(FAC, jFAC)
        return FAC

    def get_ThetaPhi_gsm_and_DayNight_idx(self, theta, phi, mjd2000_time, reference='gsm'):
        """

        :param theta: lat -90... 90
        :param phi: lon -180... 180
        :param mjd2000_time:
        :param reference:
        :return:
        """
        theta_gsm, phi_gsm = transform_points(90.+theta, phi,
                                              time=mjd2000_time, reference=reference)
        index_day = np.logical_and(phi_gsm < 90, phi_gsm > -90)
        index_night = np.logical_not(index_day)
        return theta_gsm, phi_gsm, index_day, index_night

    def get_DayNight_idx(self, dt, phi):
        index_day = []
        for i, t in enumerate(dt):
            local_time = get_local_time(t, phi[i])
            if local_time >= datetime.time(6, 0) and local_time < datetime.time(18, 0):
                index_day.append(True)
            else:
                index_day.append(False)
        index_night = np.logical_not(index_day)
        return index_day, index_night

    def get_latitudeSep_idx(self, theta):
        if theta[1] - theta[0] > 0:     # lowest value -> highest (border_idx) -> lowest
            border_idx = np.argmax(theta)
        else:   # highest value -> lowest -> highest
            border_idx = np.argmin(theta)

        index_leftFig = []
        for i in np.arange(len(theta)):
            if i <= border_idx:
                index_leftFig.append(True)
            else:
                index_leftFig.append(False)
        index_rightFig = np.logical_not(index_leftFig)
        return index_leftFig, index_rightFig, border_idx

    def get_datetimeSep_idx(self, dtime_list, sep_time):
        dtime_deltas = dtime_list - sep_time
        sep_time_idx = np.argmin(np.abs(dtime_deltas))
        index_leftFig = []
        for i in np.arange(len(dtime_list)):
            if i <= sep_time_idx:
                index_leftFig.append(True)
            else:
                index_leftFig.append(False)
        index_rightFig = np.logical_not(index_leftFig)
        return index_leftFig, index_rightFig, sep_time_idx

    def search_time_array_idx_in_fullTime_array(self, time_array, sat_full_time):
        idx_array = []
        for i, dt in enumerate(time_array):
            deltas = np.abs(dt - sat_full_time)
            #print(deltas)
            if np.min(deltas) < datetime.timedelta(minutes=60):
                idx_array.append([i, np.argmin(deltas)])
                #idx_array.append(i)
        return np.array(idx_array).astype(int) # [time_array_idx(i), sat_fill_time_idx(i)]

    def separate_timeIdx(self, array, index_array, separator_idx):
        array_L = array[np.where(index_array < separator_idx)]
        array_R = array[np.where(index_array >= separator_idx)]
        array_R[:, 1] = array_R[:, 1] - separator_idx
        return array_L, array_R

    def interpolate_array(self, array, dtype='datetime'):
        if dtype == 'datetime':
            df = pd.DataFrame({'value': array}, dtype=np.int64)
            df = df.interpolate('linear')
            return np.array([np.datetime64(int(dt), 'us') for dt in df.to_numpy()]).astype(datetime.datetime)
        elif dtype == 'float':
            df = pd.DataFrame({'value': array}, dtype=float)
            df = df.interpolate('linear')
            return np.array([v for v in df.to_numpy()]).astype(float)

    def get_full_dt_lon_array(self, sat_time, sat_coord):
        """

        :param sat_time: dt array with time empty space
        :param sat_coord: [lat, lon, ...] array
        :return: sat_time_full, sat_lon_full
        """
        """
        EASY WAY TIME LIST
        DTIME_TICKS_STEP = 7
        sat_time_no_empty_L = pd.date_range(start=sat_time[index_left][0], end=sat_time[index_left][-1], periods=DTIME_TICKS_STEP)
        sat_time_no_empty_R = pd.date_range(start=sat_time[index_right][0], end=sat_time[index_right][-1], periods=DTIME_TICKS_STEP)

        """
        sat_time_eps = []
        for i, dt in enumerate(sat_time):
            if i != 0:
                delta = (dt - sat_time[i - 1]).total_seconds()
                if delta <= (sat_time[-1] - sat_time[0]).total_seconds() / len(sat_time):
                    sat_time_eps.append(delta)
        sat_time_eps = float(np.mean(sat_time_eps))

        sat_time_full = []
        sat_lon_full = []
        sat_time_zero = sat_time[0]
        while True:
            if sat_time_zero > sat_time[-1]:
                break
            idx_close = np.argmin(np.abs(sat_time - sat_time_zero))
            close_sat_time = sat_time[idx_close]
            if np.abs(close_sat_time - sat_time_zero).total_seconds() <= sat_time_eps:
                # sat_time_full.append(np.datetime64(sat_time_zero, 'us').astype(np.int64))
                sat_time_full.append(np.datetime64(close_sat_time, 'us').astype(np.int64))
                sat_lon_full.append(float(sat_coord[idx_close, 1]))
            else:
                sat_time_full.append(np.nan)
                sat_lon_full.append(np.nan)
            sat_time_zero += datetime.timedelta(seconds=sat_time_eps)
        sat_time_full = self.interpolate_array(np.array(sat_time_full), dtype='datetime')
        sat_lon_full = self.interpolate_array(np.array(sat_lon_full), dtype='float')
        return sat_time_full, sat_lon_full