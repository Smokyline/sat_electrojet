import numpy as np
import chaosmagpy as cp
from chaosmagpy import load_CHAOS_matfile
from chaosmagpy.model_utils import synth_values
from chaosmagpy.coordinate_utils import transform_points
from settings import CHAOS_PATH
from astropy.time import Time
import cdflib

from chaosmagpy.data_utils import mjd2000
import datetime
from spacepy import pycdf
R_REF = 6371.2

class CHAOS7():
    def __init__(self):
        #   https://chaosmagpy.readthedocs.io/_/downloads/en/latest/pdf/


        self.chaos7_mat_model = load_CHAOS_matfile(CHAOS_PATH)
        self.cdf = pycdf.Library()
    def dt_unix_to_mjd(self, times):
        t = Time(times, format='iso', scale='utc')
        t.format = 'mjd'
        time_mjd_ad = np.array(t.value).astype(float) - 51544
        return time_mjd_ad


    def magfield_variation(self, n_swarm, e_swarm, x_chaos, y_chaos, ):
        """
             n same as x
             e same as y
        """
        FACT = 180. / np.pi
        x, y, = (n_swarm + x_chaos) / 2, (e_swarm + y_chaos) / 2,
        dx, dy, = (x_chaos - n_swarm) / 2, (y_chaos - e_swarm) / 2,
        h = np.sqrt(x * x + y * y)
        dd = (FACT * (x * dy - y * dx)) / (h * h)
        return dd, dx, dy

    def calc_F(self, X, Y, Z):
        dF = []
        for x, y, z in zip(X, Y, Z):
            F = np.sqrt(x ** 2 + y ** 2 + z ** 2)
            dF = np.append(dF, F)
        return np.array(dF)

    def get_sat_and_chaos_diff(self, sat_dt_array, sat_pos, sat_value):
        """

        :param sat_dt_array: datetime array
        :param sat_pos: position array [lat, lon, r]  (r above alt)
        :return: [[dN, dE, dC],[]]
        """

        sat_theta = 90. - sat_pos[:, 0]  # colat deg
        sat_phi = sat_pos[:, 1]  # deg
        sat_radius = sat_pos[:, 2] + R_REF  # radius from altitude in km
        sat_N, sat_E, sat_C = sat_value[:, 0], sat_value[:, 1], sat_value[:, 2]
        sat_F = self.calc_F(sat_N, sat_E, sat_C)
        sat_time = self.cdf.v_datetime_to_epoch(sat_dt_array)  # CDF epoch format
        sat_time = sat_time / (1e3 * 3600 * 24) - 730485  # time in modified Julian date 2000

        B_radius, B_theta, B_phi = self.chaos7_mat_model(sat_time, sat_radius, sat_theta, sat_phi)
        F_chaos = self.calc_F(B_radius, B_theta, B_phi)

        dN = sat_N + B_theta
        dE = sat_E - B_phi
        dC = sat_C + B_radius
        dF = sat_F - F_chaos


        #for n, e, c, x, y, z in zip(sat_N, sat_E, sat_C, B_theta, B_phi, B_radius):
        #    print('sat_N:%.2f sat_E:%.2f sat_C:%.2f\nchaos_N:%.2f chaos_E:%.2f chaos_C:%.2f\n' %(n, e, c, x, y, z))

        B_diff = np.array([dN, dE, dC, dF]).T
        return B_diff

    def get_chaos_components(self, dt, lat, lon, r,):
        """

        :param dt: datetime
        :param lat: deg
        :param lon: deg
        :param r: (r above alt) km
        :return:
        """
        theta = 90. - lat  # colat deg
        phi = lon  # deg
        radius = r + R_REF  # radius from altitude in km
        time = self.cdf.v_datetime_to_epoch(dt)  # CDF epoch format
        time = time / (1e3 * 3600 * 24) - 730485  # time in modified Julian date 2000

        B_radius, B_theta, B_phi = self.chaos7_mat_model(time, radius, theta, phi)
        F = self.calc_F(B_radius, B_theta, B_phi)
        return np.array([B_theta, phi, radius, F]).T

    """
        def get_coef(self):
        #lat=swarm_pos[idx, 0], lon=swarm_pos[idx, 1],  alt=swarm_pos[idx, 2]
        #swarm_liter, swarm_pos, swarm_date, swarm_time, swarm_values = self.swarm_set
        swarm_liter, swarm_pos, swarm_date, swarm_time, vector_components = self.swarm_set
        sw_n, sw_e = vector_components[:, 0], vector_components[:, 1]
        # B_r = -Z; B_phi = Y; B_theta = -X
        theta = 90. - swarm_pos[:, 0]  # colat deg
        phi = swarm_pos[:, 1]  # deg
        radius = swarm_pos[:, 2]  # radius in km

        time = self.dt_unix_to_mjd(
            [str(a) + " " + str(b) for a, b in zip(swarm_date, swarm_time)])  # time in modified Julian date 2000

        # computing core field
        coeffs = self.chaos7_mat_model.synth_coeffs_tdep(time)  # SV max. degree 16
        self.B_radius, self.B_theta, self.B_phi = synth_values(coeffs, radius, theta, phi)
        # B_radius, B_theta, B_phi = self.chaos7_mat_model(time, radius, theta, phi)
        self.SW_C_theta, self.SW_C_phi = (sw_n * -1) - self.B_theta, sw_e - self.B_phi

    
    def get_swarm_chaos_vector_subtraction(self):

        swarm_liter, swarm_pos, swarm_datetime, vector_components = self.swarm_set
        sw_n, sw_e = vector_components[:, 0], vector_components[:, 1]

        theta = 90. - swarm_pos[:, 0]  # colat deg
        phi = swarm_pos[:, 1]  # deg
        radius = swarm_pos[:, 2]  # radius in km
        #print(theta[:10])
        #print(phi[:10])
        #print(radius[:10])
        time = self.cdf.v_datetime_to_epoch(swarm_datetime)  # CDF epoch format
        time = time / (1e3 * 3600 * 24) - 730485    # time in modified Julian date 2000

        #theta_gsm, phi_gsm = transform_points(theta, phi,
        #                                      time=time, reference='gsm')
        #index_day = np.logical_and(phi_gsm < 90, phi_gsm > -90)
        #index_night = np.logical_not(index_day)

        # complete forward computation: pre-built not customizable (see ex. 1)
        B_radius, B_theta, B_phi = self.chaos7_mat_model(time, radius, theta, phi)

        B = []
        for n, e, x, y in zip(sw_n, sw_e, B_theta, B_phi):
            dd, dx, dy = self.magfield_variation(n, e, x*-1, y)
            B.append([dd, dx, dy])
        return np.array(B)
        # compute field strength and plot together with data
        #F = np.sqrt(B_radius ** 2 + B_theta ** 2 + B_phi ** 2)

        #print('RMSE of F: {:.5f} nT'.format(np.std(F - F_swarm)))


    
    """