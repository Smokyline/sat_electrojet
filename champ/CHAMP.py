import datetime

from sattelite import Sattelite
import matplotlib.pyplot as plt
import numpy as np
import sys

from dateutil import tz
from tools import get_local_time, get_lon_from_LT, rotate_NEC_vector_to_MFA
from proj_im_creator import get_proj_image, get_single_plot
from PIL import Image
from settings import RESULT_DIR
from chaos7_model.chaos_model import CHAOS7
from readers.read_mat import load_vortex


def plot_sat(save_plot=False):
    plt.clf()
    proj_extend_loc = [-179, 179, 55, 90]  # ortho n
    central_lon = get_lon_from_LT(sat_datetime[int((len(sat_datetime) - 1) / 2)])
    print('central lon', central_lon, sat_datetime[int((len(sat_datetime) - 1) / 2)], )
    (status, im1) = get_proj_image(sat_label=sat_label, sat_datetime=sat_datetime, sat_pos=sat_pos, sat_value=sat_value,
                                   legend_label=sat_label.rsplit('_')[1], sat_poly_loc=proj_extend_loc,
                                   sat_rad_loc=None, sat_rad_deg=5,
                                   # legend_label='F', sat_poly_loc=None, sat_rad_loc=rad_loc, sat_rad_deg=15,
                                   proj_type='ortho_n', draw_IGRFvector_diff=False,
                                   # proj_type='miller', draw_IGRFvector_diff=False,
                                   draw_CHAOSvector_diff=False, mag_grid_coord=False,
                                   lt_grid_coord=True, obs=obs, central_lon=central_lon,
                                   vortex_field=vortex_array)

    if 'fac' in sat_label and vortex_array is not None:
        vortex_jz = sat.calc_jz_of_vortex(vortex_array, sat_datetime, sat_pos)
        (status, im2) = get_single_plot(sat_label, sat_datetime, sat_pos, sat_value, vortex_jz=vortex_jz )
    else:
        (status, im2) = get_single_plot(sat_label, sat_datetime, sat_pos, sat_value)
    out_image = Image.fromarray((np.array(im1)).astype(np.uint8))
    # out_image = single_image(im4)
    filename = sat_label + ' %sUT' % sat_datetime[0].strftime('%H_%M')
    print('saved as %s' % RESULT_DIR + '/CHAMP/%s.jpg' % filename)
    out_image.save(RESULT_DIR + '/CHAMP/%s.jpg' % filename, dpi=(400, 400))
    if save_plot:
        Image.fromarray((np.array(im2)).astype(np.uint8)).save(RESULT_DIR + '/CHAMP/%s_plot.jpg' % filename)


obs = {
    #'RAN': [72.6, -24.6],
    #'GHC': [77.6, -33.69],
    'YKC': [62.48, -114.48],
    #'GIM': [66.3, -27.5],
    'FMC': [56.597, -111.31],
    'CBB': [69.123, -105.031],
    'BLC': [64.318, -96.012],
    'FCC': [58.759, -94.088],
    'INK': [68.25, -133.3],
    'BRW': [71.3225, -156.6231],
    'STF': [67.02, -50.72],
    'MEA': [54.616, -113.347]


}

sat = Sattelite(dt_from='2007-12-08T00:00:00', dt_to='2007-12-12T00:00:00')
#sat = Sattelite(dt_from='2007-12-10T00:00:00', dt_to='2007-12-10T00:00:00')
chaos = CHAOS7()

sat_data = sat.import_CHAMP_from_cdf(delta=1)   # dt, theta, phi, r, N, E, C, F
print('delta:', sat_data[1, 0]-sat_data[0, 0])

# select time
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 0, 00), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 12, 00))]
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 8, 0, 00), sat_data[:, 0] <= datetime.datetime(2007, 12, 8, 12, 00))]
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 0), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 0))]   # 6 utc  -  8 utc

# TRACKS
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 4, 56), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 5, 13))]  # 1nd
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 27), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 6, 43))]  # 1st main
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 7, 58), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 15))]  # 3 main
sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 9, 30), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 9, 46))]  # 2nd
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 11, 00), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 11, 20))]  # 4 main

#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 36), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 6, 43))]  # 1st vortex
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 7, 59), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 7))]  # 3 vortex
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 11, 1), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 11, 12))]  # 4 vortex


#vortex_array = None
#vortex_array = load_vortex(filename='vortex_05ut_paper')
#vortex_array = load_vortex(filename='vortex_06ut_paper')
#vortex_array = load_vortex(filename='vortex_08ut_paper')
vortex_array = load_vortex(filename='vortex_09ut_paper')
#vortex_array = load_vortex(filename='vortex_11ut_paper')


sat_datetime = sat_data[:, 0]
sat_pos = sat_data[:, 1:4]
sat_value = sat_data[:, 4:]
sat_value_diff = chaos.get_sat_and_chaos_diff(sat_datetime, sat_pos, sat_value)


#for sat_label in ['CHAMP_fac', 'CHAMP_dN', 'CHAMP_dE', 'CHAMP_dC', 'CHAMP_dF']:
for sat_label in ['CHAMP_fac',]:
    #sat_label = 'CHAMP_dN'
    # sat_label = 'CHAMP_fac'

    channel = 0  # N=0, E=1, C=2, F=3
    if 'dN' in sat_label:
        channel = 0
    elif 'dE' in sat_label:
        channel = 1
    elif 'dC' in sat_label:
        channel = 2
    elif 'dF' in sat_label:
        channel = 3

    """print(sat_pos, 'sat pos')
    print(min(sat_pos[:, 0]), max(sat_pos[:, 0]), 'lat')
    print(min(sat_pos[:, 1]), max(sat_pos[:, 1]), 'lon')
    print(min(sat_pos[:, 2]), max(sat_pos[:, 2]), 'rad')"""
    if 'fac' in sat_label:
        sat_value_diff_MFA = rotate_NEC_vector_to_MFA(sat_pos[:, 0], sat_pos[:, 1], sat_pos[:, 2]+6371.2, sat_value_diff, delta=30)
        sat_value = sat.calc_FAC(sat_datetime, sat_pos, sat_value_diff)
    else:
        sat_value = sat_value_diff[:, channel]  # !!!!!!!!!!

    print('VALUE min:%s max:%s' % (np.min(np.array([x for x in sat_value if not np.isnan(x)])),
                                   np.max(np.array([x for x in sat_value if not np.isnan(x)]))))

    sat_value = sat_value
    plot_sat(save_plot=True)
