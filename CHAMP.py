import datetime

from sattelite import Sattelite
import matplotlib.pyplot as plt
import numpy as np
import sys

from dateutil import tz
from tools import get_local_time, get_lon_from_LT
from proj_im_creator import get_proj_image, get_single_plot
from PIL import Image
from settings import RESULT_DIR
from chaos7_model.chaos_model import CHAOS7

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

#sat = Sattelite(dt_from='2007-12-08T00:00:00', dt_to='2007-12-12T00:00:00')
sat = Sattelite(dt_from='2007-12-10T00:00:00', dt_to='2007-12-10T00:00:00')
chaos = CHAOS7()

sat_data = sat.import_CHAMP_from_cdf(delta=1)   # dt, theta, phi, r, N, E, C, F
print('delta:', sat_data[1, 0]-sat_data[0, 0])

# select time
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 0, 00), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 12, 00))]
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 0), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 0))]   # 6 utc  -  8 utc
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 27), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 6, 44))]  # 1st
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 4, 55), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 5, 14))]  # 2nd
sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 7, 58), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 15))]  # 3

#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 7, 56), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 5))]  # 7
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 3, 23), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 3, 29))]

sat_datetime = sat_data[:, 0]
sat_pos = sat_data[:, 1:4]
sat_value = sat_data[:, 4:]
sat_value = chaos.get_sat_and_chaos_diff(sat_datetime, sat_pos, sat_value)


sat_label = 'CHAMP_dN'
channel = 0     # N=0, E=1, C=2, F=3

sat_value = sat_value[:, channel]   # !!!!!!!!!!
#sat_value = sat.calc_FAC(sat_datetime, sat_pos, sat_value)    # !!!!!!!!!!!!!!!!!!!!

fig, ax = plt.subplots()
#ax.plot(sat_value)
#plt.show()
print('VALUE min:%s max:%s'%(np.min(np.array([x for x in sat_value if not np.isnan(x)])), np.max(np.array([x for x in sat_value if not np.isnan(x)]))))

lonx = lambda x: (x + 180) % 360 - 180      # 0 360 to -180 180
# [lon_min, lon_max, lat_min, lat_max]
#proj_extend_loc = [-170, -14, 50, 75]   # full 10/12/07
proj_extend_loc = [-179, 179, 55, 90]   # ortho n
#proj_extend_loc = [-170, 30, 50, 90]   # 1st
#proj_extend_loc = None

#rad_loc = [90-25, lonx(304)]
#rad_loc = [90-29, lonx(274)]
#rad_loc = [90-18, lonx(288)]
#rad_loc = [90-22, lonx(257)]
#rad_loc = [62.48, -102.75]
rad_loc = None


#central_lon = None
central_lon = get_lon_from_LT(sat_datetime[int((len(sat_datetime)-1)/2)])
print('central lon', central_lon, sat_datetime[int((len(sat_datetime)-1)/2)], )
(status, im1) = get_proj_image(sat_label=sat_label, sat_datetime=sat_datetime, sat_pos=sat_pos, sat_value=sat_value,
                               legend_label=sat_label.rsplit('_')[1], sat_poly_loc=proj_extend_loc, sat_rad_loc=None, sat_rad_deg=5,
                               #legend_label='F', sat_poly_loc=None, sat_rad_loc=rad_loc, sat_rad_deg=15,
                               proj_type='ortho_n', draw_IGRFvector_diff=False,
                               #proj_type='miller', draw_IGRFvector_diff=False,
                               draw_CHAOSvector_diff=False, mag_grid_coord=False,
                               lt_grid_coord=True, obs=obs, central_lon=central_lon)

(status, im2) = get_single_plot(sat_label, sat_datetime, sat_pos, sat_value)
out_image = Image.fromarray((np.array(im1)).astype(np.uint8))
#out_image = single_image(im4)
filename = sat_label
print('saved as %s' % RESULT_DIR + '/CHAMP/%s.jpg' % filename)
out_image.save(RESULT_DIR + '/CHAMP/%s.jpg' % filename, dpi=(400, 400))

Image.fromarray((np.array(im2)).astype(np.uint8)).save(RESULT_DIR + '/CHAMP/%s_plot.jpg' % filename)
