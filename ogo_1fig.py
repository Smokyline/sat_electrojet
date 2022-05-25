import datetime

from sattelite import Sattelite
import matplotlib.pyplot as plt
import numpy as np
import sys
import timezonefinder, pytz
from tzwhere import tzwhere
from dateutil import tz
from tools import get_local_time

np.set_printoptions(threshold=sys.maxsize)
tf = timezonefinder.TimezoneFinder()
tzwhere = tzwhere.tzwhere()


def mean_lat():
    latValue_day_array = []
    latValue_night_array = []
    delta = 0.5
    for lat in np.arange(-90, 90, delta):
        lat_range = (lat, lat + delta)

        # day
        idx_lat_range = \
            np.where(np.logical_and(gd_sat_coord_day[:, 0] >= lat_range[0], gd_sat_coord_day[:, 0] < lat_range[1]))[
                0]
        if len(idx_lat_range > 0):
            latValue_day_array.append(
                np.array([np.mean(gd_sat_coord_day[idx_lat_range, 0]), np.mean(gd_sat_coord_day[idx_lat_range, 1]),
                          np.mean(dF_day[idx_lat_range])]))

        # night
        idx_lat_range = \
            np.where(np.logical_and(gd_sat_coord_night[:, 0] >= lat_range[0], gd_sat_coord_night[:, 0] < lat_range[1]))[
                0]
        if len(idx_lat_range > 0):
            latValue_night_array.append(
                np.array([np.mean(gd_sat_coord_night[idx_lat_range, 0]), np.mean(gd_sat_coord_night[idx_lat_range, 1]),
                          np.mean(dF_night[idx_lat_range])]))
    latValue_day_array = np.array(latValue_day_array).astype(float)
    latValue_night_array = np.array(latValue_night_array).astype(float)
    return latValue_day_array, latValue_night_array


def get_start_stop_idx(time_array, dt_from, dt_to):
    idx = np.arange(len(time_array))[np.logical_and(time_array >= dt_from, time_array <= dt_to)]
    return idx[0], idx[-1]

sat = Sattelite(dt_from='1970-03-06T00:00:00', dt_to='1970-03-06T00:00:00')
sat_data = sat.import_ogo6_from_cdf(delta=50)   # t, theta, phi, r, F


#idx_start, idx_stop = 1130, 4917 #-81.98712158203125 -81.97816467285156
#idx_start, idx_stop = 1340, 4703 #-70.43208312988281 -70.36573791503906
#idx_start, idx_stop = 14449, 17600 #70.08 69.983
#idx_start, idx_stop = 14371, 17773 #81.983 81.939
idx_start, idx_stop = get_start_stop_idx(time_array=sat_data[:, 0],
                                         dt_from=datetime.datetime(1970, 3, 6, 4, 0, 0),
                                         dt_to=datetime.datetime(1970, 3, 6, 8, 0, 0))


sat_data = sat_data[idx_start:idx_stop]
sat.mjd2000_dt = sat.mjd2000_dt[idx_start:idx_stop]
sat_time = sat_data[:, 0]
sat_value = sat_data[:, 4]
gc_sat_coord = sat_data[:, 1:4]
#gd_sat_coord = sat.geocentric_to_geodetic(gc_sat_coord)
gd_sat_coord = sat.geocentric_to_geodetic(gc_sat_coord, gd_alt=True) # gd_alt=True -> alt above sea lvl

for i, lat in enumerate(sat_data[:, 1]):
    print(i, lat, sat_data[i, 0].time(), 'LT:', get_local_time(sat_data[i, 0], sat_data[i, 2]))
    if i>20000:
        break
    pass

"""for i in range(len(sat_data)):
    print('gc lat:%.2f gc lon:%.2f gc alt:%.2f\ngd lat:%.2f gd lon:%.2f gd alt:%.2f\n' %
          (gc_sat_coord[i, 0], gc_sat_coord[i, 1], gc_sat_coord[i, 2],
           gd_sat_coord[i, 0], gd_sat_coord[i, 1], gd_sat_coord[i, 2],))"""



#for i in range(len(igrf_data)):
#    print('sat F:%s igrf13 F:%s' % (sat_value[i], igrf_data[i, 3]))
#print(mag)

#theta_gsm, phi_gsm, index_day, index_night = sat.get_ThetaPhi_gsm_and_DayNight_idx(theta=90-gd_sat_coord[:, 0], phi=gd_sat_coord[:, 1], mjd2000_time=sat.mjd2000_dt, reference='gsm')
index_day, index_night = sat.get_DayNight_idx(dt=sat_time, phi=gd_sat_coord[:, 1])
gd_sat_coord_day, gd_sat_coord_night = gd_sat_coord[index_day], gd_sat_coord[index_day]
#sat_time_day, sat_time_night = sat.mjd2000_dt[index_day], sat.mjd2000_dt[index_night]
sat_time_day, sat_time_night = sat_time[index_day], sat_time[index_night]


for i, lat in enumerate(gd_sat_coord[:, 0]):
    local_time = get_local_time(sat_time[i], gd_sat_coord[i, 1])
    if index_day[i]:
        print(i, 'lat:%.2f' % lat, 'lon:%.2f' % gd_sat_coord[i, 1], 'UTC:', sat_time[i].time(), 'LT:', local_time, '(day)')
    elif index_night[i]:
        print(i, 'lat:%.2f' % lat, 'lon:%.2f' % gd_sat_coord[i, 1], 'UTC:', sat_time[i].time(), 'LT:', local_time, '(night)')
    else:
        print(i, lat, 'wrong!', local_time)

#draw_LT_chart()

print('---------------')
igrf_data = sat.get_igrf13_model_data(date=sat_time, geodetic_coords=gd_sat_coord)  # N E C F
dF = sat_value-igrf_data[:, 3]
print('RMSE of F: {:.5f} nT'.format(np.std(dF)))
dF_day, dF_night = sat_value[index_day] - igrf_data[index_day, 3], sat_value[index_night] - igrf_data[index_night, 3]



fig = plt.figure(figsize=(10, 7))
# axs = host_subplot(111, axes_class=axisartist.Axes)
host = fig.add_subplot(111)
host.set_aspect('auto')
host.set_xlim([90, -90])

ax2 = host.twiny()
ax2.set_xlim([-90, 90])

#

host.scatter(gd_sat_coord[index_day, 0], dF_day, s=0.5, c='r', label='dayside')
ax2.scatter(gd_sat_coord[index_night, 0], dF_night, s=0.5, c='b', label='nightside')


#host.plot(gd_sat_coord[index_day, 0], dF_day, c='r', y_label='dayside')
#ax2.plot(gd_sat_coord[index_night, 0], dF_night, c='b', y_label='nightside')

"""latValue_day_array, latValue_night_array = mean_lat()
host.plot(latValue_day_array[:, 0], latValue_day_array[:, 2], c='r', y_label='dayside')
ax2.plot(latValue_night_array[:, 0], latValue_night_array[:, 2], c='b', y_label='nightside')"""

"""
plt.scatter(theta_gsm[index_day], dF_day, s=0.3, c='r', y_label='dayside')
plt.scatter(theta_gsm[index_night], dF_night, s=0.3, c='b', y_label='nightside')"""


host.set_xlabel('dayside Latitude ($^\\circ$)')
ax2.set_xlabel('nightside Latitude ($^\\circ$)')
host.set_ylabel('$\\mathrm{d} F$ (nT)')
host.legend(loc=3)
ax2.legend(loc=2)


plt.title('OGO-6 from %s to %s' % (sat_time[0].strftime('%Y-%m-%d %H:%M:%S'), sat_time[-1].strftime('%Y-%m-%d %H:%M:%S')))
plt.show()