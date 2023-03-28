import datetime

from champ.sattelite import Sattelite
import matplotlib.pyplot as plt
import numpy as np
import sys
import timezonefinder
from tzwhere import tzwhere
from tools import get_local_time
import matplotlib.gridspec as gridspec

def draw_LT_chart():
    day_tz = []
    night_tz = []
    for i, cd in enumerate(gd_sat_coord):
        lat, lon, alt = cd
        timezone_str = tf.certain_timezone_at(lat=lat, lng=lon)
        #timezone_str = tzwhere.tzNameAt(lat, lon) # Seville coordinates

        if index_day[i]:
            day_tz.append(timezone_str)
        elif index_night[i]:
            night_tz.append(timezone_str)
        else:
            print(i, lat, lon, sat_time[i], 'not in DAY NIGHT idx')



    tz_labels_day = np.unique(day_tz)
    tz_labels_night = np.unique(night_tz)
    means_tz_day = np.zeros(len(tz_labels_day))
    means_tz_night = np.zeros(len(tz_labels_night))

    for k, label in enumerate(tz_labels_day):
        means_tz_day[k] = day_tz.count(label)
    for k, label in enumerate(tz_labels_night):
        means_tz_night[k] = night_tz.count(label)

    tz_labels_day = tz_labels_day[np.where(means_tz_day>25)]
    means_tz_day = means_tz_day[np.where(means_tz_day>25)]

    tz_labels_night = tz_labels_night[np.where(means_tz_night > 25)]
    means_tz_night = means_tz_night[np.where(means_tz_night > 25)]

    fig, axs = plt.subplots(2)
    width = 0.35  # the width of the bars

    xD = np.arange(len(tz_labels_day))  # the y_label locations
    axs[0].bar(xD, means_tz_day, width, label='Day')
    axs[0].set_xticks(xD)
    axs[0].set_xticklabels(tz_labels_day, rotation=40)
    axs[0].tick_params(axis='x', labelsize=7)
    axs[0].set_title('Day LT timezones')

    xN = np.arange(len(tz_labels_night))  # the y_label locations
    axs[1].bar(xN, means_tz_night, width, label='Night')
    axs[1].set_xticks(xN)
    axs[1].set_xticklabels(tz_labels_night, rotation=40)
    axs[1].tick_params(axis='x', labelsize=7)
    axs[1].set_title('Night LT timezones')

    plt.show()

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

np.set_printoptions(threshold=sys.maxsize)
tf = timezonefinder.TimezoneFinder()
tzwhere = tzwhere.tzwhere()


PLOT_RANGE_DELTA = 1.5   # hour      0 eq False
DELTA_REDUCTION = 3

IGRF_DIFF = False   # if True y= satF - igrfF
IGRF_VALUE = True  # if True y= igrfF

sat = Sattelite(dt_from='1970-03-09T00:00:00', dt_to='1970-03-09T00:00:00')
sat_data = sat.import_ogo6_from_cdf(delta=DELTA_REDUCTION)   # t, theta, phi, r, F


print('delta:', sat_data[1, 0]-sat_data[0, 0])

idx_start, idx_stop = get_start_stop_idx(time_array=sat_data[:, 0],
                                         dt_from=datetime.datetime(1970, 3, 9, 3, 50, 0),
                                         dt_to=datetime.datetime(1970, 3, 9, 23, 20, 0))

sat_time = sat_data[idx_start:idx_stop, 0]
sat_value = sat_data[idx_start:idx_stop, 4]
gc_sat_coord = sat_data[idx_start:idx_stop, 1:4]
#gd_sat_coord = sat.geocentric_to_geodetic(gc_sat_coord)
gd_sat_coord = sat.geocentric_to_geodetic(gc_sat_coord, gd_alt=True)    # gd_alt=True -> alt above sea lvl

for i, lat in enumerate(gd_sat_coord[:, 0]):
    print(i, lat, sat_time[i].time(), 'LT:', get_local_time(sat_time[i], gd_sat_coord[i, 1]))
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



"""# print lat lon utc lt
for i, lat in enumerate(gd_sat_coord[:, 0]):
    local_time = get_local_time(sat_time[i], gd_sat_coord[i, 1])
    if index_day[i]:
        print(i, 'lat:%.2f' % lat, 'lon:%.2f' % gd_sat_coord[i, 1], 'UTC:', sat_time[i].time(), 'LT:', local_time, '(day)')
    elif index_night[i]:
        print(i, 'lat:%.2f' % lat, 'lon:%.2f' % gd_sat_coord[i, 1], 'UTC:', sat_time[i].time(), 'LT:', local_time, '(night)')
    else:
        print(i, lat, 'wrong!', local_time)
        
print('---------------')
"""
#draw_LT_chart()



#print('RMSE of F: {:.5f} nT'.format(np.std(sat_value-igrf_data[:, 3])))




fig = plt.figure()
#plt.rc('grid', linestyle="--", color='black')
gs = gridspec.GridSpec(1, 2, hspace=0, wspace=0)
ax1, ax2 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
# axs = host_subplot(111, axes_class=axisartist.Axes)

ax1.set_xlim([-90, 90])
ax2.set_xlim([90, -90])


#ax1.set_ylim([15000, 50000])
#ax2.set_ylim([15000, 50000])

if IGRF_DIFF:
    igrf_data = sat.get_igrf13_model_data(date=sat_time, geodetic_coords=gd_sat_coord)  # N E C F
    XX, YY = gd_sat_coord, sat_value - igrf_data[:, 3]  # satF - igrfF   (idxF == 3)
    y_min, y_max = -1*np.max(np.abs(YY)), np.max(np.abs(YY))
    title, y_label = 'OGO-IGRF13', 'OGO-IGRF13 $\\mathrm{d} F$ (nT)'
elif IGRF_VALUE:
    igrf_data = sat.get_igrf13_model_data(date=sat_time, geodetic_coords=gd_sat_coord)  # N E C F
    XX, YY = gd_sat_coord, igrf_data[:, 3]
    y_min, y_max = np.min(YY), np.max(YY)
    title, y_label = 'IGRF13', 'IGRF13 F (nT)'
else:
    XX, YY = gd_sat_coord, sat_value
    y_min, y_max = np.min(YY), np.max(YY)
    title, y_label = 'OGO-6', 'OGO-6 F (nT)'

index_day, index_night = sat.get_DayNight_idx(dt=sat_time, phi=gd_sat_coord[:, 1])
ax1.set_ylim([y_min, y_max])
ax2.set_ylim([y_min, y_max])

last_idx = 0
if PLOT_RANGE_DELTA > 0:
    for i, dt in enumerate(sat_time):
        if (dt - sat_time[last_idx] > datetime.timedelta(hours=PLOT_RANGE_DELTA)) or (i == len(sat_time) - 1):
            range_idx_day, range_idx_night = index_day[last_idx:i], index_night[last_idx:i]
            range_sat_time = sat_time[last_idx:i]
            range_sat_coord = XX[last_idx:i]
            range_sat_value = YY[last_idx:i]

            X1, Y1 = range_sat_coord[range_idx_day, 0], range_sat_value[range_idx_day]
            X2, Y2 = range_sat_coord[range_idx_night, 0], range_sat_value[range_idx_night]

            #sc = ax1.scatter(X1, Y1, s=1, label='%s from %s to %s' % (y_label,
            sc = ax1.scatter(X1, Y1, s=1)
            ax2.scatter(X2, Y2, s=1, c=sc.get_facecolors()[0].tolist())
            last_idx = i
else:
    X1, Y1 = XX[index_day, 0], YY[index_day]
    X2, Y2 = XX[index_night, 0], YY[index_night]

    sc = ax1.scatter(X1, Y1, s=0.75, label='%s from %s to %s' % (y_label,
                                                                 sat_time[0].strftime('%Y-%m-%d %H:%M:%S'), sat_time[-1].strftime('%Y-%m-%d %H:%M:%S')))
    ax2.scatter(X2, Y2, s=0.75, c=sc.get_facecolors()[0].tolist())

#ax1.scatter(gd_sat_coord[index_day, 0], dF_day, s=0.4, c='r', y_label='dayside')
#ax2.scatter(gd_sat_coord[index_night, 0], dF_night, s=0.4, c='b', y_label='nightside')



#ax1.plot(gd_sat_coord[index_day, 0], dF_day, c='r', y_label='dayside')
#ax2.plot(gd_sat_coord[index_night, 0], dF_night, c='b', y_label='nightside')

"""latValue_day_array, latValue_night_array = mean_lat()
host.plot(latValue_day_array[:, 0], latValue_day_array[:, 2], c='r', y_label='dayside')
ax2.plot(latValue_night_array[:, 0], latValue_night_array[:, 2], c='b', y_label='nightside')"""

"""
plt.scatter(theta_gsm[index_day], dF_day, s=0.3, c='r', y_label='dayside')
plt.scatter(theta_gsm[index_night], dF_night, s=0.3, c='b', y_label='nightside')"""


ax1.set_xlabel('dayside Latitude ($^\\circ$)')
ax2.set_xlabel('nightside Latitude ($^\\circ$)')
ax1.set_ylabel(y_label)
#ax1.set_ylabel('OGO-6 F (nT)')
ax1.legend(loc=3)
#ax2.legend(loc=3)
ax1.grid(linestyle='--')
ax2.grid(linestyle='--')


fig.suptitle('%s from %s to %s' % (title, sat_time[0].strftime('%Y-%m-%d %H:%M:%S'), sat_time[-1].strftime('%Y-%m-%d %H:%M:%S')))
plt.show()