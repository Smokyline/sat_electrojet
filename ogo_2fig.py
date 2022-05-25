import datetime

from sattelite import Sattelite
import matplotlib.pyplot as plt
import numpy as np
import sys

from dateutil import tz
from tools import get_local_time
import matplotlib.gridspec as gridspec
import pandas as pd
import math
import time
import matplotlib.ticker as plticker

np.set_printoptions(threshold=sys.maxsize)

PLOT_RANGE_DELTA = 0  # hour      0 eq False
DELTA_REDUCTION = 3

IGRF_DIFF = False  # if True y= satF - igrfF
IGRF_VALUE = False  # if True y= igrfF

sat = Sattelite(dt_from='1970-03-8T00:00:00', dt_to='1970-03-9T00:00:00')
sat_data = sat.import_ogo6_from_cdf(delta=DELTA_REDUCTION)  # t, theta, phi, r, F
print('delta:', sat_data[1, 0] - sat_data[0, 0])

Kp_DATA = sat.import_kp(filename='Kp_1970-03-06_1970-03-11_D.dat')  # dt, value
Dst_DATA = sat.import_dst(filename='Dst_1970-03-06_1970-03-11_D.dat')
AE_DATA = sat.import_ae(filename='AEAUALAO_1970-03-06_1970-03-11h_D.dat')
idx_start, idx_stop = sat.get_start_stop_idx(time_array=sat_data[:, 0],
                                         #dt_from=datetime.datetime(1970, 3, 6, 5, 24, 44), dt_to=datetime.datetime(1970, 3, 6, 7, 4, 2)) # t1
                                         #dt_from=datetime.datetime(1970, 3, 10, 12, 3, 1), dt_to=datetime.datetime(1970, 3, 10, 13, 42, 19)) # t2
                                         #dt_from=datetime.datetime(1970, 3, 8, 13, 41, 56), dt_to=datetime.datetime(1970, 3, 8, 15, 20, 16)) # t3
                                         #dt_from=datetime.datetime(1970, 3, 8, 22, 3, 0), dt_to=datetime.datetime(1970, 3, 8, 23, 37, 52)) # t4
                                         dt_from=datetime.datetime(1970, 3, 8, 23, 37, 53), dt_to=datetime.datetime(1970, 3, 9, 1, 17, 11)) # t741
                                         #dt_from=datetime.datetime(1970, 3, 10, 0, 27, 46), dt_to=datetime.datetime(1970, 3, 10, 2, 7, 2)) # t761
                                         #dt_from=datetime.datetime(1970, 3, 10, 2, 7, 10), dt_to=datetime.datetime(1970, 3, 10, 3, 46, 19)) # t762

sat_time = sat_data[idx_start:idx_stop, 0]
sat_value = sat_data[idx_start:idx_stop, 4]
gc_sat_coord = sat_data[idx_start:idx_stop, 1:4]
# gd_sat_coord = sat.geocentric_to_geodetic(gc_sat_coord)
gd_sat_coord = sat.geocentric_to_geodetic(gc_sat_coord, gd_alt=True)  # gd_alt=True -> alt above sea lvl

for i, lat in enumerate(gd_sat_coord[:, 0]):
    print(i, np.round(lat, 5), np.round(gd_sat_coord[i, 1], 5), sat_time[i].time(), 'LT:', get_local_time(sat_time[i], gd_sat_coord[i, 1]))
    if i > 20000:
        break
    pass
print('from:', sat_time[np.argmin(gd_sat_coord[:500, 0])], 'to:', sat_time[-500:][np.argmin(gd_sat_coord[-500:, 0])])


if IGRF_DIFF:
    igrf_data = sat.get_igrf13_model_data(date=sat_time, geodetic_coords=gd_sat_coord)  # N E C F
    XX, YY = gd_sat_coord, sat_value - igrf_data[:, 3]  # satF - igrfF   (idxF == 3)
    y_min, y_max = -1 * np.max(np.abs(YY)), np.max(np.abs(YY))
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
index_left, index_right, index_sep = sat.get_latitudeSep_idx(theta=XX[:, 0])



fig = plt.figure()
# plt.rc('grid', linestyle="--", color='black')
gs = gridspec.GridSpec(1, 2, hspace=0, wspace=0)
ax1, ax2 = plt.subplot(gs[0, 0]), plt.subplot(gs[0, 1])
ax1.zorder = 0
ax2.zorder = 0


ax1.set_xlim([-90, 90])
ax2.set_xlim([90, -90])
ax1.spines['right'].set_visible(False)
ax2.spines['left'].set_visible(False)


ax1.set_ylim([y_min, y_max])
ax2.set_ylim([y_min, y_max])
ax1_time = ax1.twiny()
ax2_time = ax2.twiny()
ax1_time.spines['right'].set_visible(False)
ax2_time.spines['left'].set_visible(False)

#ax2_ae.axis["right"] = ax2_ae.new_fixed_axis(loc="right", offset=(90, 0))
ax1.set_xlim([np.min(XX[index_left, 0]), np.max(XX[index_left, 0])])
ax2.set_xlim([np.max(XX[index_right, 0]), np.min(XX[index_right, 0])])


# sat time with empty to full
sat_time_full, sat_lon_full = sat.get_full_dt_lon_array(sat_time, gd_sat_coord)



#   LEFT RIGHT IDX OF TIME
def draw_timeticks(ax, dtime_array_full, half_index_array, DTIME_TICKS_STEP, del_last=False):
    print(len(dtime_array_full))
    dtime_array = dtime_array_full[half_index_array]
    halfTicks = np.arange(len(dtime_array))[::int(len(dtime_array)/DTIME_TICKS_STEP)]
    halfLabels = dtime_array[::int(len(dtime_array)/DTIME_TICKS_STEP)]
    if del_last:
        halfLabels = halfLabels[0:-1]
    halfLabels_all = []
    for i, dtime in enumerate(halfLabels):
        dtimestr = dtime.strftime("%H:%M:%S %d/%m")
        localdtstr = get_local_time(dtime, sat_lon_full[half_index_array][::int(len(dtime_array)/DTIME_TICKS_STEP)][i])
        halfLabels_all.append('UT: %s\nLT: %s' % (dtimestr, localdtstr))
    ax.set_xticks(halfTicks)
    ax.set_xticklabels(halfLabels_all, rotation=0)
    ax.set_xlim([halfTicks[0], halfTicks[-1]])

DTIME_TICKS_STEP = 5
index_dtLeft, index_dtRight, index_dtSepar  = sat.get_datetimeSep_idx(dtime_list=sat_time_full, sep_time=sat_time[index_sep])
print(sat_time_full[index_dtLeft][0], sat_time_full[index_dtLeft][-1])
print(sat_time_full[index_dtRight][0], sat_time_full[index_dtRight][-1])
draw_timeticks(ax1_time, sat_time_full, index_dtLeft, DTIME_TICKS_STEP, del_last=True)
draw_timeticks(ax2_time, sat_time_full, index_dtRight, DTIME_TICKS_STEP, del_last=False)



# KP DST AE DRAW
ax1_kp = ax1_time.twinx()
ax1_kp.set_ylim([0, 10])

ax1_kp.spines['right'].set_visible(False)
ax1_kp.tick_params(right=False)
ax1_kp.tick_params(labelright=False)
ax1_dst = ax1_time.twinx()
ax1_dst.spines['right'].set_visible(False)
ax1_dst.tick_params(right=False)
ax1_dst.tick_params(labelright=False)
ax1_ae = ax1_time.twinx()
ax1_ae.spines['right'].set_visible(False)
ax1_ae.tick_params(right=False)
ax1_ae.tick_params(labelright=False)

ax2_kp = ax2_time.twinx()
ax2_kp.set_ylim([0, 10])
ax2_dst = ax2_time.twinx()
ax2_ae = ax2_time.twinx()
ax2_kp.spines['left'].set_visible(False)
ax2_dst.spines['left'].set_visible(False)
ax2_ae.spines['left'].set_visible(False)


ax2_dst.spines['right'].set_position(('axes', 1.1))
ax2_ae.spines['right'].set_position(('axes', 1.1875))

#kp_time_selected_idx = [time_array_idx(i), sat_fill_time_idx(i)]
kp_time_selected_idx = sat.search_time_array_idx_in_fullTime_array(time_array=Kp_DATA[:, 0], sat_full_time=sat_time_full)
dst_time_selected_idx = sat.search_time_array_idx_in_fullTime_array(time_array=Dst_DATA[:, 0], sat_full_time=sat_time_full)
ae_time_selected_idx = sat.search_time_array_idx_in_fullTime_array(time_array=AE_DATA[:, 0], sat_full_time=sat_time_full)

kp_time_selected_idx_L, kp_time_selected_idx_R = sat.separate_timeIdx(array=kp_time_selected_idx, index_array=kp_time_selected_idx[:, 1], separator_idx=index_dtSepar)
dst_time_selected_L, dst_time_selected_R = sat.separate_timeIdx(array=dst_time_selected_idx, index_array=dst_time_selected_idx[:, 1], separator_idx=index_dtSepar)
ae_time_selected_L, ae_time_selected_R = sat.separate_timeIdx(array=ae_time_selected_idx, index_array=ae_time_selected_idx[:, 1], separator_idx=index_dtSepar)


dst_selected = Dst_DATA[dst_time_selected_idx[:, 0], 1]
ae_selected = AE_DATA[ae_time_selected_idx[:, 0], 1]
dst_selected_min, dst_selected_max = np.min(dst_selected)-np.ptp(dst_selected)/10, np.max(dst_selected)+np.ptp(dst_selected)/10
ae_selected_min, ae_selected_max = np.min(ae_selected)-np.ptp(ae_selected)/10, np.max(ae_selected)+np.ptp(ae_selected)/10
ax1_dst.set_ylim([dst_selected_min, dst_selected_max])
ax2_dst.set_ylim([dst_selected_min, dst_selected_max])
ax1_ae.set_ylim([ae_selected_min, ae_selected_max])
ax2_ae.set_ylim([ae_selected_min, ae_selected_max])


def draw_solar_indices(ax, value, full_time_idx, label=None, right_fig=False):
    legend = {
        'Kp': ['r', 'o'],
        'Dst': ['darkorange', 'X'],
        'AE': ['g', 'D']
    }
    ax.scatter(full_time_idx, value, label=label, color=legend[label][0], marker=legend[label][1], s=50)
    for x, y, in zip(full_time_idx, value):
        ax.annotate(label, (x, y))
    if right_fig:
        ax.set_ylabel(label)
        ax.spines['right'].set_color(legend[label][0])


draw_solar_indices(ax1_dst, Dst_DATA[dst_time_selected_L[:, 0], 1], dst_time_selected_L[:, 1], label='Dst')
draw_solar_indices(ax1_ae, AE_DATA[ae_time_selected_L[:, 0], 1], ae_time_selected_L[:, 1], label='AE')
draw_solar_indices(ax1_kp, Kp_DATA[kp_time_selected_idx_L[:, 0], 1], kp_time_selected_idx_L[:, 1], label='Kp')

draw_solar_indices(ax2_dst, Dst_DATA[dst_time_selected_R[:, 0], 1], dst_time_selected_R[:, 1], label='Dst', right_fig=True)
draw_solar_indices(ax2_ae, AE_DATA[ae_time_selected_R[:, 0], 1], ae_time_selected_R[:, 1], label='AE', right_fig=True)
draw_solar_indices(ax2_kp, Kp_DATA[kp_time_selected_idx_R[:, 0], 1], kp_time_selected_idx_R[:, 1], label='Kp', right_fig=True)


# PLOT VLINES OF DAYSIDE SWITCH
def draw_dayside_switch_vlines(ax, switch_array):
    ax.vlines(switch_array, 0, 1, transform=ax.get_xaxis_transform(), colors='m', linestyles='dashdot')

current_side = ''
day_nightside_switch_array = [[], []]
for i in range(len(sat_lon_full)):
    local_time = get_local_time(sat_time_full[i], sat_lon_full[i])
    if local_time >= datetime.time(6, 0) and local_time < datetime.time(18, 0):
        lt_str = 'day'
    else:
        lt_str = 'night'
    if i == 0:
        current_side = lt_str
    else:
        if lt_str != current_side:
            if i < index_dtSepar:
                day_nightside_switch_array[0].append(i)
            else:
                day_nightside_switch_array[1].append(i-index_dtSepar)
            #print(sat_time_full[i], 'lt:', local_time)
            #print('bf', sat_time_full[i-1], 'lt:', get_local_time(sat_time_full[i-1], sat_lon_full[i-1]))
            current_side = lt_str
draw_dayside_switch_vlines(ax1_time, day_nightside_switch_array[0])
draw_dayside_switch_vlines(ax2_time, day_nightside_switch_array[1])


#   PLOT SATELLITE VALUE
last_idx = 0
if PLOT_RANGE_DELTA > 0:
    for i, dt in enumerate(sat_time):
        if (dt - sat_time[last_idx] > datetime.timedelta(hours=PLOT_RANGE_DELTA)) or (i == len(sat_time) - 1):
            range_idx_day, range_idx_night = index_left[last_idx:i], index_right[last_idx:i]
            range_sat_time = sat_time[last_idx:i]
            range_sat_coord = XX[last_idx:i]
            range_sat_value = YY[last_idx:i]

            X1, Y1 = range_sat_coord[range_idx_day, 0], range_sat_value[range_idx_day]
            X2, Y2 = range_sat_coord[range_idx_night, 0], range_sat_value[range_idx_night]

            # sc = ax1.scatter(X1, Y1, s=1, label='%s from %s to %s' % (y_label,
            sc = ax1.scatter(X1, Y1, s=1)
            ax2.scatter(X2, Y2, s=1, c=sc.get_facecolors()[0].tolist())
            last_idx = i
else:
    X1, Y1 = XX[index_left, 0], YY[index_left]
    X2, Y2 = XX[index_right, 0], YY[index_right]


    sc = ax1.scatter(X1, Y1, s=1, label='%s from %s to %s' % (y_label,
                                                                 sat_time[0].strftime('%Y-%m-%d %H:%M:%S'),
                                                                 sat_time[-1].strftime('%Y-%m-%d %H:%M:%S')), alpha=0.75)
    ax2.scatter(X2, Y2, s=1, c=sc.get_facecolors()[0].tolist())


ax1.set_xlabel('1st half latitude ($^\\circ$)')
ax2.set_xlabel('2nd half latitude ($^\\circ$)')
ax1.set_ylabel(y_label)
# ax1.set_ylabel('OGO-6 F (nT)')
l = ax1.legend(loc=3)
# ax2.legend(loc=3)
ax1.grid(linestyle='--')
ax2.grid(linestyle='--')
ax2.tick_params(labelleft=False)


fig.suptitle(
    '%s from %s to %s' % (title, sat_time[0].strftime('%Y-%m-%d %H:%M:%S'), sat_time[-1].strftime('%Y-%m-%d %H:%M:%S')))
plt.show()