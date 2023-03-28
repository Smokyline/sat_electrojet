
from sattelite import Sattelite
import matplotlib.pyplot as plt
import numpy as np
import sys
from dateutil import tz
from tools import *
from sun_position import *
from settings import RESULT_DIR
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from chaos7_model.chaos_model import CHAOS7
import datetime

def eucl_range_foo(xy, data):
    eucl_array = np.zeros((1, len(data))).astype(float)
    for n, d in enumerate(xy):
        eucl_array += np.array(np.power((d - data[:, n]), 2)).astype(float)
    eucl_array = np.sqrt(eucl_array[0])
    return eucl_array



sat = Sattelite(dt_from='2007-12-08T00:00:00', dt_to='2007-12-12T00:00:00')
#sat = Sattelite(dt_from='2007-12-10T00:00:00', dt_to='2007-12-10T00:00:00')
chaos = CHAOS7()

sat_data = sat.import_CHAMP_from_cdf(delta=1)   # dt, theta, phi, r, N, E, C, F
print('delta:', sat_data[1, 0]-sat_data[0, 0])

# select time
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 0, 00), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 12, 00))]
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 8, 0, 00), sat_data[:, 0] <= datetime.datetime(2007, 12, 8, 12, 00))]
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 0), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 0))]   # 6 utc  -  8 utc

sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 27), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 6, 44))]  # 1st
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 36), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 6, 43))]  # 1st vortex
auroral_zones = {
    'diffuse_auroral_zone':[[datetime.datetime(2007, 12, 10, 6, 30, 4), datetime.datetime(2007, 12, 10, 6, 32, 5)], [datetime.datetime(2007, 12, 10, 6, 38, 40), datetime.datetime(2007, 12, 10, 6, 39, 36)]],
    'auroral_oval_precipt':[[datetime.datetime(2007, 12, 10, 6, 32, 34), datetime.datetime(2007, 12, 10, 6, 33, 24)], [datetime.datetime(2007, 12, 10, 6, 36, 28), datetime.datetime(2007, 12, 10, 6, 38, 9)]],
    'soft_diff_precipt':[[datetime.datetime(2007, 12, 10, 6, 33, 24), datetime.datetime(2007, 12, 10, 6, 36, 28)],]
}

#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 4, 55), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 5, 14))]  # 2nd

#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 7, 58), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 15))]  # 3
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 7, 59), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 7))]  # 3 vortex
"""auroral_zones = {
    'diffuse_auroral_zone':[[datetime.datetime(2007, 12, 10, 8, 0, 34), datetime.datetime(2007, 12, 10, 8, 2, 42)], [datetime.datetime(2007, 12, 10, 8, 9, 56), datetime.datetime(2007, 12, 10, 8, 10, 40)]],
    'auroral_oval_precipt':[[datetime.datetime(2007, 12, 10, 8, 3, 19), datetime.datetime(2007, 12, 10, 8, 4, 6)], [datetime.datetime(2007, 12, 10, 8, 7, 22), datetime.datetime(2007, 12, 10, 8, 9, 12)]],
    'soft_diff_precipt':[[datetime.datetime(2007, 12, 10, 8, 4, 6), datetime.datetime(2007, 12, 10, 8, 5, 7)], [datetime.datetime(2007, 12, 10, 8, 7, 6), datetime.datetime(2007, 12, 10, 8, 7, 22)]]
}
"""

#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 11, 00), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 11, 20))]  # 4
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 11, 1), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 11, 12))]  # 4 vortex
"""auroral_zones = {
    'diffuse_auroral_zone':[[datetime.datetime(2007, 12, 10, 11, 1), datetime.datetime(2007, 12, 10, 11, 2, 1)], [datetime.datetime(2007, 12, 10, 11, 13, 12), datetime.datetime(2007, 12, 10, 11, 14, 11)]],
    'auroral_oval_precipt':[[datetime.datetime(2007, 12, 10, 11, 2, 31), datetime.datetime(2007, 12, 10, 11, 4, 3)], [datetime.datetime(2007, 12, 10, 11, 11, 59), datetime.datetime(2007, 12, 10, 11, 13, 8)]],
    'soft_diff_precipt':[[datetime.datetime(2007, 12, 10, 11, 4, 3), datetime.datetime(2007, 12, 10, 11, 4, 18)], [datetime.datetime(2007, 12, 10, 11, 11, 54), datetime.datetime(2007, 12, 10, 11, 11, 59)]]
}"""

sat_datetime = sat_data[:, 0]
sat_pos = sat_data[:, 1:4]
sat_value = sat_data[:, 4:]
sat_value = chaos.get_sat_and_chaos_diff(sat_datetime, sat_pos, sat_value)


#sat_label = 'CHAMP_dF'
sat_label = 'CHAMP_fac'
channel = 3   # N=0, E=1, C=2, F=3


if 'fac' in sat_label:
    sat_value = sat.calc_FAC(sat_datetime, sat_pos, sat_value)
else:
    sat_value = sat_value[:, channel]   # !!!!!!!!!!

fig, axs = plt.subplots(ncols=1, nrows=1, figsize=(15, 5),
                        constrained_layout=True)


host = axs
host.set_clip_on(False)
ax2 = host.twiny()


#   time ticks

str_ticks = [dt.strftime('%H:%M:%S') for dt in sat_datetime]
date_ticks = np.array([[x, label] for x, label in zip(np.arange(len(sat_datetime)), str_ticks)])
host.plot(np.arange(len(sat_datetime)), sat_value, c='r', label=sat_label)
sat_line = Line2D([0], [0], label=sat_label, color='r')


solarcoord_pos = []
for i in range(len(sat_datetime)):
    #dt = decode_str_dt_param(date_list[i] + 'T' + time_list[i])
    dt = sat_datetime[i]
    lon = sat_pos[i, 1]
    # solar_lon = get_solar_coord(dt, lon)
    solar_lat, solar_lon = sun_pos(dt)
    # solar_lt = np.rint(solar_lon/15)
    lt = get_local_time(dt, lon)
    # solarcoord_pos.append('%sh %.1f'%(lt.hour, solar_lon))
    solarcoord_pos.append('%s %.1f' % (lt.strftime('%H:%M:%S'), 90. - solar_lat))
coord_pos = np.array([str(lat) + ' ' + str(lon) for lat, lon in np.round(np.array(sat_pos[:, :2]).astype(float), 2)])


coord_ticks = []
for mag, geo in zip(solarcoord_pos, coord_pos):
    coord_ticks.append(str(mag))
    coord_ticks.append('\n\n' + str(geo))
coord_ticks = np.array(coord_ticks).astype(str)
coord_ticks = np.array([''.join(x) for x in zip(coord_ticks[0::2], coord_ticks[1::2])])
coordtick_locations = np.arange(len(sat_pos))


if len(date_ticks) >= 12:
    tick_count = 9
    coord_ticks_arange = np.arange(0, len(coord_ticks), int(len(coord_ticks) / tick_count))  # top xlabels tick
    date_ticks = date_ticks[coord_ticks_arange]
else:
    coord_ticks_arange = np.arange(0, len(coord_ticks))

auroral_colors  = {
    'diffuse_auroral_zone':'b',
    'auroral_oval_precipt':'g',
    'soft_diff_precipt':'m'
}
auroral_zones_pos  = {
    'diffuse_auroral_zone':10,
    'auroral_oval_precipt':8,
    'soft_diff_precipt':6
}
nonnan_value = [x for x in sat_value if not np.isnan(x)]
for auroral_zone, time_ranges in auroral_zones.items():
    for t_range in time_ranges:
        arg_from = np.argmin(np.abs(sat_datetime - t_range[0]))
        arg_to = np.argmin(np.abs(sat_datetime - t_range[1]))
        host.vlines(x=arg_from, ymin=np.min(nonnan_value), ymax=np.max(nonnan_value),
                    color=auroral_colors[auroral_zone], linestyle='--',
                    alpha=.5, linewidths=1)
        host.vlines(x=arg_to, ymin=np.min(nonnan_value), ymax=np.max(nonnan_value),
                    color=auroral_colors[auroral_zone], linestyle='--',
                    alpha=.5, linewidths=1)
        window_idx = arg_from + int((arg_to - arg_from) / 2)
        y_amplitude = abs(np.max(nonnan_value) - np.min(nonnan_value))
        center_y_axis = np.min(nonnan_value) - y_amplitude / auroral_zones_pos[auroral_zone]
        style = dict(size=7, color='k')
        #host.text(window_idx, center_y_axis, auroral_zone, ha='center', **style)

        p2 = mpatches.Rectangle((arg_from, center_y_axis), width=abs(arg_to - arg_from),
                                height=y_amplitude / 25, color=auroral_colors[auroral_zone], clip_on=False, )
        host.add_patch(p2)
diffuse_patch = mpatches.Patch(color='b', label='Diffuse auroral zone')
auroral_patch = mpatches.Patch(color='g', label='Auroral oval precipitation')
soft_patch = mpatches.Patch(color='m', label='Soft diffuse precipitation')
plt.legend(handles=[sat_line, diffuse_patch, auroral_patch, soft_patch], loc='lower right')

if 'fac' in sat_label:
    host.set_ylabel(sat_label+ ' %sA/m²' % chr(956))
else:
    host.set_ylabel(sat_label + ' nT')

host.set_xticks(np.array(date_ticks[:, 0]).astype(int))
host.set_xticklabels(date_ticks[:, 1], rotation=40,)
#host.tick_params(axis='x', which='major', pad=20)
ax2.set_xlim(host.get_xlim())
ax2.set_xticks(coordtick_locations[coord_ticks_arange])
ax2.set_xticklabels(coord_ticks[coord_ticks_arange])
ax2.tick_params(axis='x', labelsize=8)
ax2.annotate("LT sunColat\n\nLat Lon", xy=(-0.03, 1.037), xycoords=ax2.transAxes, size=8)

# https://stackoverflow.com/questions/20532614/multiple-lines-of-x-tick-labels-in-matplotlib

host.grid()
host.xaxis.grid(False)
ax2.grid()

if sat_datetime[0].date() != sat_datetime[-1].date():  # first date != second date
    dt_from, dt_to = sat_datetime[0].date(),  sat_datetime[-1].date()
    fig.suptitle('from %s to %s' % (dt_from, dt_to), fontsize=16)
else:
    fig.suptitle(sat_datetime[0].date(), fontsize=16)

plt.grid(True)
plt.savefig(RESULT_DIR + '/auroral/%s.jpg' % sat_label, dpi=400)
#plt.show()
