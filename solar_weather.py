import datetime
from matplotlib.lines import Line2D
from sattelite import Sattelite
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.dates as mdates
import sys
import matplotlib.colors as mcolors
import matplotlib.ticker as ticker
from dateutil import tz
from tools import get_local_time
from settings import RESULT_DIR

np.set_printoptions(threshold=sys.maxsize)

sat = Sattelite(dt_from='1970-03-8T00:00:00', dt_to='1970-03-9T00:00:00')

#PLASMA_DATA = sat.import_explorer_plasma(filename='explorer33_3min_plasma_mit1970.txt')
OMNI_DATA_FULL = sat.import_omni_data(filename='omni206031970.txt', start_line=15)
"""
Selected parameters:
        1 BX, nT (GSE, GSM)
        2 BY, nT (GSM)
        3 BZ, nT (GSM)
        4 SW Plasma Temperature, K
        5 SW Proton Density, N/cm^3
        6 SW Plasma Speed, km/s
        7 Flow pressure
        8 E elecrtric field
        9 Plasma Beta
"""
dt_from, dt_to = datetime.datetime(1970, 3, 6, 0, 0, 0), datetime.datetime(1970, 3, 11, 0, 0, 0)

while dt_from < dt_to:
    dt1 = dt_from
    dt2 = dt_from + datetime.timedelta(hours=12)
    print(dt1, dt2)
    idx_start, idx_stop = sat.get_start_stop_idx(time_array=OMNI_DATA_FULL[:, 0], dt_from=dt1, dt_to=dt2)
    filename = 'OMNI %s - %s' % (dt1.strftime('%H_%M %Y-%m-%d'), dt2.strftime('%H_%M %Y-%m-%d'))

    OMNI_DATA = OMNI_DATA_FULL[idx_start:idx_stop]
    #datetime_array = pd.date_range(dt_from, dt_to, freq='h')

    BX = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 1])]), 'title': 'BX, nT'}
    BY = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 2])]), 'title': 'BY, nT'}
    BZ = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 3])]), 'title': 'BZ, nT'}

    PLASMA_TEMP = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 4])]),
                   'title': 'SW_plasma_temp, K'}
    PROTON_DENS = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 5])]),
                   'title': 'SW_proton_dens, N/cm^3'}
    PLASMA_SPEED = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 6])]),
                    'title': 'SW_plasma_speed, km/s'}

    FLOW_PRESSURE = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 7])]),
                     'title': 'Flow_pressure, nPa'}
    ELECTRIC_FIELD = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 8])]),
                      'title': 'elecrtric_field, mV/m'}
    PLASMA_BETA = {'data': np.array([[dt, x] for dt, x in zip(OMNI_DATA[:, 0], OMNI_DATA[:, 9])]),
                   'title': 'Plasma_beta'}

    fig, ax = plt.subplots(figsize=(16, 10))

    c = mcolors.ColorConverter().to_rgb

    legend = {
        'BX, nT': c('black'), 'SW_plasma_temp, K': c('#0B2DD6'),
        'Flow_pressure, nPa': c('#01E1F0'), 'BY, nT': c('#00D62E'),
        'SW_proton_dens, N/cm^3': c('#FABE00'), 'elecrtric_field, mV/m': c('#FFFF0D'),
        'BZ, nT': c('#FA4200'), 'SW_plasma_speed, km/s': c('#D60192'),
        'Plasma_beta': c('#8811F0')
    }
    custom_line_legend = []
    custom_title_legend = []
    for sw_array in [BX, BY, BZ, PLASMA_TEMP, PROTON_DENS, PLASMA_SPEED, FLOW_PRESSURE, ELECTRIC_FIELD, PLASMA_BETA]:
        if 'SW_plasma_temp' in sw_array['title']:
            ax1 = ax.twinx()
            ax1.spines['right'].set_position(('axes', 1.00))
            ax1.set_ylabel(sw_array['title'])
            ax1.plot(sw_array['data'][:, 0], sw_array['data'][:, 1], label=sw_array['title'],
                     c=legend[sw_array['title']])
        elif 'SW_plasma_speed' in sw_array['title']:
            ax1 = ax.twinx()
            ax1.spines['right'].set_position(('axes', 1.07))
            ax1.set_ylabel(sw_array['title'])
            ax1.plot(sw_array['data'][:, 0], sw_array['data'][:, 1], label=sw_array['title'],
                     c=legend[sw_array['title']])
        else:
            ax.plot(sw_array['data'][:, 0], sw_array['data'][:, 1], label=sw_array['title'],
                    c=legend[sw_array['title']])
        custom_line_legend.append(Line2D([0], [0], color=legend[sw_array['title']], lw=4))
        custom_title_legend.append(sw_array['title'])

    plt.legend(custom_line_legend, custom_title_legend)
    # plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M %Y-%m-%d'))
    # plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=3))
    # plt.gca().xaxis.set_tick_params(rotation=35)
    # print(plt.gca().xaxis.get_majorticklabels())
    ax.set_xticks(OMNI_DATA[:, 0])
    ax.set_xticklabels([x.strftime('%H:%M %Y-%m-%d') for x in OMNI_DATA[:, 0]], rotation=40)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(0.3))
    ax.grid()
    # plt.show()
    plt.title(filename)
    plt.savefig(RESULT_DIR + '\%s.jpg' % filename, format='jpeg', dpi=400)
    plt.close()
    dt_from = dt2

