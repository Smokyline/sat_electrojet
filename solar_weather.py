import datetime
from matplotlib.lines import Line2D
from champ.sattelite import Sattelite
import matplotlib.pyplot as plt
import numpy as np
import sys
import matplotlib.colors as mcolors
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

"""while dt_from < dt_to:
    dt1 = dt_from
    dt2 = dt_from + datetime.timedelta(hours=12)
    print(dt1, dt2)"""
for dt1, dt2 in zip([dt_from], [dt_to]):

    idx_start, idx_stop = sat.get_start_stop_idx(time_array=OMNI_DATA_FULL[:, 0], dt_from=dt1, dt_to=dt2)
    filename = 'OMNI E FIELD %s - %s' % (dt1.strftime('%H_%M %Y-%m-%d'), dt2.strftime('%H_%M %Y-%m-%d'))

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

    #fig, axs = plt.subplots(3, 1, figsize=(16, 10))
    #fig, axs = plt.subplots(3, sharex=True, sharey=True, hspace=0)
    fig = plt.figure(figsize=(15, 11))
    #fig.subtitle(filename)

    c = mcolors.ColorConverter().to_rgb
    legend = {
        'BX, nT': c('black'), 'SW_plasma_temp, K': c('#0B2DD6'),
        'Flow_pressure, nPa': c('#01E1F0'), 'BY, nT': c('#00D62E'),
        #'SW_proton_dens, N/cm^3': c('#FABE00'), 'elecrtric_field, mV/m': c('#FFFF0D'),
        'SW_proton_dens, N/cm^3': c('#FABE00'), 'elecrtric_field, mV/m': c('black'),
        'BZ, nT': c('#FA4200'), 'SW_plasma_speed, km/s': c('#D60192'),
        'Plasma_beta': c('#8811F0')
    }
    custom_line_legend = []
    custom_title_legend = []
    #for sw_array in [BX, BY, BZ, PLASMA_TEMP, PROTON_DENS, PLASMA_SPEED, FLOW_PRESSURE, ELECTRIC_FIELD, PLASMA_BETA]:
    #for ax_i, sw_array in enumerate([BX, BY, BZ]):
    #for ax_i, sw_array in enumerate([PLASMA_TEMP, PROTON_DENS, PLASMA_SPEED]):
    for ax_i, sw_array in enumerate([FLOW_PRESSURE, ELECTRIC_FIELD, PLASMA_BETA]):
        temp = 310 + ax_i+1
        ax = plt.subplot(temp)
        plt.subplots_adjust(hspace=.001)
        #temp = tic.MaxNLocator(3)
        #ax.yaxis.set_major_locator(temp)
        #ax.set_xticklabels(())
        #ax.title.set_visible(False)
        #ax.label_outer()
        if ax_i==0:
            ax.set_title(filename)
        if 'SW_plasma_temp' in sw_array['title']:
            ax.set_ylabel(sw_array['title'])
            ax.plot(sw_array['data'][:, 0], sw_array['data'][:, 1], label=sw_array['title'],
                     c=legend[sw_array['title']])
        elif 'SW_plasma_speed' in sw_array['title']:
            ax.set_ylabel(sw_array['title'])
            ax.plot(sw_array['data'][:, 0], sw_array['data'][:, 1], label=sw_array['title'],
                     c=legend[sw_array['title']])
        else:
            ax.plot(sw_array['data'][:, 0], sw_array['data'][:, 1], label=sw_array['title'],
                    c=legend[sw_array['title']])
        ax.set_xticks(OMNI_DATA[::3, 0])
        if ax_i==2:
            ax.set_xticklabels([x.strftime('%m/%d %H:%M UT') for x in OMNI_DATA[::3, 0]], rotation=90)
        else:
            ax.set_xticklabels(['' for x in OMNI_DATA[::3, 0]], rotation=40)

        ax.grid()
        ax.set(ylabel= sw_array['title'])
        ax.legend([Line2D([0], [0], color=legend[sw_array['title']], lw=4)], [sw_array['title']])

        #custom_line_legend.append(Line2D([0], [0], color=legend[sw_array['title']], lw=4))
        #custom_title_legend.append(sw_array['title'])

    #plt.legend(custom_line_legend, custom_title_legend)

    #ax.set_xticks(OMNI_DATA[::3, 0])
    #ax.set_xticklabels([x.strftime('%H:%M %Y-%m-%d') for x in OMNI_DATA[::3, 0]], rotation=40)
    #ax.grid()

    # plt.show()
    #plt.title(filename)
    plt.savefig(RESULT_DIR + '\%s.jpg' % filename, format='jpeg', dpi=400)
    plt.close()
    dt_from = dt2

