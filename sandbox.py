import datetime
import pymysql
import warnings
import numpy as np
from sattelite import Sattelite

from chaos7_model.chaos_model import CHAOS7
import pandas as pd


def data_reduction(respond, delta, fac2_mod=False):
    warnings.simplefilter("ignore", category=RuntimeWarning)
    """сжимание секундных данных до delta шага"""
    # fac2 = (Y, X, R), dt, (fac2)
    # vector, measure mu, fac, chaos = (Y, X, R), dt, (N, E, C)
    N, M = respond.shape
    if fac2_mod:
        idx999 = np.where(respond[:, 4] == 999)[0]
        # respond = respond[idx999]
        respond[idx999, 4] = np.nan
        redu_resp = np.empty((0, 5))
    else:
        redu_resp = np.empty((0, M))

    window = int(delta / 2)
    if window == 0:
        window = 1
    st_idx = 0
    while st_idx < N:
        if st_idx != 0:
            left_idx = st_idx - window
        else:
            left_idx = 0
        right_idx = st_idx + window
        delta_resp = respond[left_idx:right_idx]
        if fac2_mod:
            dt, y, x, r, fac2 = delta_resp[-1, (0, 1, 2, 3, 4)]
            fac2 = np.nanmean(delta_resp[:, 4])
            #if np.isnan(fac2):
            #    fac2 = 0.
            redu_resp = np.append(redu_resp, [[dt, y, x, r, fac2]], axis=0)

        else:
            dt, y, x, r = delta_resp[-1, (0, 1, 2, 3)]
            n = np.mean(delta_resp[:, 4])
            e = np.mean(delta_resp[:, 5])
            c = np.mean(delta_resp[:, 6])
            f = np.mean(delta_resp[:, 7])
            redu_resp = np.append(redu_resp, [[dt, y, x, r, n, e, c, f]], axis=0)
        st_idx += delta

        """if fac2_mod:
        #https://pandas.pydata.org/docs/user_guide/missing_data.html
        redu_resp[:, 4] = pd.DataFrame(np.array(redu_resp[:, 4]), dtype='float32').interpolate().to_numpy().T[0]
        """

        """
        miss_values_pd = pd.DataFrame(redu_resp[:, 4])
        fac2_miss_values_pd = miss_values_pd.fillna(miss_values_pd.mean())
        #fac2_miss_values_pd = miss_values_pd.fillna(value=miss_values_pd)
        redu_resp[:, 4] = fac2_miss_values_pd.T.to_numpy()
        """

            #for i, resp in enumerate(fac2_miss_values_pd):
             #       if np.isnan(respond[i, 3]):
             #                   print(respond[i], 'is nan')
    return redu_resp

sql_connect = pymysql.connect(host='imagdb.gcras.ru', port=3306,
                                  user='data_reader',
                                  passwd='V9e2OeroufrluwluN2u88leP9lAPhluzL7dlU67oAb5afROub9iUv7unLEhiEs9o',
                                  db='intermagnet',
                                  charset='utf8mb4',
                                  )
request = "SELECT `date`, latitude, longitude, radius, n, e, c, f FROM sat_sec_plain WHERE code='SWA' AND date BETWEEN UNIX_TIMESTAMP('2018-1-10 00:00:59') AND UNIX_TIMESTAMP('2018-1-10 05:59:59')"

cur = sql_connect.cursor()
cur.execute(request)
respond = cur.fetchall()
sql_connect.close()
respond = data_reduction(np.array(respond), delta=1, fac2_mod=False)
print(respond)
sat_datetime_unix = respond[:, 0]
sat_datetime = []
for i, t in enumerate(sat_datetime_unix):
    sat_datetime.append(datetime.datetime.utcfromtimestamp(int(t)))

sat_pos = respond[:, 1:4]
sat_value = respond[:, 4:]
chaos = CHAOS7()
print(sat_pos, 'sw_pos')
print(sat_value, 'sw value')
sat_value = chaos.get_sat_and_chaos_diff(sat_datetime, sat_pos, sat_value)
print(sat_value)