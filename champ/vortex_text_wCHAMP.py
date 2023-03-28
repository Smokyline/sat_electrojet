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

import datetime

import scipy.io
from settings import DATA_DIR, RESULT_DIR
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import (LongitudeFormatter, LatitudeFormatter,
                                LatitudeLocator)
import matplotlib.ticker as mticker
from sun_position import sun_pos
from celluloid import Camera
import os
from tools import latlt2polar
import math

sat = Sattelite(dt_from='2007-12-08T00:00:00', dt_to='2007-12-12T00:00:00')
#sat = Sattelite(dt_from='2007-12-10T00:00:00', dt_to='2007-12-10T00:00:00')
chaos = CHAOS7()

sat_data = sat.import_CHAMP_from_cdf(delta=1)   # dt, theta, phi, r, N, E, C, F
print('delta:', sat_data[1, 0]-sat_data[0, 0])


# TRACKS
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 4, 56), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 5, 13))]  # 1nd
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 6, 27), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 6, 43))]  # 1st main
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 7, 58), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 8, 15))]  # 3 main
#sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 9, 30), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 9, 46))]  # 2nd
sat_data = sat_data[np.logical_and(sat_data[:, 0] >= datetime.datetime(2007, 12, 10, 11, 00), sat_data[:, 0] <= datetime.datetime(2007, 12, 10, 11, 20))]  # 4 main

sat_datetime = sat_data[:, 0]
sat_pos = sat_data[:, 1:4]

#vortex_array = None
#[LON, LAT, vX, vY] = load_vortex(filename='vortex_05ut_paper')
#[LON, LAT, vX, vY] = load_vortex(filename='vortex_06ut_paper')
#[LON, LAT, vX, vY] = load_vortex(filename='vortex_08ut_paper')
#[LON, LAT, vX, vY] = load_vortex(filename='vortex_09ut_paper')
[LON, LAT, vX, vY] = load_vortex(filename='vortex_11ut_paper')

TITLE = 'vortex_11ut'
time_start = datetime.datetime(2007, 12, 10, 11, 00)

LON_deg = np.rad2deg(LON)
LAT_deg = np.rad2deg(LAT)




for dk_idx in range(17):
    fig = plt.figure(figsize=(10, 10))  # a*dpi x b*dpi aka 3000px x 3000px
    projection = ccrs.NorthPolarStereo(central_longitude=0)
    ax = fig.add_subplot(1, 1, 1, projection=projection)
    ax.set_facecolor((1.0, 1.0, 1.0))
    ax.coastlines(resolution='110m', color='k', zorder=7)
    ax.add_feature(cfeature.LAND, facecolor='0.75', zorder=0)
    gl = ax.gridlines(draw_labels=True)
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 180, 15))
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 10))
    gl.xformatter = LongitudeFormatter()
    gl.yformatter = LatitudeFormatter()
    gl.ylabel_style = {'size': 11, 'color': 'gray'}
    gl.xlabel_style = {'size': 11, 'color': 'gray'}
    ax.set_global()
    global_extent = ax.get_extent(crs=ccrs.PlateCarree())
    ax.set_extent(global_extent[:2] + (50, 90), crs=ccrs.PlateCarree())
    crs = ccrs.PlateCarree()

    u_src_crs = -vX[:, :, dk_idx] / np.cos(LAT_deg / 180 * np.pi)
    v_src_crs = vY[:, :, dk_idx]
    magnitude = np.sqrt(vX[:, :, dk_idx] ** 2 + vY[:, :, dk_idx] ** 2)
    magn_src_crs = np.sqrt(u_src_crs ** 2 + v_src_crs ** 2)

    q=ax.quiver(LON_deg, LAT_deg, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
              transform=crs, width=0.0008)
    u = 50
    ax.quiverkey(q, X=0.75, Y=0.99, U=u, labelpos='E', label='{X, Y} vector length = %s' % u,
                      transform=crs)

    sat_current_pos_idx = np.where(sat_datetime <= time_start+datetime.timedelta(minutes=dk_idx))[0]
    ax.plot(sat_pos[sat_current_pos_idx, 1], sat_pos[sat_current_pos_idx, 0], ls=':', color='r', transform=ccrs.Geodetic())
    ax.set_title(time_start+datetime.timedelta(minutes=dk_idx))
    plt.savefig(RESULT_DIR + '/vortex_%i.png' % dk_idx, dpi=600)
    plt.clf()


# ffmpeg -framerate 1 -pattern_type sequence -i vortex_%01d.png -s:v 1920x1080 -c:v libx264 -pix_fmt yuv420p out.mp4