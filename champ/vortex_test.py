import datetime

import numpy as np
import scipy.io
from settings import DATA_DIR, RESULT_DIR
import matplotlib.pyplot as plt
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

def get_solar_coord(date, lon):
    lon_to360 = lambda x: (x - 180) % 360 - 180  # -180 180 to 0 360
    lon_to180 = lambda x: (x + 180) % 360 - 180  # 0 360 to -180 180
    midnight_lat, midnight_lon = sun_pos(date)

    """#midnight_lon = get_lon_from_LT(date)
    solar_coord_lon = midnight_lon + lon_to360(lon)
    if solar_coord_lon < 0:
        solar_coord_lon = 360 + solar_coord_lon
    elif solar_coord_lon >= 360:
        solar_coord_lon = solar_coord_lon - 360"""
    return midnight_lon

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)

def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def sph_to_cart(lon, lat):
    r = 6371.2
    az, el = math.radians(lon), math.radians(lat)  # lon, lat
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

mat = scipy.io.loadmat(DATA_DIR + '/vortex_11ut_paper.mat')
date = datetime.datetime(2007, 12, 10, 5, 00)
dk_idx = 3

LON = np.array(mat['LON'])
LAT = np.array(mat['LAT'])
vY = np.array(mat['vY'])
vX = np.array(mat['vX'])

LON_deg = np.rad2deg(LON)
LAT_deg = np.rad2deg(LAT)


def example_1(dk_idx):
    # полярная проекция
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    ax.quiver(LON,np.pi/2-LAT, vY[:, :, dk_idx], vX[:, :, dk_idx],  width=0.001, )


    plt.savefig(RESULT_DIR + '/vortex_%i.png' % dk_idx, dpi=500)

    #plt.show()

def example_2():
    # на графике
    fig, ax = plt.subplots()
    #ax.quiver(LON, LAT, -vY[:, :, dk_idx], -vX[:, :, dk_idx], width=0.001)
    #ax.quiver(LON, LAT, vX[:, :, dk_idx], -vY[:, :, dk_idx], width=0.001)
    #ax.quiver(LON, LAT, np.mean(-vY, axis=2), np.mean(vX, axis=2), width=0.001)

    u_src_crs = -vX[:, :, dk_idx] / np.cos(LAT_deg / 180 * np.pi)
    v_src_crs = vY[:, :, dk_idx]
    magnitude = np.sqrt(vX[:, :, dk_idx] ** 2 + vY[:, :, dk_idx] ** 2)
    magn_src_crs = np.sqrt(u_src_crs ** 2 + v_src_crs ** 2)

    ax.quiver(LON_deg, LAT_deg, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
             width=0.0008)

    plt.show()


def example_3(dk_idx):
    projection = ccrs.NorthPolarStereo(central_longitude=0)
    fig = plt.figure(figsize=(10, 10))  # a*dpi x b*dpi aka 3000px x 3000px
    camera = Camera(fig)
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
    # ax.set_extent([xlims[0], xlims[1], ylims[0], ylims[1]], crs=ccrs.Geodetic())

    # crs = ccrs.RotatedPole(pole_longitude=get_solar_coord(date, None), pole_latitude=90)
    crs = ccrs.PlateCarree()
    #projection = ccrs.Orthographic(central_latitude=90., central_longitude=0)


    u_src_crs = -vX[:, :, dk_idx] / np.cos(LAT_deg / 180 * np.pi)
    v_src_crs = vY[:, :, dk_idx]
    magnitude = np.sqrt(vX[:, :, dk_idx] ** 2 + vY[:, :, dk_idx] ** 2)
    magn_src_crs = np.sqrt(u_src_crs ** 2 + v_src_crs ** 2)

    """
    u_src_crs = -np.mean(vX, axis=2) / np.cos(LAT_deg / 180 * np.pi)
    v_src_crs = np.mean(vY, axis=2)
    magnitude = np.sqrt(np.mean(vX, axis=2) ** 2 + np.mean(vY, axis=2) ** 2)
    magn_src_crs = np.sqrt(u_src_crs ** 2 + v_src_crs ** 2)
    """

    ax.quiver(LON_deg, LAT_deg, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
              transform=crs,  width=0.0008)



    """vX2 = np.cos(LON) * (1 * vX[:, :, dk_idx]) + np.sin(LON) * (1 * vY[:, :, dk_idx])
    vY2 = np.sin(LON) * (1 * vX[:, :, dk_idx]) - np.cos(LON) * (1 * vY[:, :, dk_idx])

    ax.quiver(LON_deg, LAT_deg, -vY2, vX2, width=0.001, transform=crs)"""


    #q = ax.quiver(LON_deg, LAT_deg, np.mean(vY, axis=2), np.mean(vX, axis=2) , angles="xy", width=0.0008, transform=crs)
    #q.set_UVC(np.mean(vX, axis=2), np.mean(vY, axis=2))

    #q = ax.quiver(LON_deg, LAT_deg, vY[:, :, dk_idx], vX[:, :, dk_idx], width=0.0008, transform=crs, angles="uv")
    #q.set_UVC(vY[:, :, dk_idx], vX[:, :, dk_idx])
    #ax.quiver(LON_deg,LAT_deg, vY[:, :, dk_idx], vX[:, :, dk_idx], width=0.001,  transform=crs)

    plt.title(str(dk_idx+1) + ' min')

    #plt.savefig(RESULT_DIR + '/vortex_%i.png' % dk_idx, dpi=600)
    #plt.show()


def example_anim():
    fig = plt.figure(figsize=(10, 10))  # a*dpi x b*dpi aka 3000px x 3000px
    camera = Camera(fig)



    for dk_idx in range(17):
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

        ax.quiver(LON_deg, LAT_deg, u_src_crs * magnitude / magn_src_crs, v_src_crs * magnitude / magn_src_crs,
                  transform=crs, width=0.0008)

        plt.title(str(dk_idx + 1) + ' min')
        camera.snap()

    animation = camera.animate(interval=500)

    plt.rcParams['animation.ffmpeg_path'] = 'C:\\ffmpeg\\bin\\ffmpeg.exe'
    animation.save(os.path.join(RESULT_DIR, 'vortex_video.mp4'), writer='ffmpeg')
    print('DONE!!!')


#example_3(dk_idx)
example_anim()
#for idx in range(17):
#        example_anim(idx)
