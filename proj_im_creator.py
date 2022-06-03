from sword import SWORD
from chaos7_model.chaos_model import CHAOS7

from tools import *
import numpy as np

def get_proj_image(sat_label, sat_datetime, sat_pos, sat_value, legend_label,
                   sat_poly_loc=None, sat_rad_loc=None, sat_rad_deg=5, proj_type='miller',
                   draw_IGRFvector_diff=False, draw_CHAOSvector_diff=False,
                   mag_grid_coord=False, lt_grid_coord=False, central_lon=None, obs=None):

    ax_label = fr'{sat_label} {sat_datetime[0]} -- {sat_datetime[-1]} '

    #   установка границ проекции на основе координат области значений swarm
    if sat_poly_loc is not None:
        proj_extend_loc = sat_poly_loc #[lon_min, lon_max, lat_min, lat_max, ]
        ax_label += 'loc:lat{%0.2f:%0.2f},lon{%0.2f:%0.2f}' % (proj_extend_loc[2], proj_extend_loc[3],
                                                               proj_extend_loc[0], proj_extend_loc[1],)
        cut_swarm_value_bool = True
    elif sat_rad_loc is not None:
        proj_extend_loc = get_swarm_poly_loc(sat_rad_loc, deg_radius=sat_rad_deg)
        cut_swarm_value_bool = True
    else:
        cut_swarm_value_bool = False
        proj_extend_loc = None

    STATUS = 1
    ######################################################################################################
    #   инициация рендера, указание проекции, и если требуется extend_loc - приближение конкретной области
    sword = SWORD(proj_type=proj_type, extend=proj_extend_loc, central_lon=central_lon)
    sword.draw_sun_terminator(sat_datetime)

    if sat_rad_loc is not None:
        sword.draw_point_with_annotate([sat_rad_loc[1], sat_rad_loc[0]], annotate='center', marker='o')


    if mag_grid_coord:
        sword.draw_mag_coord_lines(str(sat_datetime[0].date()), geomag_pole=True)    # True to_coord='MAG' /  False to_coord='GSM'
    if lt_grid_coord:
        sword.draw_lt_coord_lines(sat_datetime)
    #   выбор канала n, e, c, fac
    #legend_label = ['Bn', 'Be', 'Bc', 'fac']

    if obs is not None:
        for obs_name in obs.keys():
            y, x = obs[obs_name]
            sword.draw_point_with_annotate([x, y], str(obs_name))

    # выбор точек измерений SWARM в указанном полигоне
    # lat, lon
    """
    poly_points = [i1, i2, i3, i4] # lat... lat, lon... lon
    i1 = [lat, lon]
    [i1]--------[i2]
      |         |
      |  p_in_p |
      |         |
    [i4]--------[i3]
    """

    if cut_swarm_value_bool:
        if proj_extend_loc is not None:
            print('cut swarm_pos manual by proj_extend_loc')
            """
            loc:lat{proj_extend_loc[2]:proj_extend_loc[3]},
                lon{proj_extend_loc[0]:proj_extend_loc[1]} 
            """
            poly_loc = [[proj_extend_loc[1],proj_extend_loc[2],], [proj_extend_loc[1],proj_extend_loc[3],],
                        [proj_extend_loc[0],proj_extend_loc[3],], [proj_extend_loc[0],proj_extend_loc[2],]]
            p_in_p, poly = get_points_in_poly(sat_pos[:, :2], poly_loc, proj_type)

        #print(len(swarm_pos))
        #print(len(p_in_p))
        #print(p_in_p)
        sat_pos_in_poly = sat_pos[p_in_p]
        sat_value_in_poly = sat_value[p_in_p]
        sat_datetime_in_poly= sat_datetime[p_in_p]
        print('len cut pos', len(sat_datetime_in_poly))
        vline_min_path = True
        if len(sat_datetime_in_poly) == 0:
            STATUS = 0
    else:
        """if proj_type == 'ortho_n':
            p_in_p = data_lat_up(swarm_pos, lat=0, hemisphere='N')    # bool
            swarm_pos_in_poly = swarm_pos[p_in_p]
            swarm_values_in_poly = swarm_values[p_in_p]
        elif proj_type == 'ortho_s':
            p_in_p = data_lat_up(swarm_pos, lat=0, hemisphere='S')     # bool
            swarm_pos_in_poly = swarm_pos[p_in_p]
            swarm_values_in_poly = swarm_values[p_in_p]"""

        sat_pos_in_poly = sat_pos
        sat_value_in_poly = sat_value
        sat_datetime_in_poly = sat_datetime
        vline_min_path = False
    # отрисовка временных линий на пролете спутника
    if vline_min_path:
        sat_str_dt = []
        for dt in sat_datetime:
            sat_str_dt.append(str(dt))
        vline_dt = np.array(sat_str_dt)
    else:
        vline_dt = None
    sword.draw_swarm_path(sat_pos, points_time=vline_dt)
    #sword.draw_swarm_path(sat_pos[:, :2], points_time=None) # no vlines on path
    # отрисовка вектора (X, Y, n, e) или (X, Y, |n-x|, |e-y|)
    if draw_IGRFvector_diff or draw_CHAOSvector_diff:
        #vector_components = swarm_values_nec[:, :2]   # n,e component
        vector_components = sat_value_in_poly[:, :2]   # n,e component
        chaosm = CHAOS7(swarm_set=['l', sat_pos_in_poly, sat_datetime_in_poly, vector_components])
        if draw_IGRFvector_diff:
            vector_subtraction = swarm_egrf_vector_subtraction(sat_pos_in_poly, vector_components, sat_datetime_in_poly)
            model_name = 'IGRF'
        elif draw_CHAOSvector_diff:
            vector_subtraction = chaosm.get_swarm_chaos_vector_subtraction()
            model_name = 'CHAOS7'

        sat_value_in_poly = vector_subtraction[:, 0]     # dd
        vector_components = vector_subtraction[:, (1, 2)]   # dx, dy

        # convert to geomagnetic coords
        #swarm_pos_in_poly = geo2mag(swarm_pos_in_poly, swarm_date_in_poly)
        legend_label = model_name + '_Dd'
        ax_label += '|SWARM-%s|'%model_name

    """    
        if convert_coord is not None:
            # конвертация географических координат в dest_sys
            swarm_pos_in_poly = convert_coord_system(swarm_pos_in_poly, dest_sys=convert_coord)
            ax_label += 'coord_sys:%s ' % convert_coord
        vector_components = swarm_egrf_vector_subtraction(swarm_pos_in_poly, vector_components, swarm_date)
        sword.draw_vector(swarm_pos_in_poly, B=vector_components)"""
    #sword.draw_mag_coord_lines(swarm_date_in_poly[0], geomag_pole=False)
    #sword.draw_swarm_path(swarm_pos_in_poly[:, :2])

    # отрисовка точек измерений swarm в в указанном полигоне
    # если полигона нет - отрисовка всех (swarm_pos_in_poly = swarm_pos)
    if STATUS == 1:

        sword.draw_swarm_scatter(sat_pos_in_poly, sat_value_in_poly, sat_datetime_in_poly, custom_label=legend_label,
                                 annotate=False)  # отрисовка значение точек на орбите
        if draw_CHAOSvector_diff or draw_IGRFvector_diff:
            sword.draw_vector(sat_pos_in_poly, B=vector_components)

    # конвертация figure matplotlib в PIL image для stacker.py
    if cut_swarm_value_bool == True or proj_extend_loc is not None:
        sword.set_axis_label(ax_label, zoom_axis=True)
    else:
        sword.set_axis_label(ax_label)

    if STATUS == 1:
        out = sword.fig_to_PIL_image()
        sword.clear()
    if STATUS == 0:
        sword.clear()
        out = 'NO SWARM DATA IN TIME INTERVAL FROM %s TO %s IN LOCATION lat:%s... %s lon:%s... %s\n\n' \
              'hint: expand specified location or decrease delta' % (
           sat_datetime[0], sat_datetime[-1], proj_extend_loc[0], proj_extend_loc[1],
                                                                       proj_extend_loc[2], proj_extend_loc[3],)
    return (STATUS, out)


def get_plot_im(swarm_sets, labels, auroral, channel, delta, measure_mu, ground_station=None, txt_out=False):
    #swarm_liter, swarm_pos, swarm_date, swarm_time, swarm_values = swarm_info[0]
    #from_date, to_date = swarm_info[1], swarm_info[2]
    # инициация рендера
    sword = SWORD()

    #   отрисовка графика
    draw_list = []
    auroral_list = []
    d2txt = Data2Text(SWARM_liter='A')

    for i, swarm_set in enumerate(swarm_sets):
        position_list, date_list, time_list = swarm_set[1], swarm_set[2], swarm_set[3]
        d2txt.annotate = 'from %s to %s' % (date_list[0]+'T'+time_list[0], date_list[-1]+'T'+time_list[-1])
        #   find swarm value shape size
        if len(np.array(swarm_set[4]).shape) == 1:
            shape_size = 1
        else:
            shape_size = np.array(swarm_set[4]).shape[1]

        #   append value[channel] or value
        if channel is not None:
            try:
                value = np.array(swarm_set[4])[:, channel]
            except:
                value = swarm_set[4]
            if measure_mu:
                value = get_measure_mu(value)
            label = labels[i]
            draw_list.append([label, date_list, time_list, value, position_list])
            d2txt.SWARM_liter = label.rsplit('-')[1][0]
            d2txt.SWARM_channel = ['N', 'E', 'C'][channel]
            d2txt.append(position_list, 'SWARM_pos')
            d2txt.append(value, 'SWARM')

        if channel is None:
            if shape_size == 1:
                value = swarm_set[4]
                if measure_mu:
                    value = get_measure_mu(value)
                label = labels[i]
                draw_list.append([label, date_list, time_list, value, position_list])
                d2txt.SWARM_liter = label.rsplit('-')[1][0]
                d2txt.SWARM_channel = 'FAC2'
                d2txt.append(position_list, 'SWARM_pos')
                d2txt.append(value, 'SWARM')
            else:
                for ch in range(3):
                    label = labels[ch]
                    value = np.array(swarm_set[4])[:, ch]
                    if measure_mu:
                        value = get_measure_mu(value)
                    draw_list.append([label, date_list, time_list, value, position_list])
                    d2txt.SWARM_liter = label.rsplit('-')[1][0]
                    d2txt.SWARM_channel = str(channel)
                    d2txt.append(position_list, 'SWARM_pos')
                    d2txt.append(value, 'SWARM')
        if auroral:
            auroral_near_swarm = get_nearest_auroral_point_to_swarm(swarm_set)
            auroral_list.append(auroral_near_swarm)
            d2txt.auroral_shp = 'near'
            d2txt.append(auroral_near_swarm, 'auroral')



    status = 1
    if status == 1:
        if txt_out:
            d2txt.array_to_column()
            d2txt.save_columns_to_txt()
            out = d2txt.id
        else:
            sword.draw_plot(draw_list, auroral_list, delta, draw_station=ground_station)
            out = sword.fig_to_PIL_image()
    return (status, out)


def get_single_plot(sat_label, sat_time, sat_pos, sat_value):
    sword = SWORD()
    date_list, time_list = [], []
    for dt in sat_time:
        date_list.append(str(dt.date()))
        time_list.append('%s:%s:%s' % (dt.hour, dt.minute, dt.second))
    draw_list = [[sat_label, date_list, time_list, sat_value, sat_pos]]
    sword.draw_plot(draw_list, auroral_list=[])
    out = sword.fig_to_PIL_image()

    status = 1

    return (status, out)