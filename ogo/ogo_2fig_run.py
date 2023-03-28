from ogo_2fig import draw_ogo2fig
import datetime


datetime_range_array = {
        't1': (datetime.datetime(1970, 3, 6, 5, 24, 44), datetime.datetime(1970, 3, 6, 7, 4, 2)),
        't2': (datetime.datetime(1970, 3, 10, 12, 3, 1), datetime.datetime(1970, 3, 10, 13, 42, 19)),
        't3': (datetime.datetime(1970, 3, 8, 13, 41, 56), datetime.datetime(1970, 3, 8, 15, 20, 16)),
        't4': (datetime.datetime(1970, 3, 8, 22, 3, 0), datetime.datetime(1970, 3, 8, 23, 37, 52)),
        't744': (datetime.datetime(1970, 3, 8, 23, 37, 53), datetime.datetime(1970, 3, 9, 1, 17, 12)),
        't745': (datetime.datetime(1970, 3, 9, 1, 17, 13), datetime.datetime(1970, 3, 9, 2, 56, 31)),
        't746': (datetime.datetime(1970, 3, 9, 2, 56, 33), datetime.datetime(1970, 3, 9, 4, 35, 50)),
        't760': (datetime.datetime(1970, 3, 9, 22, 48, 26), datetime.datetime(1970, 3, 10, 0, 27, 44)),
        't761': (datetime.datetime(1970, 3, 10, 0, 27, 46), datetime.datetime(1970, 3, 10, 2, 7, 2)),
        't762': (datetime.datetime(1970, 3, 10, 2, 7, 10), datetime.datetime(1970, 3, 10, 3, 46, 19)),
}

for key, dt_range in datetime_range_array.items():
        for igrf in [[False, False], [True, False], [False, True]]:
                igrf_value, igrf_diff = igrf
                for index in [[True, False, False, False], [False, True, False, False], [False, False, True, False], [False, False, False, True]]:
                        kp_dst_ae, solar_xyz, sw_plasma, electr_field = index
                        #draw_ogo2fig(str(key), datetime_range_array[key], delta_reduction=1, igrf_diff=igrf_diff, igrf_value=igrf_value, kp_dst_ae=kp_dst_ae, solar_xyz=solar_xyz, sw_plasma=sw_plasma, electr_field=electr_field)


igrf_value = True
igrf_diff = False
kp_dst_ae, solar_xyz, sw_plasma, electr_field = [False, False, False, True]
draw_ogo2fig('t762', datetime_range_array['t762'], delta_reduction=1, igrf_diff=igrf_diff, igrf_value=igrf_value, kp_dst_ae=kp_dst_ae, solar_xyz=solar_xyz, sw_plasma=sw_plasma, electr_field=electr_field)
