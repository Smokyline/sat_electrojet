from champ.sattelite import Sattelite
import codecs
from settings import DATA_DIR
import matplotlib.pyplot as plt
from tools import *


def import_cosmos_from_txt(delta=100, turn_from=0, turn_to=1, col_value=13):
    # 0 - виток, 1...3 - дата, 4..6 время МСК, 7...9 - время локал, 10 - широта, 11 - долгота, 12 - высота км над морем
    # 13 - Тизм. в гамм., 14 - Тиспр. в гамм., 15 - Тразн. в гамм., 16 - J

    f = codecs.open(DATA_DIR + '\cosmos321.txt', 'r', 'cp1251')

    sat_data = np.empty((0, 6))
    for i, line in enumerate(f.readlines()):
        if len(line) > 25 and i != 0:
            try:
                strip_line = line.rsplit('	')
                turn = int(strip_line[0])
                r = float(strip_line[12].replace(',', '.'))
                theta = float(strip_line[10].replace(',', '.'))
                phi = (float(strip_line[11].replace(',', '.')) + 180) % 360 - 180
                #phi = float(strip_line[11].replace(',', '.'))
                F = float(strip_line[col_value].replace(',', '.'))
                dt = datetime.datetime.strptime('%s %s' %
                                                ('%i-%i-%i' % (
                                                    int(strip_line[1]), (int(strip_line[2])), int(strip_line[3])),
                                                 '%i:%i:%i' % (int(strip_line[4]), int(strip_line[5]),
                                                               int(strip_line[6].rstrip(',')[0]))),
                                                '%Y-%m-%d %H:%M:%S') - datetime.timedelta(hours=3)  # MSK is UTC +3

                sat_data = np.append(sat_data, np.array([[turn, dt, theta, phi, r, F]]), axis=0)
            except:
                print(i, 'line ERROR')

    sat_data = sat_data[sat_data[:, 1].argsort()]

    return sat_data


sat = Sattelite(dt_from='', dt_to='')
sat_data_full = import_cosmos_from_txt(delta=1, turn_from=740, turn_to=765, col_value=13)

turn_from=740
turn_to=765
turn_full_array = sat_data_full[:, 0]
sat_data = sat_data_full[:, 1:]

turn_last = 0
turn_dt_unique = []
turn_unique_array = []

delta_turn_array = []
delta_turn_array_2 = []
for i, t in enumerate(turn_full_array):
    if t > turn_last or (i==len(turn_full_array)-1):
        if i != 0:
            print(turn_full_array[i - 1], sat_data[i - 1, 0] - turn_dt_unique[-1])
            delta_turn_array.append(sat_data[i - 1, 0] - turn_dt_unique[-1])
        turn_last = t
        turn_dt_unique.append(sat_data[i, 0])
        turn_unique_array.append(t)
for i, turn_dt in enumerate(turn_dt_unique):
    if i != 0:
        # print(turn_unique_array[i-1], turn_unique_array[i], turn_dt-turn_dt_unique[i-1])
        delta_turn = turn_unique_array[i] - turn_unique_array[i - 1]
        delta_time = turn_dt - turn_dt_unique[i - 1]
        # print(delta_time/delta_turn)
        # print('------')
        delta_turn_array_2.append(delta_time / delta_turn)

# turn_mean_time = np.mean(delta_turn_array)
turn_mean_time = np.max(delta_turn_array)
print(np.mean(delta_turn_array), np.max(delta_turn_array))
print(np.mean(delta_turn_array_2), np.max(delta_turn_array_2))
# zero_turn_idx = 660
# zero_turn_dt = datetime.datetime(1970, 3, 3, 18, 23)
# zero_turn_idx = 642
# zero_turn_dt = datetime.datetime(1970, 3, 2, 15, 20)
for i, dt in enumerate(turn_dt_unique):
    zero_turn_idx = turn_unique_array[i]
    zero_turn_dt = dt
    turn_from_dt = zero_turn_dt + turn_mean_time * (turn_from - zero_turn_idx)
    print('исходный виток:', zero_turn_idx,
          'время начала %s витка:' % turn_from, turn_from_dt,
          'время начала %s витка:' % str(turn_from + 1), turn_from_dt + turn_mean_time,
          'время начала %s витка:' % turn_to, turn_from_dt + turn_mean_time * (turn_to - turn_from))





fig = plt.figure()
ax = fig.add_subplot(projection='3d')
# axs = host_subplot(111, axes_class=axisartist.Axes)

ax.scatter(0, 0, 0, s=40, c='k', marker='o')

#selected_turns = [423, 449, 609, 621, 625]
selected_turns = [609]
#selected_turns = np.unique(sat_data_full[:, 0])
#print(selected_turns, len(selected_turns))
print(np.where(sat_data_full[:, 5]>100000))
for i, turn in enumerate(selected_turns):
    turn_idx = np.where(sat_data_full[:, 0]==turn)[0]
    sat_data = sat_data_full[turn_idx, 1:]


    for i, dt in enumerate(sat_data[:, 0]):
        pass
        #print(dt, sat_data[i, 1:4])
    xyz = np.empty((0, 3))
    for lat, lon, R in zip(sat_data[:, 1], sat_data[:, 2], sat_data[:, 3]):
        xyz = np.append(xyz, [gc_latlon_to_xyz(lat, lon, R)], axis=0)

    index_day, index_night = sat.get_DayNight_idx(dt=sat_data[:, 0], phi=sat_data[:, 2])
    #sc = ax.scatter(xyz[index_day, 0], xyz[index_day, 1], xyz[index_day, 2], s=3, y_label=str(turn), marker='o')
    #ax.scatter(xyz[index_night, 0], xyz[index_night, 1], xyz[index_night, 2], s=3,  marker='x', c=sc.get_facecolors()[0].tolist())
    ax.scatter(xyz[index_day, 0], xyz[index_day, 1], xyz[index_day, 2], s=3, label=str(turn), marker='o', c='r')
    ax.scatter(xyz[index_night, 0], xyz[index_night, 1], xyz[index_night, 2], s=3, marker='x',c='b')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

ax.legend(loc=2)
#ax2.legend(loc=1)

fig.suptitle('COSMOS-321')
#plt.title()
plt.show()