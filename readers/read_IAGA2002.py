import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from settings import DATA_DIR
import datetime

def AE():
    clf = pd.read_csv(DATA_DIR + "\AEAUALAO_1970-03-06_1970-03-11h_D.dat", skiprows=26, sep='\s+', )
    np_clf = clf.iloc[:, [0, 1, 3]].to_numpy()
    dt = [datetime.datetime.strptime(str(np_clf[i, 0] + 'T' + np_clf[i, 1]), '%Y-%m-%dT%H:%M:%S.%f') for i in np.arange(len(np_clf))]
    return np.vstack((dt, np_clf[:, 2])).T  # dt (1h), value

#print(AE())

def Kp():
    Kp_indices = {
        '0o': 0., '0+': 0.33, '1-': 0.67, '1o': 1.00, '1+': 1.33, '2-':	1.67, '2o': 2.00, '2+': 2.33, '3-': 2.67,
        '3o': 3.00, '3+': 3.33, '4-': 3.67, '4o': 4.00, '4+': 4.33, '5-': 4.67, '5o': 5.00, '5+': 5.33, '6-': 5.67,
        '6o': 6.00, '6+': 6.33, '7-': 6.67, '7o': 7.00, '7+': 7.33, '8-': 7.67, '8o': 8.00, '8+': 8.33, '9-': 8.67,
        '9o': 9.0}


    clf = pd.read_csv(DATA_DIR + "\Kp_1970-03-06_1970-03-11_D.dat", skiprows=35, sep='\s+', )
    np_clf = clf.iloc[:, [0, 1, 3]].to_numpy()
    dt = [datetime.datetime.strptime(str(np_clf[i, 0] + 'T' + np_clf[i, 1]), '%Y-%m-%dT%H:%M:%S.%f') for i in np.arange(len(np_clf))]
    kp_array = [Kp_indices[key] for key in np_clf[:, 2]]
    return np.vstack((dt, kp_array)).T  # dt (3h), value

def Dst():
    clf = pd.read_csv(DATA_DIR + "\Dst_1970-03-06_1970-03-11_D.dat", skiprows=24, sep='\s+', )
    np_clf = clf.iloc[:, [0, 1, 3]].to_numpy()
    dt = [datetime.datetime.strptime(str(np_clf[i, 0] + 'T' + np_clf[i, 1]), '%Y-%m-%dT%H:%M:%S.%f') for i in
          np.arange(len(np_clf))]
    return np.vstack((dt, np_clf[:, 2])).T  # dt (1h), value

print(Dst()[:10])
print(Kp()[:10])
print(AE()[:10])
