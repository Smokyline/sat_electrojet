import datetime

from tools import *
lonx = lambda x: (x + 180) % 360 - 180

dt1 = datetime.datetime(2007, 12, 10, 6, 0 )
dt2 =  datetime.datetime(2007, 12, 10, 7, 40 )

#lon = get_lon_from_LT(dt1)
#lon = get_lon_from_LT(datetime.datetime(2007, 12, 10, 7, 34))   #MEA [54.616, -113.347]
lon = get_lon_from_LT(datetime.datetime(2007, 12, 10, 6, 0))   #CBB [69.123, -105.031],
#lon = get_lon_from_LT(datetime.datetime(2007, 12, 10, 12, 0))
print('lon:', lon, lonx(lon))
