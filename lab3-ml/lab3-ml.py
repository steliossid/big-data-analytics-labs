from __future__ import division
from math import radians, cos, sin, asin, sqrt, exp
import datetime
from pyspark import SparkContext
import numpy as np

sc = SparkContext(appName = "lab3-ml")

temp_rdd = sc.textFile("BDA/input/temperature-readings.csv")
temp_parts = temp_rdd.map(lambda l: l.split(";"))

stations_rdd = sc.textFile("BDA/input/stations.csv")
stations_parts = stations_rdd.map(lambda l: l.split(";"))

def haversine(lon1, lat1, lon2, lat2):
  """
  Calculate the great circle distance between two points
  on the earth (specified in decimal degrees)
  """
  # convert decimal degrees to radians
  lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
  # haversine formula
  dlon = lon2 - lon1
  dlat = lat2 - lat1
  a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
  c = 2*asin(sqrt(a))
  km = 6367*c
  return km

def d_distance(b, a, longitude, latitude):
  dist = haversine(b, a, longitude, latitude)
  return(dist)

def d_time(time, time_col):
  d1 = datetime.datetime.strptime(time, "%H:%M:%S")
  d2 = datetime.datetime.strptime(time_col, "%H:%M:%S")
  diff = abs(d2-d1).total_seconds()/3600
  min_diff = min(diff, 24-diff)
  return min_diff

def d_date(date, date_col):
  d1 = datetime.datetime.strptime(date[5:10], "%m-%d")
  d2 = datetime.datetime.strptime(date_col[5:10], "%m-%d")
  diff = abs(d2-d1).days
  min_diff = min(diff, 365-diff)
  return min_diff

def gaussian_kernel(u, h):
  return(exp(-(u/h)**2))
  
# tweak params
h_distance = 180
h_date = 40
h_time = 4
a = 58.4274 # Latitude
b = 14.826 # Longitude
date = "2014-05-04"
  
d1 = stations_parts.map(lambda x: (x[0], d_distance(b, a, float(x[4]), float(x[3]))))
stations = sc.broadcast(d1.collect()).value
station_ids = []
for st in stations:
  station_ids.append(st[0])

temperatures = temp_parts.map(lambda x: (x[0], x[1], x[2], float(x[3])))
temperatures = temperatures.filter(lambda x: int(x[1][5:7]) != 2 and int(x[1][8:10])!= 29)
temperatures = temperatures.filter(lambda x: x[0] in station_ids and datetime.datetime.strptime(date, "%Y-%m-%d") >= datetime.datetime.strptime(x[1], "%Y-%m-%d"))
#temperatures = temperatures.sample(False, 0.1)
temperatures = temperatures.cache()

stations_dict = dict(stations)
d1 = temperatures.map(lambda x: (stations_dict[x[0]]))

d2 = temperatures.map(lambda x: (d_date(date, x[1])))

times = ["24:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", 
         "14:00:00", "12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]

temps = temperatures.map(lambda x: (float(x[3]))).collect()
temps = np.array([temps])

k1 = d1.map(lambda x: (gaussian_kernel(x, h_distance))).collect()
k1 = np.array([k1]).T
k2 = d2.map(lambda x: (gaussian_kernel(x, h_date))).collect()
k2 = np.array([k2]).T

pred_temps = []

for time in times:
  if time == "24:00:00":
    time = "00:00:00"
  d3 = temperatures.map(lambda x: (d_time(time, x[2])))
  k3 = d3.map(lambda x: (gaussian_kernel(x, h_time))).collect()
  k3 = np.array([k3]).T
  ks = np.c_[k1, k2, k3]
  sum_rows = np.array([np.sum(ks, axis=1)]).T
  sumrows_temps = sum_rows.dot(temps)
  #sumrows_temps = np.multiply(sum_rows, temps)
  kernel_sum_temp  = np.sum(sumrows_temps)
  kernel_sum_rows = np.sum(sum_rows)
  pred_temps.append(round(kernel_sum_temp/kernel_sum_rows,1))

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Lab3 Predicted Temperatures")
print(pred_temps)
sc.parallelize(pred_temps).saveAsTextFile("BDA/output")