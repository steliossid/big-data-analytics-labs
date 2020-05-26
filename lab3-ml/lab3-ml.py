from __future__ import division
from math import radians, cos, sin, asin, sqrt, exp
from datetime import datetime
from pyspark import SparkContext

sc = SparkContext(appName="lab_kernel")  

stations = sc.textFile("BDA/input/stations.csv")
temp = sc.textFile("BDA/input/temperature-readings.csv")

lines_st=stations.map(lambda line:line.split(";"))
lines_temp =temp.map(lambda line:line.split(";"))

h_distance = 180
h_date = 30
h_time = 4 
a = 55.90000 # latitude
b = 12.71660 # longtitude
date = "2014-03-10"

def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
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

d1=lines_st.map(lambda x: (x[0],d_distance(b,a,float(x[3]),float(x[4]))))
id_dist = sc.broadcast(d1.collect()).value

station_ids = []
for id in id_dist:
    if id not in station_ids:
        station_ids.append(id[0])
      
temps = lines_temp.map(lambda x: (x[0],x[1],x[2],float(x[3])))
filtered_temps= temps.filter(lambda x: x[0] in station_ids and x[1]<= date and x[1][5:10]!= '02-29').cache()

st_dict= dict(id_dist)
filtered_temps_kernels = filtered_temps.map(lambda x: (x[0],x[1],x[2],x[3],gaussian_kernel(st_dict[x[0]],h_distance),
                                                       gaussian_kernel(d_date(date,x[1]),h_date))).cache()
  
predicted_sum=[]
predicted_mult=[]
times = ["00:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", "14:00:00",
"12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]

for time in times:
    sum_allkernels = filtered_temps_kernels.map(lambda x: ((x[0],x[1],x[2],x[3]),x[4]+x[5]+
                                                            gaussian_kernel(d_time(time,x[2]),h_time))).cache()
															
    mult_allkernels = filtered_temps_kernels.map(lambda x: ((x[0],x[1],x[2],x[3]),x[4]*x[5]*
                                                            gaussian_kernel(d_time(time,x[2]),h_time))).cache()

    predicted_sum.append(sum_allkernels.map(lambda x: (x[0][3]*x[1])).sum() / sum_allkernels.map(lambda x: x[1]).sum())
    
    predicted_mult.append(mult_allkernels.map(lambda x: (x[0][3]*x[1])).sum() / mult_allkernels.map(lambda x: x[1]).sum())
    
print(predicted_sum)
print(predicted_mult)

sc.parallelize(predicted_sum).saveAsTextFile("BDA/output")
sc.parallelize(predicted_mult).saveAsTextFile("BDA/output")

sc.stop()