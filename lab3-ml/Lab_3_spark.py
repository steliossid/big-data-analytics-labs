from __future__ import division
from math import radians, cos, sin, asin, sqrt, exp
from datetime import datetime
from pyspark import SparkContext

sc = SparkContext(appName="lab_kernel")

def haversine(lon1, lat1, lon2, lat2):
# convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
# haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    km = 6367 * c
    return km



h_distance =  500000
h_date = 100
h_time = 5 
a = 55.90000 # latitude of interest
b = 12.71660 # longtitude of interest
date = "2014-07-10" #date of interest

## Write the three types of distances we want
## Function to coumpute distance between place of interest and stations
def d_distance(b,a,longtitude,latitude):
    dist = haversine(b,a,longtitude,latitude)
    return dist

## Function to compute difference in days between date of interest and date of measurment
def d_date(date1, date2):
    date1 = datetime.strptime(date1, "%Y-%m-%d")
    date2 = datetime.strptime(date2, "%Y-%m-%d")
    abs_diff= abs((date2 - date1).days)
    return abs_diff % 365

## Function to compute difference in hours between the hour we want and the hour of a measurment
def d_time(time1,time2):
    time_diff = abs(datetime.strptime(time2, '%H:%M:%S').hour - datetime.strptime(time1, '%H:%M:%S').hour)
    if time_diff <=12:
        return time_diff
    else:
        return (24-time_diff)


#### Import datasets and process them    

stations = sc.textFile("BDA/input/stations.csv")
temp = sc.textFile("BDA/input/temperature-readings.csv")

lines_st=stations.map(lambda line:line.split(";")) #split each line
lines_temp =temp.map(lambda line:line.split(";"))

### Compute lat-lon distances and save in type list
d1=lines_st.map(lambda x: (x[0],d_distance(a,b,float(x[3]),float(x[4]))))  ## Type = RDD
id_dist = sc.broadcast(d1.collect()).value   ## Type = list

## Save station ids
station_ids = []
for id in id_dist:
    if id not in station_ids:
        station_ids.append(id[0])

## Filter out later dates        
temps = lines_temp.map(lambda x: (x[0],x[1],x[2],float(x[3])))
filtered_temps= temps.filter(lambda x: x[0] in station_ids and x[1]<= date  and x[1][5:7]!= '02' and x[1][8:10]!= '29').cache() #station number,date,hour,temperature

st_dict= dict(id_dist)

filtered_temps_kernels = filtered_temps.map(lambda x: (x[0],x[1],x[2],float(x[3]),exp(-(st_dict[x[0]]/h_distance)**2),
                                                       exp(-(d_date(date,x[1])/h_date)**2))).cache()


predicted_sum=[]
predicted_mult=[]
for time in ["00:00:00", "22:00:00", "20:00:00", "18:00:00", "16:00:00", "14:00:00",
"12:00:00", "10:00:00", "08:00:00", "06:00:00", "04:00:00"]:
    sum_allkernels = filtered_temps_kernels.map(lambda x: ((x[0],x[1],x[2],float(x[3])),x[4]+x[5]+
                                                            exp(-(d_time(time,x[2])/h_time)**2))).cache()
    
    mult_allkernels = filtered_temps_kernels.map(lambda x: ((x[0],x[1],x[2],float(x[3])),x[4]*x[5]*
                                                            exp(-(d_time(time,x[2])/h_time)**2))).cache()
    
    predicted_sum.append(sum_allkernels.map(lambda x: (x[0][3]*x[1])).sum() / sum_allkernels.map(lambda x: x[1]).sum())
    
    predicted_mult.append(mult_allkernels.map(lambda x: (x[0][3]*x[1])).sum() / mult_allkernels.map(lambda x: x[1]).sum())
    
print(predicted_sum)
print(predicted_mult)
sc.parallelize(predicted_sum).saveAsTextFile("sum_kernel_predictions")
sc.parallelize(predicted_mult).saveAsTextFile("mult_kernel_predictions")   
    
sc.stop()