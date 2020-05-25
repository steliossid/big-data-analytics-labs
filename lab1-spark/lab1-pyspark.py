from pyspark import SparkContext

sc = SparkContext(appName = "lab1-spark")

temperature_file = sc.textFile("input_data/temperature-readings.csv")
lines = temperature_file.map(lambda line: line.split(";"))

precipitation_file = sc.textFile("input_data/precipitation-readings.csv")
prec_lines = precipitation_file.map(lambda line: line.split(";"))

ostergotland_file = sc.textFile("input_data/stations-Ostergotland.csv")
ostergotland_lines = ostergotland_file.map(lambda line: line.split(";"))

# 1
year_temperature = lines.map(lambda x: (x[1][0:4], float(x[3])))
year_temperature = year_temperature.filter(lambda x: int(x[0])>=1950 and int(x[0])<=2014)

max_temperatures = year_temperature.reduceByKey(max)
min_temperatures = year_temperature.reduceByKey(min)

max_temperatures = max_temperatures.sortBy(ascending = False, keyfunc=lambda k: k[1])
min_temperatures = min_temperatures.sortBy(ascending = False, keyfunc=lambda k: k[1])

# print(min_temperatures.collect())
# print(max_temperatures.collect())
min_temperatures.saveAsTextFile("output/1_min_temps")
max_temperatures.saveAsTextFile("output/1_max_temps")

# 2
date_temperature = lines.map(lambda x: (x[1][0:7], float(x[3])))
date_temperature = date_temperature.filter(lambda x: int(x[0][0:4])>=1950 and int(x[0][0:4])<=2014 and x[1]>=10)

count_month_occ = date_temperature.map(lambda x: (x[0], 1))
count_month_occ = count_month_occ.reduceByKey(lambda a,b:a+b)

# print(count_month_occ.collect())
count_month_occ.saveAsTextFile("output/2_count_readings")

# distinct readings from each station
date_temperature = lines.map(lambda x: (x[1][0:7], (x[0],float(x[3]))))
date_temperature = date_temperature.filter(lambda x: int(x[0][0:4])>=1950 and int(x[0][0:4])<=2014 and x[1][1]>=10)

date_temperature = date_temperature.map(lambda x: (x[0], x[1][0])).distinct()

count_month_occ = date_temperature.map(lambda x: (x[0], 1))
count_month_occ = count_month_occ.reduceByKey(lambda a,b:a+b)

# print(count_month_occ.collect())
count_month_occ.saveAsTextFile("output/2_count_distinct_readings")

# 3
date_temperature = lines.map(lambda x: ((x[1], x[0]), float(x[3])))
date_temperature = date_temperature.filter(lambda x: int(x[0][0][0:4])>=1960 and int(x[0][0][0:4])<=2014)

min_max_temperatures = date_temperature.reduceByKey(max) + date_temperature.reduceByKey(min)
month_temperatures = min_max_temperatures.map((lambda x: ((x[0][0][0:7], x[0][1]), x[1])))
sum_month_temperatures = month_temperatures.reduceByKey(lambda a,b: a+b)
avg_month_temperatures = sum_month_temperatures.map(lambda x: ((x[0][0], x[0][1]), round(x[1]/62, 1)))

# print(avg_month_temperatures.collect())
avg_month_temperatures.saveAsTextFile("output/3_avg_monthly_temp")

# 4
station_precipitation = prec_lines.map(lambda x: ((x[0], x[1]), float(x[3])))
station_temperature = lines.map(lambda x: (x[0], float(x[3])))

station_precipitation = station_precipitation.reduceByKey(lambda a,b: a+b)
station_precipitation = station_precipitation.map(lambda x: (x[0][0], x[1]))

station_precipitation = station_precipitation.reduceByKey(max)
station_temperature = station_temperature.reduceByKey(max)

station_precipitation = station_precipitation.filter(lambda x: x[1]>=100 and x[1]<=200)
station_temperature = station_temperature.filter(lambda x: x[1]>=25 and x[1]<=30)

station_precipitation_temperature = station_precipitation.join(station_temperature)

# print(station_precipitation_temperature.collect())
station_precipitation_temperature.saveAsTextFile("output/4_station_precipitation_temperature")

# 5
og_stations = ostergotland_lines.map(lambda x: (x[0])).collect()
og_stations = sc.broadcast(og_stations).value
station_precipitation = prec_lines.map(lambda x: ((x[1], x[0]), float(x[3])))

station_precipitation = station_precipitation.filter(lambda x: int(x[0][0][0:4])>=1993 
                                                     and int(x[0][0][0:4])<=2016 
                                                     and x[0][1] in og_stations)

daily_prec_for_station = station_precipitation.reduceByKey(lambda x,y: x+y)
total_monthly_prec_for_station = daily_prec_for_station.map(lambda x: ((x[0][0][0:7], x[0][1]), x[1])).reduceByKey(lambda x,y: x+y)
sum_of_month_all = total_monthly_prec_for_station.map(lambda x: (x[0][0], x[1])).reduceByKey(lambda x,y: x+y)
number_of_stations_in_month = total_monthly_prec_for_station.map(lambda x: (x[0][0], 1)).reduceByKey(lambda x,y: x+y)
avg_prec_all_stations = sum_of_month_all.union(number_of_stations_in_month).reduceByKey(lambda x,y: x/y)
avg_prec_all_stations = avg_prec_all_stations.map(lambda x: (x[0], round(x[1],1)))

# print(prec_avg.collect())
prec_avg.saveAsTextFile("output/5_ostergotland_monthly_avg_prec")
