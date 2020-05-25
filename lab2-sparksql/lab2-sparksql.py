from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
from pyspark.sql import functions as F

sc = SparkContext(appName = "lab2-sparksql")
sqlContext = SQLContext(sc)

temp_rdd = sc.textFile("data/temperature-readings.csv")
temp_parts = temp_rdd.map(lambda l: l.split(";"))

prec_rdd = sc.textFile("data/precipitation-readings.csv")
prec_parts = prec_rdd.map(lambda l: l.split(";"))

og_rdd = sc.textFile("data/stations-Ostergotland.csv")
og_parts = og_rdd.map(lambda l: l.split(";"))

# 1
tempReadingsRow = temp_parts.map(lambda p: (p[0], int(p[1].split("-")[0]), float(p[3])))
tempReadingsString = ["station", "year", "value"]

schemaTempReadings = sqlContext.createDataFrame(tempReadingsRow, tempReadingsString)
schemaTempReadings.registerTempTable("tempReadingsTable")

schemaTempReadingsMin = sqlContext.sql("SELECT year, FIRST(station) as station_id, MIN(value) AS yearlymin \
                                        FROM tempReadingsTable \
                                        WHERE year>=1950 AND year<=2014 \
                                        GROUP BY year \
                                        ORDER BY yearlymin DESC")

schemaTempReadingsMax = sqlContext.sql("SELECT year, FIRST(station) as station_id, MAX(value) AS yearlymax \
                                        FROM tempReadingsTable \
                                        WHERE year>=1950 AND year<=2014 \
                                        GROUP BY year \
                                        ORDER BY yearlymax DESC")

schemaTempReadingsMin.show()
schemaTempReadingsMax.show()

# 2
tempReadingsRow = temp_parts.map(lambda p: (int(p[1].split("-")[0]), int(p[1].split("-")[1]), float(p[3])))
tempReadingsString = ["year", "month", "value"]

schemaTempReadings = sqlContext.createDataFrame(tempReadingsRow, tempReadingsString)
schemaTempReadings.registerTempTable("tempReadingsTable")

tempReadings_Higher10 = schemaTempReadings.filter((F.col("year") >= 1950) & (F.col("year") <= 2014) & (F.col("value") >= 10)) \
.groupBy("year", "month").agg(F.count(F.lit(1)).alias("count")) \
.orderBy("count", ascending=False)

tempReadings_Higher10.show()

# distinct readings from each station
tempReadingsRow = temp_parts.map(lambda p: (p[0], int(p[1].split("-")[0]), int(p[1].split("-")[1]), float(p[3])))
tempReadingsString = ["station", "year", "month", "value"]

schemaTempReadings = sqlContext.createDataFrame(tempReadingsRow, tempReadingsString)
schemaTempReadings.registerTempTable("tempReadingsTable")

tempReadings_Higher10_distinct = schemaTempReadings.filter((F.col("year") >= 1950) & (F.col("year") <= 2014) & (F.col("value") >= 10)) \
.groupBy("year", "month").agg(F.countDistinct(F.col("station")).alias("count")) \
.orderBy("count", ascending=False)

tempReadings_Higher10_distinct.show()

# 3
tempReadingsRow = temp_parts.map(lambda p: (p[0], int(p[1].split("-")[0]), int(p[1].split("-")[1]), int(p[1].split("-")[2]), float(p[3])))
tempReadingsString = ["station", "year", "month", "day", "value"]

schemaTempReadings = sqlContext.createDataFrame(tempReadingsRow, tempReadingsString)
schemaTempReadings.registerTempTable("tempReadingsTable")

avgMonthlyTempStation = schemaTempReadings.select("year", "month", "day", "station", "value") \
.filter((F.col("year") >= 1960) & (F.col("year") <= 2014)) \
.groupBy("year", "month", "day", "station").agg(F.max("value").alias("max_day"), F.min("value").alias("min_day")) \
.groupBy("year", "month", "station").agg(F.sum("max_day").alias("sum_max_day"), F.sum("min_day").alias("sum_min_day")) \
.groupBy("year", "month", "station").agg(F.round(F.sum((F.col("sum_max_day") + F.col("sum_min_day"))/62),1).alias("avgMonthlyTemperature")) \
.orderBy("avgMonthlyTemperature", ascending=False)

avgMonthlyTempStation.show()

# 4
tempReadingsRow = temp_parts.map(lambda p: (p[0], float(p[3])))
tempReadingsString = ["station", "temperature"]
precReadingsRow = prec_parts.map(lambda p: (p[0], int(p[1].split("-")[0]), int(p[1].split("-")[1]), int(p[1].split("-")[2]), float(p[3])))
precReadingsString = ["station", "year", "month", "day", "precipitation"]

schemaTempReadings = sqlContext.createDataFrame(tempReadingsRow, tempReadingsString)
schemaTempReadings.registerTempTable("tempReadingsTable")
schemaPrecReadings = sqlContext.createDataFrame(precReadingsRow, precReadingsString)
schemaPrecReadings.registerTempTable("precReadingsTable")

station_precipitation = schemaPrecReadings.select("station", "year", "month", "day", "precipitation") \
.groupBy("station", "year", "month", "day").agg(F.sum(F.col("precipitation")).alias("daily_prec")) \
.groupBy("station").agg(F.max(F.col("daily_prec")).alias("max_daily_prec")) \
.filter((F.col("max_daily_prec") >= 100) & (F.col("max_daily_prec") <= 200)) \
.orderBy("station", ascending=False)

station_temperature  = schemaTempReadings.select("station", "temperature") \
.groupBy("station").agg(F.max(F.col("temperature")).alias("max_temp")) \
.filter((F.col("max_temp") >= 25) & (F.col("max_temp") <= 30)) \
.orderBy("station", ascending=False)

# station_precipitation.show() # only 4 stations with dailt max prec between 100 and 200
# station_temperature.show()

sp = station_precipitation.alias("sp")
st = station_temperature.alias("st")
stationPrecTemp = sp.join(st, ["station"], "inner")

stationPrecTemp.show() # empty table - no match between stations in daily max prec and max temp

# 5
og_stations_col = og_parts.map(lambda p: (p[0], p[1]))
og_stations_str = ["station", "name"]
precReadingsRow = prec_parts.map(lambda p: (p[0], int(p[1].split("-")[0]), int(p[1].split("-")[1]), int(p[1].split("-")[2]), float(p[3])))
precReadingsString = ["station", "year", "month", "day", "precipitation"]

schemaOGstations = sqlContext.createDataFrame(og_stations_col, og_stations_str)
schemaOGstations.registerTempTable("OGstationsTable")
schemaPrecReadings = sqlContext.createDataFrame(precReadingsRow, precReadingsString)
schemaPrecReadings.registerTempTable("precReadingsTable")

og_stations = schemaOGstations.select("station")

sp = schemaPrecReadings.alias("sp")
og = og_stations.alias("og")
OG_station_Prec = sp.join(og, ["station"], "inner")

OG_station_Prec = OG_station_Prec.select("station", "year", "month", "day", "precipitation") \
.filter((F.col("year") >= 1993) & (F.col("year") <= 2016)) \
.groupBy("station", "year", "month", "day").agg(F.sum(F.col("precipitation")).alias("daily_sum_st")) \
.groupBy("station", "year", "month").agg(F.sum(F.col("daily_sum_st")).alias("monthly_sum_st")) \
.groupBy("year", "month").agg(F.sum(F.col("monthly_sum_st")).alias("monthly_sum_all"), F.count(F.col("monthly_sum_st")).alias("number_of_stations_in_month")) \
.select("year", "month", (F.round((F.col("monthly_sum_all")/F.col("number_of_stations_in_month")),1)).alias("avgMonthlyPrecipitation")) \
.orderBy(["year", "month"], ascending=False)

OG_station_Prec.show()
