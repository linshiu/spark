########################################################################
# Predict the number of crime events in the next week at the beat level.
# the higher the IUCR is, the more severe the crime is. Violent crime events
# are more important and thus it is desirable that they are forecasted more accurately.
########################################################################

# http://www.ncdc.noaa.gov/cdo-web/datasets

# Weather data was chosen to take into account sesonality that was seen in the previous
# problems
# Weather data at daily level for chicago from 1995-Current
# Run from command line:
# wget http://academic.udayton.edu/kissock/http/Weather/gsod95-current/ILCHICAG.txt
# hadoop fs -put ILCHICAG.txt assignment5

#inputPathW = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/ILCHICAG.txt"

"""
Two Arguments:

@param: inputPathFile (in hdfs)
@param: outputFileName (will be saved in current local directory)

"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description='*** lin_3.py ***')
parser.add_argument("inputPath", help="Input FILE path in Hadoop with dataset")
parser.add_argument("outFileName", help="Output File Name")
args = parser.parse_args()

# path for input and output
#inputPath  = "/Users/Steven/Documents/Spark/Crimes_-_2001_to_present.csv"
#outputPath = "/Users/Steven/Documents/Spark/output"
#outFileName = 'lin_2_2.txt'

# inputPath  = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"
# outputPath = "hdfs://wolf.iems.northwestern.edu/user/huser76/output"

inputPath = args.inputPath
outFileName = args.outFileName


# Plan
# 1) Aggregate crimes at beat, year-weeklylevel
# 2) Get Average temperature at year-week level
# 3) Combine datasets
# 4) Random Forest
# 5) Prediction Next Week

#### Load Data and process #############################################

# sc is an existing SparkContext.
from datetime import datetime, date

from pyspark import SparkContext
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors

import math
import csv

from numpy import array
import numpy as np
from pyspark.sql import SQLContext, Row

sc = SparkContext()
sqlContext = SQLContext(sc)

# Load a text file and convert each line to a dictionary.
lines = sc.textFile(inputPath)

# get rid of first line which is a header
lines2 = lines.zipWithIndex().filter(lambda line: line[1]>0).map(lambda line: line[0])

# split to columns and create rows object
# Note: ok to split just by "," since commas within quotes appear for 
# variables after the relevant variables for this problem
parts = lines2.map(lambda line: line.split(","))

def getDateTime(dateString):
	date_object = datetime.strptime(dateString, '%m/%d/%Y %I:%M:%S %p')
	return date_object
	# return date_object.strftime('%B')

# extract date
# 5,595,321
# [(year,month,weekday,hour)]
# the day of the week as an integer, where Monday is 0 and Sunday is 6
# hour in range (24) format
crimes = parts.map(lambda p: (getDateTime(p[2]).year,getDateTime(p[2]).month,
	getDateTime(p[2]).weekday(),
	getDateTime(p[2]).hour))

crimes = crimes.map(lambda p: Row(year=p[0],month=p[1],
	weekday=p[2],
	hour=p[3]))

# Infer the schema, and register the SchemaRDD as a table.
schemaCrimes= sqlContext.inferSchema(crimes)
schemaCrimes.registerTempTable("crimes")

#schemaCrimes.printSchema()
#schemaCrimes.show(10)
#schemaCrimes

#### SQL ##################################################

## Save File Formula
def saveFile(sub, header, data, ext="txt",rootFile=outFileName, delim="."):
	"""

	Saves RDD to File. Ex. if outputFileName = line_4.txt,
	each RDD will be saved as lin_4_<sub>.txt

	@param sub: subName
	@param header: header in file
	@param data: RDD

	"""

	outputFileNameTemp = "{0}_{1}.{2}".format(outFileName.split(delim)[0],sub,ext)
	
	with open(outputFileNameTemp , 'wb') as f:
	    writer = csv.writer(f)
	    writer.writerow(header)
	    for line in data:
	    	writer.writerow(line.split(","))

# Notes Queries are not efficient, could have created a general table with counts and then
# query that table

query = sqlContext.sql("SELECT month, CAST(COUNT(*) AS FLOAT)/COUNT(DISTINCT(year)) as average FROM crimes GROUP BY month")
results = query.map(lambda p: '{0},{1}'.format(p.month, p.average))
resultsArray = results.collect()
saveFile("byMonth", ['month','average'], resultsArray)

query = sqlContext.sql("SELECT weekday, CAST(COUNT(*) AS FLOAT)/COUNT(DISTINCT(year)) as average FROM crimes GROUP BY weekday")
results = query.map(lambda p: '{0},{1}'.format(p.weekday, p.average))
resultsArray = results.collect()
saveFile("byWeekday", ['weekday','average'], resultsArray)

query = sqlContext.sql("SELECT hour, CAST(COUNT(*) AS FLOAT)/COUNT(DISTINCT(year)) as average FROM crimes GROUP BY hour")
results = query.map(lambda p: '{0},{1}'.format(p.hour, p.average))
resultsArray = results.collect()
saveFile("byHour", ['hour','average'], resultsArray)

query = sqlContext.sql("SELECT weekday,hour, CAST(COUNT(*) AS FLOAT)/COUNT(DISTINCT(year)) as average FROM crimes GROUP BY weekday,hour")
results = query.map(lambda p: '{0},{1},{2}'.format(p.weekday, p.hour, p.average))
resultsArray = results.collect()
saveFile("byWeekDayHour", ['weekday','hour','average'], resultsArray)

query = sqlContext.sql("SELECT month,weekday, CAST(COUNT(*) AS FLOAT)/COUNT(DISTINCT(year)) as average FROM crimes GROUP BY month,weekday")
results = query.map(lambda p: '{0},{1},{2}'.format(p.month, p.weekday, p.average))
resultsArray = results.collect()
saveFile("byMonthWeekDay", ['month','weekday','average'], resultsArray)

#crimesbyMonth.show()
#crimesbyMonth.printSchema()
#crimesbyMonth 

# c=avg.collect()
# rdd=sc.parallelize(c)
# rdd.saveAsTextFile(outputPath)
# output looks like:
# Row(month=1, average=19962.0)

# Save to hdfs (need to do getmerge after)
#header = sc.parallelize(['month, average'])
#header.union(results).saveAsTextFile(outputPath)
# Alternative save directly csv to local directory

# The results of SQL queries are RDDs and support all the normal RDD operations.