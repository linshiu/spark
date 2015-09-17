########################################################################
# By using SparkSQL generate a histogram of average 
# crime events by month. Find an explanation of results
########################################################################

"""
Two Arguments:

@param: inputPathFile (in hdfs)
@param: outputFileName (will be saved in current local directory)

"""

import argparse
import os
import sys

parser = argparse.ArgumentParser(description='*** lin_1.py ***')
parser.add_argument("inputPath", help="Input FILE path in Hadoop with dataset")
parser.add_argument("outFileName", help="Output File Name")
args = parser.parse_args()

# Note: Example taken from Spark SQL website
# https://spark.apache.org/docs/latest/sql-programming-guide.html

# path for input and output
# inputPath  = "/Users/Steven/Documents/Spark/sample1.csv"
# outputPath = "/Users/Steven/Documents/Spark/output"

#inputPath  = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"
#outputPath = "hdfs://wolf.iems.northwestern.edu/user/huser76/output"

inputPath = args.inputPath
outFileName = args.outFileName

# Plan
# 1) Get counts of rows (# crimes) by month
# 2) Get counts of years (distinct) by month (# of months in dataset)
# 3) Divide # crimes by # of months to get average # crimes by month

# Note: cannot divide 1) by 15 since some months appear in 14 years
# due to dataset not containing entire calendar year
# Note: python 2: so integer divided by integer results in integer

#### Load Data and process #############################################

# sc is an existing SparkContext.
from datetime import datetime
from pyspark import SparkContext
from pyspark.sql import SQLContext, Row
import csv
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

# extract year and month of crime

# https://docs.python.org/2/library/datetime.html
def getMonth(dateString):
	date_object = datetime.strptime(dateString, '%m/%d/%Y %I:%M:%S %p')
	return date_object.month
	# return date_object.strftime('%B')

def getYear(dateString):
	date_object = datetime.strptime(dateString, '%m/%d/%Y %I:%M:%S %p')
	return date_object.year

# This method was done to practice, it could be done more efficient using:
# month=int(p[2][0:2]), year= int(p[2][6:10])))
# because date object is not created and grouping by/joining using numeric
crimes = parts.map(lambda p: Row(month=getMonth(p[2]), year=getYear(p[2])))

# Infer the schema, and register the SchemaRDD as a table.
schemaCrimes= sqlContext.inferSchema(crimes)
schemaCrimes.registerTempTable("crimes")

#schemaCrimes.printSchema()
#schemaCrimes.show(10)
#schemaCrimes

#### Get Averages ##################################################

avg = sqlContext.sql("SELECT month, CAST(COUNT(*) AS FLOAT)/COUNT(DISTINCT(year)) as average FROM crimes GROUP BY month")
#avg.show()
#avg
#avg.printSchema()

# c=avg.collect()
# rdd=sc.parallelize(c)
# rdd.saveAsTextFile(outputPath)
# output looks like:
# Row(month=1, average=19962.0)

# The results of SQL queries are RDDs and support all the normal RDD operations.
results = avg.map(lambda p: '{0}, {1}'.format(p.month, p.average))

# Save to hdfs (need to do getmerge after)
#header = sc.parallelize(['month, average'])
#header.union(results).saveAsTextFile(outputPath)

# Alternative save directly csv to local directory
resultsArray = results.collect()

with open(outFileName, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['month','average_crimes'])
    for line in resultsArray:
    	writer.writerow(line.split(","))