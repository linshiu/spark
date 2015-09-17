########################################################################
# Establish if the number of crime events is different between Majors Daly 
# and Emanuel at a granularity of your choice (not only at the city level). 
# Find an explanation of results
########################################################################
"""
Two Arguments:

@param: inputPathFile (in hdfs)
@param: outputFileName (will be saved in current local directory)

"""
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='*** lin_2_1.py ***')
parser.add_argument("inputPath", help="Input FILE path in Hadoop with dataset")
parser.add_argument("outFileName", help="Output File Name")
args = parser.parse_args()

# Note: Example taken from Spark SQL website
# https://spark.apache.org/docs/latest/sql-programming-guide.html

# path for input and output
#inputPath  = "/Users/Steven/Documents/Spark/Crimes_-_2001_to_present.csv"
#outputPath = "/Users/Steven/Documents/Spark/output"

# inputPath  = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"
# outputPath = "hdfs://wolf.iems.northwestern.edu/user/huser76/output"

inputPath = args.inputPath
outFileName = args.outFileName


# Plan
# 1) Create the datasets for daley and emanuel by filtering date
# 2) Aggregate counts of crimes (rows) by district
# 3) Normalize crimes by days (since Daley spent more years in office)
#    to get the average monthly crimes by district
# 4) Take the difference between the two mayors for the same districts
# 5) Conduct paired t-test to test signficance difference between mayors

# Emanuel: May 16, 2011 - Present
# Daley: April 24, 1989 - May 16, 2011
# http://en.wikipedia.org/wiki/Rahm_Emanuel
# http://en.wikipedia.org/wiki/Richard_M._Daley

#### Load Data and process #############################################

# sc is an existing SparkContext.
from datetime import datetime, date
from pyspark import SparkContext
from pyspark.mllib.stat import Statistics
from pyspark.mllib.linalg import Vectors
import math
import csv
#from pyspark.sql import SQLContext, Row
sc = SparkContext()
#sqlContext = SQLContext(sc)

# Load a text file and convert each line to a dictionary.
lines = sc.textFile(inputPath)

# get rid of first line which is a header
lines2 = lines.zipWithIndex().filter(lambda line: line[1]>0).map(lambda line: line[0])

# split to columns and create rows object
# Note: ok to split just by "," since commas within quotes appear for 
# variables after the relevant variables for this problem
parts = lines2.map(lambda line: line.split(","))

# select relevant columns 2: date,11: District (key = district, value=date)
crimes = parts.map(lambda p: (p[10],p[11],datetime.strptime(p[2],
 '%m/%d/%Y %I:%M:%S %p').date())).filter(lambda x: x[0].isdigit()).map(lambda x: (x[1],x[2]))

# note that beats can have commas inside quotes which will mess up
# the districts since it comes after beats

# beats should be numeric. After getting all distinct beats,
# it was found that some were not. For example there are 
# cases where beats are "SCHOOL, PUBLIC, BUILDING"
# which will cause errors when splitting by comma
# This represents 3.6% of the data, so it was removed
# 304 Unique beats remain (6 deleted that resulted from split ",")

# crimes.persist()

#### Conduct Test #######################################################

# 1) Split by major
daley = crimes.filter(lambda x: x[1] < date(2011, 05, 16))  
emanuel =  crimes.filter(lambda x: x[1] >= date(2011, 05, 16))

daley.persist()
emanuel.persist()

# 2) and 3) Aggregates by count and normalize 

pairsD = daley.map(lambda x: (x[0],1)) # key is district       
countsD = pairsD.reduceByKey(lambda x,y: int(x) + int(y))
monthsD = daley.map(lambda x: x[1].strftime('%Y-%m')).distinct().count() # months Daley in data

pairsE = emanuel.map(lambda x: (x[0],1)) # key is district         
countsE = pairsE.reduceByKey(lambda x,y: int(x) + int(y))
monthsE = emanuel.map(lambda x: x[1].strftime('%Y-%m')).distinct().count() # months Emanuel in data

# type(monthsD)

avgD = countsD.map(lambda x: (x[0],float(x[1])/monthsD))
avgE = countsE.map(lambda x: (x[0],float(x[1])/monthsE))

daley.unpersist()
emanuel.unpersist()

# 4) Take the difference (join returns block, (avgD, avgE))
joined = avgD.join(avgE).map(lambda x: x[1][1] - x[1][0])

joined.persist()

d = joined.mean()
n = joined.count()
s = joined.stdev()
t = d/(s/math.sqrt(n))

# alternative use Vectors mllib
# joined2 = joined.map(lambda x: Vectors.dense(x))
# summary = Statistics.colStats(joined2)

# print summary.mean()
# print summary.count()
# print summary.variance()

joined.unpersist()

# p-value 
# Save to hdfs (need to do getmerge after)
#results = sc.parallelize(['p-value paired t-test', t]).saveAsTextFile(outputPath)

# Alternative save directly csv to local directory

with open(outFileName, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['t-test', t])

