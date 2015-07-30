########################################################################
#  Find the top 10 blocks in crime events in the last 3 years,
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
#outFileName = 'lin_2_1.txt'
inputPath = args.inputPath
outFileName = args.outFileName

#inputPath  = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"
#outputPath = "hdfs://wolf.iems.northwestern.edu/user/huser76/output"

# Plan
# 1) Filter last 3 years
# 2) Aggregate counts of crimes (rows) by block
# 3) Sort by counts
# 4) Get top 10

# Assumption: interpret last 3 years as 2013,2014 and 2015 
# and not 3 years from the last data point

#### Load Data and process #############################################

# sc is an existing SparkContext.
from datetime import datetime, date
from pyspark import SparkContext
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

# select relevant columns 2: date,3: Block, 5,801,844 records
crimes = parts.map(lambda p: (datetime.strptime(p[2],
 '%m/%d/%Y %I:%M:%S %p').date(),p[3]))

# There are some blocks with no name, remove these
blockNameLengths = crimes.map(lambda x: len(x[1])).distinct().sortBy(lambda x: x)
# blockNameLengths.collect()
crimes2 = crimes.filter(lambda x: len(x[1])>0) # 5,801,842

# Get the last word for the block name to check consistency (Ave, Av, St, etc)
# This is an issue because 008XX N MICHIGAN AVE = 008XX N MICHIGAN AV but aggregating
# will be counted as separte blocks 
# http://www.gossamer-threads.com/lists/python/python/777013
# http://stackoverflow.com/questions/19454412/regex-to-find-last-word-in-a-string-python
lastWord = crimes2.map(lambda x: x[1].rsplit(None,1)[1]).distinct().sortBy(lambda x: x)
# lastWord.collect()
# Inconsistencies found: 
# AV, AV.,AVE, AVENUE, AVE,AVE`, Ave, ave
# B, BL, BLVD, BV, Blvd, 
# DR, DR., DRIVE, Dr
# HW, HWY
# PK, PKWY, PW, Pkwy, 
# PL, PLACE, Pl, pl
# RD, RD., Rd
# ST., ST, st
# WAY, WY

# Convert All to Upper Case and remove leading and trailing whitespace
crimes3 = crimes2.map(lambda x: (x[0],x[1].upper().strip()))

streetSuffix = {'AV': 'AVE', 'AVE`':'AVE', 'AV.': 'AVE', 'AVENUE': 'AVE',
	'B': 'BLVD', 'BL': 'BLVD', 'BV': 'BLVD',
	'DR.':'DR','DRIVE':'DR','HW':'HWY','PK':'PKWY','PW':'PKWY',
	'PLACE':'PL','RD.':'RD','ST.':'ST','STREET':'ST','WY':'WAY'}

# Note: One should also check other words not only the endings (North, N, etc) are consistent

# http://stackoverflow.com/questions/6266727/python-cut-off-the-last-word-of-a-sentence
def fixSuffix(s):
	"""
	Makes the street suffixes consistent
	@ param: string
	@ return: fixed string with replaced suffix
	"""
	end = s.rsplit(None,1)[1] # get suffix
	if end in streetSuffix:
		s = s.rsplit(' ', 1)[0] # remove current suffix
		s = s + " " + streetSuffix[end] # add new suffix
	return s

crimes4 = crimes3.map(lambda x: (x[0],fixSuffix(x[1])))

# Check it was fixed
lastWord2 = crimes4.map(lambda x: x[1].rsplit(None,1)[1]).distinct().sortBy(lambda x: x)
#lastWord2.collect()

# crimes.persist()

#### Find TOP 10 #######################################################

filtered = crimes4.filter(lambda x: x[0].year >= 2012 and  x[0].year <= 2014)     
pairs = filtered.map(lambda x: (x[1],1))                 
counts = pairs.reduceByKey(lambda x,y: int(x) + int(y))

# switch key, values and sort by key (number of crimes) and take top 10
top10 = counts.map(lambda x: (x[1], x[0])).sortByKey(ascending = False).take(10) # this is a py list
top10rdd =  sc.parallelize(top10) # this is rdd
resultsRDD = top10rdd.map(lambda x: '{0}, {1}'.format(x[1], x[0])) # format into one string per row

# Save to hdfs (need to do getmerge after)
#header = sc.parallelize(['block, crime events'])
#header.union(resultsRDD).saveAsTextFile(outputPath)

# Alternative save directly csv to local directory
resultsArray = resultsRDD.collect()

with open(outFileName, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['block','crime_events'])
    for line in resultsArray:
    	writer.writerow(line.split(","))


#datetime.date(2015, 1,1)
#crimes.unpersist()