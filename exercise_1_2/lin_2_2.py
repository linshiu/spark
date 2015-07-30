########################################################################
# Find the two beats that are adjacent with the highest correlation 
# in the number of crime events
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
#outFileName = 'lin_2_2.txt'

# inputPath  = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"
# outputPath = "hdfs://wolf.iems.northwestern.edu/user/huser76/output"

inputPath = args.inputPath
outFileName = args.outFileName

# Plan
# 1) Aggregate data at year and beatlevel. 
#    For each beat, get the total number of crimes  per year. 
# 2) Create vector RDD where each vector is an year, and values are
#    number of crimes of each beat
# 3) Find correlation between beats and choose Top 10

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

# select relevant columns 2: date,10: Beat (key = beat, value=date)
crimes = parts.map(lambda p: (p[10].strip(),datetime.strptime(p[2],
 '%m/%d/%Y %I:%M:%S %p').date())).filter(lambda x: x[0].isdigit())

# beats should be numeric. After getting all distinct beats,
# it was found that some were not. For example there are 
# cases where beats are "SCHOOL, PUBLIC, BUILDING"
# which will cause errors when splitting by comma
# This represents 3.6% of the data, so it was removed
# 304 Unique beats remain (6 deleted that resulted from split ",")

# crimes.persist()

#### Prepare Data For Correlation ##########################################################

#### create relevant fields
# (beat, year)
crimes2 = crimes.map(lambda x: (x[0],x[1].year))

#crimes2 = crimes2.sample(False, 0.01,81)
#crimes2.persist()

#### aggregate by key
# ((beat,year), # crimes)
crimes3 = crimes2.map(lambda x: ((x[0],x[1]),1)).reduceByKey(lambda x,y: x+y) # 4340 records

crimes3.persist()

#### get all possible combinations beat year
# (beat, year)
# want all possible combinations since some yeras don't have all beats
# and these should be represented as having zero crimes (not null)
# this is also important for correlation since each beat should have
# the same length (number of records)
beats = crimes3.map(lambda x: x[0][0]).distinct().sortBy(lambda x: x ) # 304 records
years = crimes3.map(lambda x: x[0][1]).distinct().sortBy(lambda x: x ) #  15 records
beatsYears = beats.cartesian(years).map(lambda x: (x,0)) # 4560 records

# create beats dictionary (key: index, value: beat) to use to match
# index in matrix to beat
beatsDic = dict(beats.zipWithIndex().map(lambda x: (x[1],x[0])).collect())

#### join with aggregated data (beat,year) so can get beat,years withn no crimes
# if None, means that beat-year combination didn't exist (had no crimes)
# ((beat,year),(0, # crimes or None))
joined = beatsYears.leftOuterJoin(crimes3) # 4560 records

# replace none with zero
joined2 = joined.map(lambda x: (x[0],x[1][0]) if x[1][1] is None else (x[0],x[1][1]))

# (year, (beat, # crimes))
joined3 = joined2.map(lambda x: (x[0][1], (x[0][0],x[1])))

#### aggregate by year
# (year, [ (beat,#crimes), (beat,#crimes)..])
# this actually looks like: 
# [(2001, <pyspark.resultiterable.ResultIterable object at 0x23e88d0>),..]
# http://stackoverflow.com/questions/29717257/pyspark-groupbykey-returning-pyspark-resultiterable-resultiterable
crimesByYear = joined3.groupByKey()

#### sort # crimes by beat
# (year, [ (beat1,#crimes), (beat2,#crimes)..])
crimesByYear2 = crimesByYear.map(lambda x: (x[0],sorted(x[1], key = lambda t: t[0] )))

# remove beats column
crimesByYear3 = crimesByYear2.map(lambda x: (x[0],[t[1] for t in x[1]]))

#### convert python list of crimes to Vectors RDD[Vector]
# each vector represents a year, with values corrresponding to crimes for each beat
# years are the "rows", and beats are "columns"
crimesVectors = crimesByYear3.map(lambda x: Vectors.dense(x[1]))

crimes3.unpersist()

####  Correlation ##########################################################

# If a single RDD of Vectors is passed in, a correlation 
# matrix comparing the columns in the input RDD is returned.

# If you want to explore your data it is best to compute both, since 
# the relation between the Spearman (S) and Pearson (P) correlations will give some information. Briefly, 
# S is computed on ranks and so depicts monotonic relationships while P is on true values and depicts linear relationships.
# http://stats.stackexchange.com/questions/8071/how-to-choose-between-pearson-and-spearman-correlation
pearsonCorr = Statistics.corr(crimesVectors)
spearmanCorr = Statistics.corr(crimesVectors, method="spearman")
print pearsonCorr
print spearmanCorr
type(pearsonCorr)

# Check dimension should be #beats, #beats
pearsonCorr.shape

# create correlation dictionary function
def createCorrDic(corr):
	"""
	Key: (i,j), Value: value 
	So (i,j) represents to index for beats, value the correlation value between them

	@ param: correlation matrix
	@ return: dictionary
	"""

	# more general version: 
	# http://stackoverflow.com/questions/9545139/python-2d-array-to-dict
	# dict(((j,i), arr[i][j]) for i in range(len(arr)) for j in range(len(arr[0])) if i<j)

	return dict(((i,j), corr[i][j]) for i in range(corr.shape[0]) for j in range(corr.shape[0]) if i<j)

# create dictionary
corrSDic = createCorrDic(spearmanCorr)
corrPDic = createCorrDic(pearsonCorr)

# top keys from dictionary
# http://stackoverflow.com/questions/7197315/5-maximum-values-in-a-python-dictionary
# http://stackoverflow.com/questions/11902665/top-values-from-dictionary
# http://stackoverflow.com/questions/5352546/best-way-to-extract-subset-of-key-value-pairs-from-python-dictionary-object
def findTopK(d, d_ref, k = 10):
	"""
	Top k values of dictionary

	@ param d: correlation dictionary (corrSDic) key: (beat1,beat2), value = corr
	@ param d_ref: dictionary for reference to use the beats instead of index of beats
	               key: index, value: beat
	@ param: optional top k, default is 10
	@ return: array with [(beats1, beats2, corr)]
	"""

	topKeys = sorted(d, key=d.get, reverse=True)[:k]
	return [(d_ref[k[0]], d_ref[k[1]], round(d[(k[0],k[1])],3)) for k in topKeys]

# get top 10
top10 = findTopK(corrPDic, beatsDic)

with open(outFileName, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(["beat1", "beat2", "Pearson_Corr"])
    writer.writerows(top10)


