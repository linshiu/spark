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

inputPathW = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/ILCHICAG.txt"

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

#inputPath  = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"
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

from pyspark.mllib.regression import LabeledPoint, LinearRegressionWithSGD
from numpy import array
import numpy as np

from pyspark.context import SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils

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

# select relevant columns 2: date,10: Beat (key = beat, value=date), 5: Primary Type Crime 
# 5,595,321
# [(date,beat, crime type)]
crimes = parts.map(lambda p: (datetime.strptime(p[2],
 '%m/%d/%Y %I:%M:%S %p').date(),str(p[10]).strip(), str(p[5]).strip().upper())).filter(lambda x: x[1].isdigit())

# beats should be numeric. After getting all distinct beats,
# it was found that some were not. For example there are 
# cases where beats are "SCHOOL, PUBLIC, BUILDING"
# which will cause errors when splitting by comma
# This represents 3.6% of the data, so it was removed
# 304 Unique beats remain (6 deleted that resulted from split ",")

crimes.persist()

# get max and min day to ensure all weeks cover 7 days, otherwise
# the weekly crime count will not be accurate because those weeks might have
# fewer than 7 days
minDay = crimes.map(lambda x: x[0]).min() # datetime.date(2001, 1, 1)
maxDay = crimes.map(lambda x: x[0]).max() # datetime.date(2015, 5, 19)

# min day is ok since week 1 starts Jan 1
# max day 5/19/2015 corresponds to week number 21, which goes from 5/18/2015
# to 5/24/2015. So filter out days < 5/18/2015 to have complete weeks

crimesFiltered = crimes.filter(lambda x: x[0] < date(2015, 5, 18)) # 5,594,019

# NOTE: I forgot to split violent crimes vs non-violent. So I am creating subsets
# and running the entire code again for the two separate datatsets since I won't
# have to make any changes to the code, which is is very inefficient

# Classify violent crimes as ASSAULT or BATTERY
violent = ('ASSAULT', 'BATTERY')
# Comment out either one of the next lines to run for violent or non-violent
crimesFiltered = crimesFiltered.filter(lambda x: x[2] in violent)
#crimesFiltered = crimesFiltered.filter(lambda x: x[2] not in violent)

# key = year, week number, beat, value = 1
# [((year,week,beat),1)]
crimesWeek = crimesFiltered.map(lambda x: ((x[0].year, x[0].isocalendar()[1],x[1]),1))

# aggregate
# [((year,week,beat),count of crimes)]
crimesWeek2 = crimesWeek.reduceByKey(lambda x,y: x+y) # 211,238 records

crimes.unpersist()


crimesWeek2.persist()

# cases where no crimes in a week for a beat, should be counted as zero crimes
# in the dataset instead of not showing up

# [(year, week)]
yearsWeek =crimesWeek2.map(lambda x: (x[0][0], x[0][1])).distinct().sortBy(lambda x: x) # 752
# yearsWeek.count()

# [(beat)]
beats = crimesWeek2.map(lambda x: x[0][2]).distinct().sortBy(lambda x: x ) # 304 records
# beats.count()

# get all possible combinations years-week and beats
# [((beat, (year, week)), 0)] 
beatsYearsWeek = beats.cartesian(yearsWeek).map(lambda x: (x,0)) #  752*304 = 228,608 records
# [((year,week,beat),0)]
beatsYearsWeek = beatsYearsWeek.map(lambda x: ((x[0][1][0],x[0][1][1],x[0][0]),x[1]))
# beatsYearsWeek.count()

# Since data containts 211,238 < 228,608 there are beats that have no crimes in certain year-week
# It is important to set these cases with 0 crimes rather than just not appearing at all in the 
# dataset since the predictions will change for those week-beat combinations

# join crimesWeek2 and beatsYearsWeek
# [((year, week, beat), (0, crimes or None))] #None when year-week-beat combination didn't exist in data
crimesWeek3 = beatsYearsWeek.leftOuterJoin(crimesWeek2) # 228,608 records
# crimesWeek3.count()
# crimesWeek3.filter(lambda x: x[1][1] is None).take(1) # check None

crimesWeek2.unpersist()

# replace none with zero
#[((year, week, beat), # crimes)]
crimesWeek4 = crimesWeek3.map(lambda x: (x[0],x[1][0]) if x[1][1] is None else (x[0],x[1][1]))
# crimesWeek4.filter(lambda x: x[1] is None).take(1) # check None

# rearrange
# key = year, week number, value = beat, count (so can be joined with weather data)
# [((year,week),(beat,count))]
crimesWeek5 = crimesWeek4.map(lambda x: ((x[0][0],x[0][1]),(x[0][2],x[1])))


# Not used: this is for data where beat-weeks with no crimes do not show up
# rearrange
# key = year, week number, value = beat, count (so can be joined with weather data)
# [((year,week),(beat,count))]
# crimesWeek5 = crimesWeek2.map(lambda x: ((x[0][0],x[0][1]),(x[0][2],x[1])))


#### External Data ##########################################################

linesW = sc.textFile(inputPathW)
partsW = linesW.map(lambda line: line.split()) # split by whitespace

# convert to date format and float
# [(date, temp)]
weather = partsW.map(lambda x: (date(int(x[2]),int(x[0]),int(x[1])),float(x[3])))

# filtered so that full weeks (can divide by 7 to get weekly average temperature)
weatherFiltered = weather.filter(lambda x: x[0] >= date(2001, 1, 1) and x[0]<date(2015, 5, 18))

### Aggregate by year, week
# key = year, week number, value = temp
# [((year,week),temp)]
weather2 = weatherFiltered.map(lambda x: ((x[0].year, x[0].isocalendar()[1]),x[1]))

# [((year,week),(temp,1))]
weather3 = weather2.mapValues(lambda x: (x,1))

# [((year,week),(sum of temp, sum of records/days))]
weather4 = weather3.reduceByKey(lambda x,y: (x[0]+y[0], x[1]+y[1]))

# [((year,week),avg Temp)]
avgTemp = weather4.map(lambda x: (x[0],x[1][0]/x[1][1]))

#### Combine Dataset ##########################################################

# join weather and crimes data on year, week
# [((year, week),((beat, #crimes), temp))]
joined = crimesWeek5.join(avgTemp) # 228,608
# joined.count()

# [(#crimes, beat, year, week, temp)]
joined2 = joined.map(lambda x: ((x[1][0][1],x[1][0][0],x[0][0],x[0][1],x[1][1])))

# each row represents a unique year, week, beat combination
# NOTE: don't aggretate by week across years since losing information
# Instead just remove the year column

# Create dictionary starting at index = 0, ending at index = n-1,
# where n is the number of categories for features
# This is because trees for categorical features require start at index 0

# Key: weekn number, value: weeknumber -1  
weekDic = dict(zip(range(1,54), range(0,53)))

# key: value, index
# index in matrix to beat
beatsDic = dict(beats.zipWithIndex().map(lambda x: (x[0],x[1])).collect())

#beats = crimesWeek2.map(lambda x: x[0][2]).distinct().sortBy(lambda x: x ) # 304 records

# change coding of categories and remove year
# [(crimes, [beat, week, temp])]
joined3 = joined2.map(lambda x: (x[0],[beatsDic[x[1]], weekDic[x[3]],x[4]]))

# covert labeled point, (response, [predictors])
dataLP = joined3.map(lambda x: LabeledPoint(x[0],x[1]))

# Split the data into training and test sets (30% held out for testing)
(trainingData, testData) = dataLP.randomSplit([0.7, 0.3])

#### Regression ###############################################################

########### random forest
# https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#module-pyspark.mllib.regression
# https://spark.apache.org/docs/latest/mllib-decision-tree.html
# https://spark.apache.org/docs/latest/mllib-ensembles.html
# https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/random_forest_example.py
from pyspark.context import SparkContext
from pyspark.mllib.tree import RandomForest
from pyspark.mllib.util import MLUtils

#  Train a RandomForest model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
#  Note: Use larger numTrees in practice.
#  Setting featureSubsetStrategy="auto" lets the algorithm choose
 
# Parameters:	
# data - Training dataset: RDD of LabeledPoint. Labels should take values {0, 1, ..., numClasses-1}.
# numClasses - number of classes for classification.
# categoricalFeaturesInfo - Map storing arity of categorical features. E.g., an entry (n -> k)
#                           indicates that feature n is categorical with k categories indexed from 
#                           0: {0, 1, ..., k-1}.
# numTrees - Number of trees in the random forest.
# featureSubsetStrategy - Number of features to consider for splits at each node. 
#                         Supported: "auto" (default), "all", "sqrt", "log2", "nethird".
#                         If "auto" is set, this parameter is set based on numTrees: 
#                         if numTrees == 1, set to "all"; if numTrees > 1 (forest) 
#                         set to "sqrt".
# impurity - Criterion used for information gain calculation. Supported values: "gini" (recommended) or "entropy".
# maxDepth - Maximum depth of the tree. E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes. (default: 4)
# maxBins - maximum number of bins used for splitting features (default: 100)
# seed - Random seed for bootstrapping and choosing feature subsets.

featuresDic = {0: 304, 1: 53} # feature 1 has 53 categories, 0 ..to .. 52 (corresponding to week 1 .. 53)
# [(crimes, [beat, week, temp])]
# feature 0: beat
# feature 1: week
# feature 2: temp
# featuresDic = {} # for all continuous predictors

maxBins = max(len(beatsDic),len(weekDic)) # ecisionTree requires maxBins >= max categories in categorical features (304)

### Fit
model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo=featuresDic,
                                    numTrees=10, featureSubsetStrategy="auto",
                                    impurity='variance', maxDepth=5, maxBins=maxBins)
### Evalute
# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda v_p1: (v_p1[0] - v_p1[1]) * (v_p1[0] - v_p1[1]))\
    .sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression forest model:')
# print(model.toDebugString())

### Compute R2
SSE = labelsAndPredictions.map(lambda v_p1: (v_p1[0] - v_p1[1]) * (v_p1[0] - v_p1[1])).sum()
summary = Statistics.colStats(testData.map(lambda x: Vectors.dense(x.label)))
meanY = float(summary.mean())

# Alternative for mean
# testData.map(lambda x: Vectors.dense(x.label)).mean()
SST = testData.map(lambda y: (y.label-meanY)**2).sum()

n = float(testData.count())
params = 3

Rsqrd = 1 - SSE/SST
RsqrdAdj = 1 - SSE/(n-params)/(SST/(n-1))

print('R-sqruared: {0}'.format(Rsqrd))
print('R-sqruared Adj: {0}'.format(RsqrdAdj))

### Predictions

## Next Week Data
# The maximum week in original data is 21, but it was removed as explained 
# before. So the next week prediction is for week 21
maxWeek = maxDay.isocalendar()[1]
nextWeek = maxWeek

# Source: weather.com for week 21
# Avereage forecast of next week
forecastWeather = np.mean([81,54,51,70,68,83,76])

dataNextWeek = beats.map(lambda x: [beatsDic[x], weekDic[nextWeek], forecastWeather])

predictionsNextWeek = model.predict(dataNextWeek)

# [((week, prediction), beat)]
results = predictionsNextWeek.map(lambda x: (nextWeek, x)).zip(beats)

# [[week, beat, prediction]]
results = results.map(lambda x: [x[0][0], x[1], x[0][1]])
resultsOutput = results.collect()


#### Results ###############################################################

with open(outFileName, 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Test MSE: {0}'.format(testMSE)])
    writer.writerow(['R-sqruared Adj: {0}'.format(RsqrdAdj)])
    writer.writerow(['**************'])
    writer.writerow(["week", "beat", "prediction"])
    writer.writerows(resultsOutput)

"""
with open("lin_3_Rsq.txt", 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['R-sqruared Adj: {0}'.format(RsqrdAdj)])

with open("lin_3_MSE.txt", 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(['Test MSE: {0}'.format(testMSE)])
"""

#### Note Used ##########################################################

"""
# Note: Don't use because Random Forest Has Better Performan ce

########## Decision Tree
# https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#module-pyspark.mllib.regression
# https://spark.apache.org/docs/latest/mllib-decision-tree.html
from pyspark.mllib.regression import LabeledPoint
from pyspark.mllib.tree import DecisionTree, DecisionTreeModel
from pyspark.mllib.util import MLUtils

# Train a DecisionTree model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = DecisionTree.trainRegressor(trainingData, categoricalFeaturesInfo=featuresDic,
                                    impurity='variance', maxDepth=5, maxBins=53)

# maxbins 32 to 52

# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda (v, p): (v - p) * (v - p)).sum() / float(testData.count())
print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression tree model:')
print(model.toDebugString())

# Save and load model
model.save(sc, "myModelPath")
sameModel = DecisionTreeModel.load(sc, "myModelPath")

"""

"""
# Note: Cannot Run because error: DecisionTree requires maxBins due to categorical

########### boosted
# https://spark.apache.org/docs/latest/api/python/pyspark.mllib.html#module-pyspark.mllib.regression
# https://spark.apache.org/docs/latest/mllib-decision-tree.html
# https://spark.apache.org/docs/latest/mllib-ensembles.html
# https://github.com/apache/spark/blob/master/examples/src/main/python/mllib/gradient_boosted_trees.py

from pyspark.context import SparkContext
from pyspark.mllib.tree import GradientBoostedTrees
from pyspark.mllib.util import MLUtils

# Train a GradientBoostedTrees model.
#  Empty categoricalFeaturesInfo indicates all features are continuous.
model = GradientBoostedTrees.trainRegressor(trainingData, categoricalFeaturesInfo=featuresDic,
                                            numIterations=30, maxDepth=4)
# Evaluate model on test instances and compute test error
predictions = model.predict(testData.map(lambda x: x.features))
labelsAndPredictions = testData.map(lambda lp: lp.label).zip(predictions)
testMSE = labelsAndPredictions.map(lambda vp: (vp[0] - vp[1]) * (vp[0] - vp[1])).sum() \
    / float(testData.count())

print('Test Mean Squared Error = ' + str(testMSE))
print('Learned regression ensemble model:')
print(model.toDebugString())
"""

"""
# Note: No Need to Standardize with trees

# Standardize
label = test2.map(lambda x: x.label)
features = test2.map(lambda x: x.features)
scaler1 = StandardScaler().fit(features)
data1 = label.zip(scaler1.transform(features))
data2 = label.zip(scaler1.transform(features.map(lambda x: Vectors.dense(x.toArray()))))

# To convert a vector dense to 
#[f for f in Vectors.dense([1.0, 2.0])]
"""


