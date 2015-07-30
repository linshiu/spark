
"""
Script runs python Spark
Assumes ml.conf and python script are located from where this script is run

@param: Input FILE path in Hadoop with dataset
        Ex: inputPath: "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"

@param: Script File Name
        Ex: script: "lin_2_1.py"

To call in command line: $python lin_driver.py <inputFilePath> <script>

Saves results in current local directory with extension txt and name as script file

@author: steven
"""

import argparse
import os
import sys
parser = argparse.ArgumentParser(description='*** Driver for Spark ***')
parser.add_argument("inputPath", help="Input FILE path in Hadoop with dataset")
parser.add_argument("scriptFile", help="Script File Name")
args = parser.parse_args()

delim = "."
outputFileName = args.scriptFile.split(delim)[0] + ".txt"

# run Spark
# Slow version:
os.system("/opt/cloudera/parcels/SPARK/bin/spark-submit --total-executor-cores 2 --properties-file ml.conf "
	      "{script} {input} {output}".format(script= args.scriptFile, input=args.inputPath, output= outputFileName))

#os.system("/opt/cloudera/parcels/SPARK/bin/spark-submit --properties-file ml.conf "
#	      "{script} {input} {output}".format(script= args.scriptFile, input=args.inputPath, output= outputFileName))


print("Completed....{output} in local directory".format(output=outputFileName))




