"""
To Run in Command Line: $python lin_run_3.py
Script Uses lin_driver.py to call python scripts for the different
parts of the assignment

Note: inputpath for weather data in lin_3_violent.py and lin_3_nonViolent.py was hardcoded
      to inputPathW = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/ILCHICAG.txt"

"""

import os
import sys

inputPath = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"
#scriptFile = "lin_1.py"

scripts = ["lin_3_violent.py", "lin_3_nonViolent.py"]

for scriptFile in scripts:
	os.system("python lin_driver_slow.py {input} {script}".format(input=inputPath, script=scriptFile))

