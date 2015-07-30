"""
To Run in Command Line: $python lin_run_1_2.py
Script Uses lin_driver.py to call python scripts for the different
parts of the assignment

"""

import os
import sys

inputPath = "hdfs://wolf.iems.northwestern.edu/user/huser76/assignment5/Crimes_-_2001_to_present.csv"
#scriptFile = "lin_1.py"

scripts = ["lin_1.py", "lin_2_1.py", "lin_2_2.py", "lin_2_3.py"]

for scriptFile in scripts:
	os.system("python lin_driver.py {input} {script}".format(input=inputPath, script=scriptFile))

