# Crime in Chicago

The Chicago crime data from 2011 is publicly available online. The field name is self explanatory. Less obvious fields: block = the first 5 characters correspond to the block code and the rest specify the street location; IUCR = Illinois Uniform Crime Reporting code; X/Y coordinates = to visualize the data on a map, not needed in the assignment; District, Beat = police jurisdiction geographical partition; the region is partitioned in several districts; each district is partitioned in several beats.

Spark version: 1.3

[Crime Data Source](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-present/ijzp-q8t2)

[Wards](http://www.cityofchicago.org/city/en/depts/doit/dataset/boundaries_-_wards.html)

[Community areas](http://www.cityofchicago.org/city/en/depts/doit/dataset/boundaries_-_communityareas.html)

[Beats](http://gis.chicagopolice.org/pdfs/district_beat.pdf)

[Temperature Data Source](http://academic.udayton.edu/kissock/http/Weather/gsod95-current/ILCHICAG.txt)

**To run:**
	1. Configure ml.config file
	2. Run the corresponding python run file
	3. Results in corresponding findings files

## 1. Distribution of crime events

* Histogram of Average crime events by month using SparkSQL

	```shell
	python lin_run_1_2.py
	```

## 2. Exploratory Analysis

* Top 10 blocks in crime events in the last 3 years
* The two beats that are adjacent with the highest correlation in the number of crime events
* Determine if the number of crime events is different between Majors Daly and Emanuel

	```shell
	python lin_run_1_2.py
	```

## 3. Prediction at beat level

* Predict the number of crime events in the next week at the beat level
* External data source used

	```shell
	python lin_run_3.py
	```

## 4. Patterns with respect to time

* Find patterns of crimes with arrest with respect to time of the day, day of the week, and month. 

	```shell
	python lin_run_4.py
	```
 
 ## Visualizations

 ![Histogram by Month](/msia_big_data_spark/1_Visuals/lin_findings1.png)

 ![Histogram by Weekday](/msia_big_data_spark/1_Visuals/lin_findings4_2.png)

 ![Histogram by Hour](/msia_big_data_spark/1_Visuals/lin_findings4_1.png)

 ![Heat Map 1](/msia_big_data_spark/1_Visuals/lin_findings4_3.png)

 ![Heat Map 2](/msia_big_data_spark/1_Visuals/lin_findings4_3.png)

 ![Map Crimes](/msia_big_data_spark/1_Visuals/lin_findings2.png)

