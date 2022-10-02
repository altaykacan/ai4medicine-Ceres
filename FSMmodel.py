import glob, os, getopt, sys
import csv, json
from os import walk
import statistics
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
import datetime
import collections
from collections import deque
from collections import OrderedDict
from itertools import tee, groupby
from matplotlib import pyplot as plt
from operator import itemgetter
from datetime import date, timedelta
os.chdir(".")

# This is the submission from Team Ceres for the ai4medicine challenge in the 2022 TUM.ai makeathon.
# The main body of the code is taken from the GitHub repo: https://github.com/StanfordBioinformatics/wearable-infection - which
# unfortunately was lacking in documentation and comments. The code implements the NightSignal algorithm from that repo
# which raises red or yellow alarms by comparing resting heart rates (rhr) of users measured in a given night to the history 
# of measured rhr's at night.
#
# The decision of which alarm to raise is made based on a finite state machine developed by the authors of the 
# corresponging papers which are linked below.
#
# This script goes through each .csv file in the "data_test" directory and collects all red and yellow alerts in
# potential_reds.csv. In theory these results can be further processed to provide better predictions but our current
# implementation only takes these as the final predictions.
#
# There were two datasets available (phase 1 and 2). The NightSignal algorithm was created based on the phase 2 study,
# but here we used it for phase 1. That's because we wanted it to combine with pretrained LSTM-Autoencoder Anomaly Detection
# (LAAD) models but didn't have the time to do so. Our approach is explained in more detailed in the README of our GitHub repo.
#
# Phase-1 study: https://www.nature.com/articles/s41551-020-00640-6
# LAAD applied to our usecase: https://www.medrxiv.org/content/10.1101/2021.01.08.21249474v1
# GitHub repo for the LAAD implementation we wanted to use: https://github.com/gireeshkbogu/LAAD
# LAAD in general: https://arxiv.org/abs/1607.00148
# Phase-2 study, where the NightSignal algorithm is from: https://www.nature.com/articles/s41591-021-01593-2

#functions
def consecutive_groups(iterable, ordering=lambda x: x):
    for k, g in groupby(enumerate(iterable), key=lambda x: x[0] - ordering(x[1])):
        yield map(itemgetter(1), g)
        
def merge_two_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z
    
def sort_dict_data(data):
    return OrderedDict((k, v)
                       for k, v in sorted(data.iteritems()))
                       
def round10Base( n ):
    a = (n // 10) * 10
    b = a + 10
    return (b if n - a > b - n else a)




########## ALGORITHM SETTINGS ##########
medianConfig = "MedianOfAvgs" # MedianOfAvgs | AbsoluteMedian
yellow_threshold = 3 # the higher the threshold, the less sensitive the algorithm.
red_threshold = 6
########################################

# Clear residual files
if os.path.exists("potential_reds.csv"):
    os.remove("potential_reds.csv")
    print("Potential reds have been deleted successfully")
else:
    print("The red file does not exist!")

if os.path.exists("potential_yellows.csv"):
    os.remove("potential_yellows.csv" )
    print("Potential yellows have been deleted successfully")
else:
    print("The yellow file does not exist!")

count = 0 # last minute quick-fix #TODO do this better

# Walking a directory tree and printing the names of the directories and files
for dirpath, dirnames, files in os.walk('./data_test'):
    for file_name in files: # goes through all the test csv files 
        with open(f'./data_test/{file_name}') as hrFile: 
            records = hrFile.readlines()
            print(file_name)
            file_name = file_name.strip('.csv') #reformat to fit desired predictions
            date_hrs_dic = {}
            
            for record in records:
                if ("timestamp" not in record): # skip the header row
                    record_elements = record.split(",")
                    rec_date = record_elements[0][:10] # takes the date
                    rec_time = record_elements[0][11:] # takes the total time
                    rec_hr = record_elements[1][:6].strip('\n') # gets the heart rate and remove newline
                                                                # getting only first 6 chars
                    if ((rec_time.startswith("00:")) or (rec_time.startswith("01:")) or (rec_time.startswith("02:")) or (rec_time.startswith("03:")) or (rec_time.startswith("04:")) or (rec_time.startswith("05:")) or (rec_time.startswith("06:"))):
                        if (rec_date not in date_hrs_dic):
                            date_hrs_dic[rec_date] = rec_hr
                        else:
                            date_hrs_dic[rec_date] = date_hrs_dic[rec_date] + "*" + rec_hr 
                
            ###Calculate AVGs , Imputation, Healthy baseline Median, and Alerts
            date_hr_avgs_dic = {}
            for key in date_hrs_dic: # for each day
                AVGHR = 0
                temp = date_hrs_dic[key]
                numOfHRs = str(temp).count("*") + 1
                hrs = temp.split("*")
                for hr in hrs:
                    AVGHR = AVGHR + int(float(hr))
                AVGHR = int(AVGHR/numOfHRs)
                date_hr_avgs_dic[key] = AVGHR

            missed_days_avg_dic = {}
            sorted_keys = sorted(date_hr_avgs_dic.keys())
            sorted_avgs = sorted(date_hr_avgs_dic.items())
            for i,v in enumerate(sorted_keys):
                if(i!=0 and i!=len(sorted_keys)-1):
                    today = datetime.datetime.strptime(sorted_keys[i] , "%Y-%m-%d")
                    nextDay = datetime.datetime.strptime(sorted_keys[i+1] , "%Y-%m-%d")
                    prevDay = datetime.datetime.strptime(sorted_keys[i-1] , "%Y-%m-%d")
                    if ( (nextDay-today).days==1 and (today-prevDay).days==2):
                        missDate = today - datetime.timedelta(days=1)
                        missed_days_avg_dic[missDate.strftime("%Y-%m-%d")] = round((date_hr_avgs_dic[sorted_keys[i]] + date_hr_avgs_dic[sorted_keys[i-1]])/2 , 1)
            for key in missed_days_avg_dic:
                if key not in date_hr_avgs_dic:
                    date_hr_avgs_dic[key] = missed_days_avg_dic[key]
            temp = OrderedDict(sorted(date_hr_avgs_dic.items(), key=lambda t: t[0]))
            date_hr_avgs_dic = dict(temp)


            ### median of averages config
            if(medianConfig == "MedianOfAvgs"):
                prev_keys_dic = {}
                for k1 in date_hr_avgs_dic:
                    k1_prev_keys = []
                    for k2 in date_hr_avgs_dic:
                        if (k1>=k2):
                            k1_prev_keys.append(k2)
                    prev_keys_dic[k1] = k1_prev_keys

                date_hr_meds_dic = {}
                for k in prev_keys_dic:
                    list_for_med = []
                    prev_keys = prev_keys_dic[k]
                    for item in prev_keys:
                        list_for_med.append(date_hr_avgs_dic[item])
                    date_hr_meds_dic[k] = int(statistics.median(list_for_med))

            ### median config
            elif(medianConfig == "AbsoluteMedian"):
                live_dates_hrs_dic = {}
                for record in records:
                    if ("Device" not in record):
                        record_elements = record.split(",")
                        rec_date = record_elements[1]
                        rec_time = record_elements[2]
                        rec_hr = record_elements[3].strip(' \t\n\r')
                        if ((rec_time.startswith("00:")) or (rec_time.startswith("01:")) or (rec_time.startswith("02:")) or (rec_time.startswith("03:")) or (rec_time.startswith("04:")) or (rec_time.startswith("05:")) or (rec_time.startswith("06:"))):
                            if (rec_date not in live_dates_hrs_dic):
                                live_dates_hrs_dic[rec_date] = rec_hr
                            else:
                                live_dates_hrs_dic[rec_date] = live_dates_hrs_dic[rec_date] + "*" + rec_hr

                live_dates_hrs_dic_new = {}
                for key1 in live_dates_hrs_dic:
                    live_dates_hrs_dic_new[key1] = ""
                    temp = ""
                    for key2 in live_dates_hrs_dic:
                        if key1 >= key2:
                            if temp=="":
                                temp = live_dates_hrs_dic[key2]
                            else:
                                temp = temp + "*" + live_dates_hrs_dic[key2]
                    live_dates_hrs_dic_new[key1] = temp

                date_hr_meds_dic = {}
                for key in live_dates_hrs_dic_new:
                    MEDHR = 0
                    temp = live_dates_hrs_dic_new[key]
                    hrs = temp.split("*")
                    med_list = []
                    for hr in hrs:
                        med_list.append(int(float(hr)))
                    MEDHR = int(statistics.median(med_list))
                    date_hr_meds_dic[key] = MEDHR


            # Add column names for the first run only
            if count == 0:
                with open("potential_reds.csv" , "a") as out_file:
                    out_file.write("case , symptom_onset" "\n")
                with open("potential_yellows.csv" , "a") as out_file:
                    out_file.write("case , symptom_onset" "\n")
                count += 1 # this is ugly but got no time
            
            # Add the data to the csv files
            for key in date_hr_avgs_dic:
                if (key in date_hr_meds_dic):
                    if (date_hr_avgs_dic[key] >= date_hr_meds_dic[key] + red_threshold):
                        with open("potential_reds.csv" , "a") as out_file:
                            out_file.write(file_name + ',' + key + "\n")
                    if (date_hr_avgs_dic[key] >= date_hr_meds_dic[key] + yellow_threshold):
                        with open("potential_yellows.csv" , "a") as out_file:
                            out_file.write(file_name + ',' + key + "\n")
            
            print('Done preparing data')  #NOTE

            ###Red alerts (red states in NightSignal deterministic finite state machine)
            ###If for two consecutive nights, average RHR is above the red_threshold based on the median of average RHR overnight, then a red alert is triggered
            red_alert_dates = []
            dates_array = []
            try:
                with open("potential_reds.csv", "r") as my_file:
                    for line in my_file:
                        dates_array.append(line.strip(' \t\n\r'))
                track = []
                for i in range(len(dates_array)-1):
                    today = datetime.datetime.strptime(dates_array[i], '%Y-%m-%d')
                    next = datetime.datetime.strptime(dates_array[i+1], '%Y-%m-%d')
                    if((next - today).days == 1):
                        if(today in track):
                            red_alert_dates.append(str(next).split(' ')[0])
                            track.append(next)
                        else:
                            red_alert_dates.append(str(next).split(' ')[0])
                            track.append(today)
                            track.append(next)
            except:
                print("no red file")


            ###Yellow alerts (yellow states in NightSignal deterministic finite state machine)
            ###If for two consecutive nights, average RHR is above the yellow_threshold, but not above the red_threshold, based on the median of average RHR overnight, then a yellow alert is triggered
            yellow_alert_dates = []
            dates_array = []
            try:
                with open("potential_yellows.csv", "r") as my_file:
                    for line in my_file:
                        dates_array.append(line.strip(' \t\n\r'))
                track = []
                for i in range(len(dates_array)-1):
                    today = datetime.datetime.strptime(dates_array[i], '%Y-%m-%d')
                    next = datetime.datetime.strptime(dates_array[i+1], '%Y-%m-%d')
                    if((next - today).days == 1):
                        if(today in track):
                            if(str(next).split(' ')[0] not in red_alert_dates):
                                yellow_alert_dates.append(str(next).split(' ')[0])
                                track.append(next)
                        else:
                            if(str(next).split(' ')[0] not in red_alert_dates):
                                yellow_alert_dates.append(str(next).split(' ')[0])
                                track.append(today)
                                track.append(next)
            except:
                print("no yellow file")



            print('Done generating potential reds and yellows')  #NOTE
            # print('red alerts: ', red_alert_dates)

