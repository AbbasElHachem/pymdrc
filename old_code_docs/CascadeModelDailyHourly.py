# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas, IWS
"""

'''
This Script is for a Multiplicative Discrete Random Cascade Model (MDRC)

First read rainfall data on a fine resolution (1hour) and aggregate the data
to higher time frequencies:
(input Level: 1day; upper Level: 12hrs; middle Level: 6hrs, lower level: 3hrs
lowest level: 1.5hrs).

Second disaggregating the data through a mutiplicative Discrete Random Cascade
model (MDRC) (1440min->720min->360min->180min->90min),
at the end, rainfall on a finer resolution will be simulated
using a higher time frequency data.

This MDRC is a Microcanonical model: it conserves volumes in every level.

The first step is to find the weights (W1 and W2) of every level, this is done
by finding if the volume (V0) in the upper level (60min) has fallen
in the first sub interval (V1=V0.W1) or the second (V2=V0.W2) or in both.

Finding model parameters:
For every recorded rainfall in the upper level if volume > threshhold (0.3mm)
find W1 = R1/R and (W2 = 1-W1).
A sample of W is obtained in every level, plot histogram to find distribution.

The weights represent a probability of how the rainfall volume is distributed,
Three possible values for the weights:
W1 = 0 means all rainfall fell in 2nd sub-interval P (W=0)
W1 = 1 means all rainfall fell in 1st sub-interval P (W=1)
0 < W1 < 1 means part of rainfall fell in 1st sub iterval and part in the 2nd
For calculating P01, the relation between the volumes and the weights is
modeled through a logistic regression.
For calculating the prob P (0<W<1) a beta distribution is assigned
and using the maximum likelihood method the parameter ß is estimated
for every cascade level and every station.
The MDRC baseline model has two parameters P01 and ß per level.

the MDRC unbounded model is introduced and allows relating the probability
P01 to the rainfall volume R through a logistic regression function
the parameters of the logisticRegression fct: a an b are estimated
using the maximum likelihood method. This is done by first identifying where
w is 0 or 1 and for these values, find the corresponding rainfall volume R
and use log(logisticRegression fct) and where w is between ]0:1[ use
log(1-logisticRegression fct), in that way the parameters a and b are estimated
using all of the observed weights. This is done for every station and every
cascade level, therefore the unbounded model has three parameters per level
a, b and beta. the value of beta is the same used in the baseline model


Analysing:
Once parameters are found, study the effect of the time and space on them:
First divide the events into 4 different boxes:
(Isolated: 010, Enclosed: 111, Followed: 011, Preceded: 110  ), plot them
Second extract the P01 for every month and plot it
'''

from scipy import stats
from scipy import optimize
from collections import Counter

import os
import timeit
import time
import math

import pandas as pd
pd.set_option('display.max_rows', 1000)

import numpy as np
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt
plt.ioff()

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer() # to get the runtime of the program

main_dir = (r'X:\hiwi\El Hachem\CascadeModel_Daily_Hourly')
os.chdir(main_dir)

out_dir = os.path.join(main_dir, r'Weights')
if not os.path.exists(out_dir): os.mkdir(out_dir)

#define rainfall data directory
data_dir = r'X:\hiwi\El Hachem\Peru_Project'

#read 5min data file, original rainfall values
in_60m_df_file = os.path.join(data_dir, 'Rain_mm_Tx_Tot_T60 (mm, Tot).csv')

assert os.path.exists(in_60m_df_file),\
        'wrong 5min df file location'
#==============================================================================
#
#==============================================================================

#parameters for reading df
df_sep = ';'
date_format = '%Y-%m-%d %H:%M:%S'

#define start & end date (common dates all dfs_L1)
start_date = '2012-09-06 00:00:00'
end_date = '2017-01-01 00:00:00'

#read in_df_5m and convert index to datetime object
in_df_60m = pd.read_csv(in_60m_df_file,
                       sep=df_sep,
                       index_col=0,
                       encoding='utf-8')

in_df_60m.index = pd.to_datetime(in_df_60m.index,
                                format=date_format)

#extract date range to work with
in_df_60m = in_df_60m.loc[start_date:end_date, :]

in_df_60m.drop(in_df_60m.columns[[0, 3, 10, 13]], axis=1,
              inplace=True) #drop stations with bad data

#list of month nbrs to find Weights per month, later used
months = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

#==============================================================================
# Cascade Model (1440min-->720min-->360min-->180min-->90min) (5 Levels)
#==============================================================================
'''
The rainfall volume R (mm) in each disagregation level is subdivided into two
sub-interval. Calculating the ratios between R and R1 and R2 in sub-interval 1
and sub-interval 2, allows sampling the weights W1 = R1/R and W2= R2/R.
For each rainfall volume R above a certain threshold the weights are calculated.

This MDRC is a Microcanonical MDRC, which means that in each disagregation level
the rainfall volume is preserved, so that R1.W1 + R2.W2 = R --> W1 + W2 = 1.

In this script weights for both sub-intervals are calculated and the results is
asserted using the statement that R1.W1 + R2.W2 = R; this is verified for all R
above the threshold and for each cascade level and station

The first step is to find the weights W1 = 1-W2 and then to fit a Beta pdf to W
and to calculate from the weights the probabiblity P01.
'''

# define volume threshhold R (mm), R > 0.3 only considered
threshhold = 0.3 #mm what to change

min_nbr_vals = 10. # min nbr of weights for Prob cals

#==============================================================================
#LEVEL ONE (60min-->30min)
#==============================================================================
#define name of cascade level, will be used when saving and reading results
cascade_level_1 = 'Level_one'

#define time of original level, upper level
freq_level0 = '1440T' #60min

#define time of sub intervals, level one
freq_level1 = '720T' #30min

#aggregate 60min df to 1440min df, input data
df_60_to_1440m = in_df_60m.resample(freq_level0,
                                    label='right',
                                    closed='right').sum()

#aggeragete 5min data to 30min data using freq_level1
df_60_to_720m = in_df_60m.resample(freq_level1,
                                 label='right',
                                 closed='right').sum()

#verify that volumes are conserved after resampling
assert np.isclose(df_60_to_720m.sum(),
                  df_60_to_1440m.sum(),
                  in_df_60m.sum()).any() == True,\
                  'error in resampling'

#save resampled df of 30min timestamp, used in model evaluation
df_60_to_720m.to_csv('resampled 720min.csv', sep=df_sep)

#define time delta to substract from df index to get values in other df
time_delta_720m_left = 86400. #s 1 hour
time_delta_720m_right = time_delta_720m_left/2 #1800s 30min
#1 day = 86 400 s
#12hrs = 43 200 s
#6hrs = 21 600 s
#3hrs = 10 800 s
#1.5hrs = 5 400 s

#==============================================================================
# find Percentage of nglected data when using threshold
#==============================================================================
def neglectedData(df_level_1, R_threshhold):

    ''' input: ppt input df and threhhold
        output: count df vals b4 and after use of threshhold
    '''
    #create dicts to hold output
    stn_values_count_b4 = {k:[] for k in df_level_1.columns}
    stn_values_count_after = {k:[] for k in df_level_1.columns}
    stn_values_count_ratio = {k:[] for k in df_level_1.columns}

    #initial values count per station
    for station in df_level_1.columns:

        #drop nans from station data and count data
        df_level_1[station].dropna(axis=0, inplace=True)
        stn_values_count_b4[station] = df_level_1[station].count()

    #select values above threhold
    df_level_1 = df_level_1[df_level_1 >= R_threshhold]

    #count remaining values above threshold per station
    for station in df_level_1.columns:
        stn_values_count_after[station] = df_level_1[station].count()

    #find ratio of lost values in %
    for station in df_level_1.columns:
        stn_values_count_ratio[station] =\
        100 * (stn_values_count_after[station]\
        / stn_values_count_b4[station])

    #save dict to data frame and save results
    df_output = pd.DataFrame.from_dict(stn_values_count_ratio,
                                           orient='index')
    #adjust columns name
    df_output.rename(columns={0:'df remaining values %'}, inplace=True)

    for station in stn_values_count_after.keys():
        df_output.loc[station, 'nbr of used values'] =\
        stn_values_count_after[station]

    #extract time stamp of df, used when saving df
    df_time_frequency = df_level_1.index.freqstr

    #save df
    df_output.to_csv(os.path.join(out_dir,
                      'df %s remaning values.csv' %df_time_frequency),
                      sep=df_sep,)
    return df_output

#call fct to know % of available values above threhhold
neglectedData(df_60_to_1440m, threshhold)
print('done calculating remaning data after use of threshold')

#calculate weights W1 and W2, cascade level 1
def cascadeWeightsLevel_1(df_level_1, df_level_2, R_threshhold):

    '''
    input: rainfall data level 1, level2, threshold
    output: weights left and right per station
            weights per month
            original values in upper df per weights
    '''

    #define dict to hold weights left W1, right W2 and corresponding rainfall
    #defined as globals because used later, (maybe change it later)
    global weights_left_L1, weights_right_L1, dict_vals_level_one

    weights_left_L1 = {k:[] for k in df_level_1.columns}
    weights_right_L1 = {k:[] for k in df_level_1.columns}
    dict_vals_level_one = {k:[] for k in df_level_1.columns}

    #define dict to hold weights per station per month (dict within dict)
    global  weight_month_left_L1, weight_month_right_L1

    weight_month_left_L1 =\
            {k : {stn : [] for stn in df_level_1.columns} for k in months}
    weight_month_right_L1 =\
        {k : {stn : [] for stn in df_level_1.columns} for k in months}

    #select values above threshold from upper df
    df_level_1 = df_level_1[df_level_1 > R_threshhold]

    #start going through stations in upper df
    for station in df_level_1.columns:

        #drop nan from station values level 1
        df_level_1[station].dropna(axis=0, inplace=True)

        #make sure no more nans in data frame
        assert df_level_1[station].isnull().sum() == 0,\
                'still nans in DF level 1'

        #go through station index df level 1 and start calculating Ws
        for idx in df_level_1[station].index:

            #extract volume R in level 1, original rainfall
            val_level_1 = df_level_1[station].loc[idx]

            #append R to dict_values_level_1
            dict_vals_level_one[station].append(val_level_1)

            '''values LEFT in sub level 1'''

            #end idx of sub int 1 (12:00:00)
            idx1_1 = idx - pd.\
                Timedelta(seconds=time_delta_720m_right)

            #extract value R1 in df2, volume sub-int 1
            volume_left_l1 = df_level_2[station].loc[idx1_1]

            #check if value is nan, if so replace it, maybe nan in second df
            if math.isnan(volume_left_l1): volume_left_l1 = 1e-36

            '''values RIGHT in sub level 1'''

            #end idx sub-int 2 (00:00:00)
            idx2_2 = idx

            #extract value R2 in df2, volume sub-int 2
            volume_right_l1 = df_level_2[station].loc[idx2_2]

            #check if value is nan, if so replace it
            if math.isnan(volume_right_l1): volume_right_l1 = 1e-36

            #assert that both sub-interval volumes sum to upper interval
            assert np.abs((volume_left_l1+volume_right_l1)\
                          - val_level_1) < 1e-5

            ''' calculate weights W1 and W2 '''

            #calculate ratio volumes left W1 = R1/R
            rainfall_volume_ratio_left =\
                volume_left_l1 / val_level_1

            #calculate ratio volumes right W2 = R2/R
            rainfall_volume_ratio_right =\
                volume_right_l1 / val_level_1

            #append weights to dict weights_left
            weights_left_L1[station].\
                append(rainfall_volume_ratio_left)

            #append weights to dict weights_right
            weights_right_L1[station].\
                append(rainfall_volume_ratio_right)

            #append weights to month_dict_left
            weight_month_left_L1[idx1_1.month][station].\
                append(rainfall_volume_ratio_left)

            #append weights to month_dict_right
            weight_month_right_L1[idx2_2.month][station].\
                append(rainfall_volume_ratio_right)

    return (weights_left_L1, weights_right_L1,
            dict_vals_level_one, weight_month_left_L1,
            weight_month_right_L1)

#call fct to find weights level 1
cascadeWeightsLevel_1(df_60_to_1440m, df_60_to_720m, threshhold)
print('done with LEVEL1 weights, calculating weights L2')

#==============================================================================
# LEVEL 2 (12hrs-->6hrs)
#==============================================================================

#define name of cascade level
cascade_level_2 = 'Level_two'

'''
this level refers to disaggrgraion each of the
sub intervals from level 1 to two sub-intervals
so disaggregation depends on which sub-int it is
and the result is 4 sub-intervals in level 2

4 sub interval if (0-12) then (0-6) (6-12)
               if (12-0) then (12-18) (18-24)

note: all weights W1 are added together
      all weights W2 are added together

      W1: weights from sub-int 1 and 3
      W2: weights from sub-int 2 and 4

'''

#define time of sub intervals (360min) level two
freq_level2 = '360T'

#define time delta to substract from df index
#to get values in other df
time_delta_360m_left = time_delta_720m_left/2 #s 30min
time_delta_360m_right = time_delta_360m_left/2 #s 15min

#aggeragete 5min data to 15min data using freq_level2
df_60m_to_360m = in_df_60m.resample(freq_level2,
                                 label='right',
                                 closed='right').sum()

#verify that volumes are conserved after resampling
assert np.isclose(df_60m_to_360m.sum(),
                  in_df_60m.sum()).any() == True,\
                  'error in resampling'

#save df 15min, used in evaluation later
df_60m_to_360m.to_csv('resampled 360min.csv', sep=df_sep)

def cascadeWeightslevel_2(df_level_2, df_level_3, R_threshhold):

    '''
    input: df_level_2 and df_level_3 and R_threshold

    output: weights per stn dict: sub int one left and right (2 dict)
            weights per month dict: sub int one left and right (2 dict)

    '''
    #define dict to hold weights left and right and orig R vals
    global weights_left_L2, weights_right_L2,\
             dict_vals_level_two

    weights_left_L2 = {k:[] for k in df_level_2.columns}
    weights_right_L2 = {k:[] for k in df_level_2.columns}
    dict_vals_level_two = {k:[] for k in df_level_2.columns}


    #define dict to hold weights per month
    global weight_month_left_L2, weight_month_right_L2

    weight_month_left_L2 =\
            {k : {stn : [] for stn in df_level_2.columns} for k in months}
    weight_month_right_L2 =\
        {k : {stn : [] for stn in df_level_2.columns} for k in months}

    #select values above threshold from upper df
    df_level_2 = df_level_2[df_level_2 > R_threshhold]

    for station in df_level_2.columns:

        #drop nan from station values of upper level df
        df_level_2[station].dropna(axis=0, inplace=True)

        #make sure no more nans in df
        assert df_level_2[station].isnull().sum() == 0,\
            print('still nans in DF level 2')

        #go through station index df upper level
        for idx in df_level_2[station].index:

            #check which time it is, affects disaggregation
            if idx.hour == 12:

                '''disaggregate to sub-int (0-6; 6-12)'''

                #extract value in upper level df
                val_level_2_1 = df_level_2[station].loc[idx]

                #add original values to dict
                dict_vals_level_two[station].append(val_level_2_1)

                '''values LEFT in level 2 sub_interval 1'''

                #end idx (06:00:00) (sum sub-interval left 1)
                idx2_1_0 = idx - pd.\
                    Timedelta(seconds=time_delta_360m_right)
                assert idx2_1_0.hour == 6, 'error locating 1st idx'

                #extract value sub interval left one in df3
                volume_left_l2_0 = df_level_3[station].loc[idx2_1_0]

                #check if value is nan, if so replace it
                if math.isnan(volume_left_l2_0): volume_left_l2_0 = 1e-36

                '''values RIGHT in level 2 sub_interval 1'''

                #end idx (12:00:00), (sum sub-interval right 1)
                idx2_2_1 = idx
                assert idx2_2_1.hour == 12, 'error locating 2nd idx'

                #extract value sub interval right one in df3
                volume_right_l2_1 = df_level_3.loc[idx2_2_1, station]

                #check if value is nan, if so replace it
                if math.isnan(volume_right_l2_1): volume_right_l2_1 = 1e-36

                #assert that both sub-interval volumes sum to upper interval
                assert np.abs((volume_right_l2_1+volume_left_l2_0)\
                              - val_level_2_1) < 1e-5

                ''' calculate weights W1 and W2 sub int 1 and 2 '''

                #calculate ratio volumes left, W1=R1/R
                rainfall_volume_ratio_left_0 =\
                    volume_left_l2_0 / val_level_2_1

                #calculate ratio volumes right, W2=R2/R
                rainfall_volume_ratio_right_0 =\
                    volume_right_l2_1/val_level_2_1

                #append weights to dict left
                weights_left_L2[station].\
                    append(rainfall_volume_ratio_left_0)

                #append weights to dict right
                weights_right_L2[station].\
                    append(rainfall_volume_ratio_right_0)

                #append weights to month_dict_left
                weight_month_left_L2[idx2_1_0.month][station].\
                    append(rainfall_volume_ratio_left_0)

               #append weights to month_dict_right
                weight_month_right_L2[idx2_2_1.month][station].\
                    append(rainfall_volume_ratio_right_0)

            elif idx.hour == 0:

                '''disaggregate to sub int (12-18, 18-24)'''

                #extract value level 2_1(volume in upper df, second int)
                val_level_2_2 = df_level_2[station].loc[idx]

                #add original values to dict
                dict_vals_level_two[station].append(val_level_2_2)

                '''values LEFT in level 2 sub_intervals 2'''

                # idx (18:00:00),
                idx2_2_1 = idx - pd.\
                    Timedelta(seconds=time_delta_360m_right)
                assert idx2_2_1.hour == 18. , 'error locating 3rd idx'

                #extract values of sub interval 2 left
                volume_left_l2_2_0 = df_level_3[station].loc[idx2_2_1]

                #check if value is nan, if so replace it
                if math.isnan(volume_left_l2_2_0): volume_left_l2_2_0 = 1e-36

                '''values RIGHT in level 2 sub_interval 2'''

                #idx (00:00:00)
                idx2_2_2 = idx
                assert idx2_2_2.hour == 0.,'error locating 4th idx'

                #extract value sub interval 2 right
                volume_right_l2_2_1 = df_level_3[station].loc[idx2_2_2]

                #check if value is nan, if so replace it
                if math.isnan(volume_right_l2_2_1): volume_right_l2_2_1 = 1e-36

                #assert that both sub-interval volumes sum to upper interval
                assert np.abs((volume_right_l2_2_1+volume_left_l2_2_0)\
                              - val_level_2_2) < 0.01

                '''calculate Volumes ratio and append to dict weights'''

                #calculate ratio volume left, W1=R1/R
                rainfall_volume_ratio_left_1 =\
                    volume_left_l2_2_0 / val_level_2_2

                #calculate ratio volume right, w2=R2/R
                rainfall_volume_ratio_right_1 =\
                    volume_right_l2_2_1 / val_level_2_2

                #append weight to dict left
                weights_left_L2[station].\
                    append(rainfall_volume_ratio_left_1)

                #append weight to dict right
                weights_right_L2[station].\
                    append(rainfall_volume_ratio_right_1)

                #append weights to month_dict_left
                weight_month_left_L2[idx2_2_1.month][station].\
                    append(rainfall_volume_ratio_left_1)

               #append weights to month_dict_right
                weight_month_right_L2[idx2_2_2.month][station].\
                    append(rainfall_volume_ratio_right_1)

    return (weights_left_L2, weights_right_L2,
            weight_month_left_L2, weight_month_right_L2,
            dict_vals_level_two)

#call fct with df level 2 and 3
cascadeWeightslevel_2(df_60_to_720m, df_60m_to_360m, threshhold)
print('done with LEVEL2 weights, writing weights L1 and L2 to dfs')

#==============================================================================
# LEVEL 2 (6hrs-->3hrs)
#==============================================================================

#define name of cascade level
cascade_level_3 = 'Level_three'

'''
this level refers to disaggrgraion each of the
sub intervals from level 2 to two sub-intervals
so disaggregation depends on which sub-int it is
and the result is 8 sub-intervals in level 3

4 sub interval if (0-6) then (0-3) (3-6)
               if (6-12) then (6-9) (9-12)
               if (12-18) then (12-15) (15-18)
               if (18-24) then (18-21) (21-24)
note: all weights W1 are added together
      all weights W2 are added together

      W1: weights from sub-int 1 and 3 and 5 and 7
      W2: weights from sub-int 2 and 4 and 6 and 8

'''

#define time of sub intervals (360min) level two
freq_level3 = '180T' #6hours

#define time delta to substract from df index
#to get values in other df
time_delta_180m_left = time_delta_360m_left/2 #s 6hours
time_delta_180m_right = time_delta_180m_left/2 #s 3hours

#aggeragete 5min data to 15min data using freq_level2
df_60m_to_180m = in_df_60m.resample(freq_level3,
                                    label='right',
                                    closed='right').sum()

#verify that volumes are conserved after resampling
assert np.isclose(df_60m_to_180m.sum(),
                  in_df_60m.sum()).any() == True,\
                  'error in resampling'

#save df 15min, used in evaluation later
df_60m_to_180m.to_csv('resampled 180min.csv', sep=df_sep)

def cascadeWeightslevel_3(df_level_3, df_level_4, R_threshhold):

    '''
    input: df_level_3 and df_level_4 and R_threshold

    output: weights per stn dict: sub int one left and right (2 dict)
            weights per month dict: sub int one left and right (2 dict)

    '''
    #define dict to hold weights left and right and orig R vals
    global weights_left_L3, weights_right_L3,\
             dict_vals_level_three

    weights_left_L3 = {k:[] for k in df_level_3.columns}
    weights_right_L3 = {k:[] for k in df_level_3.columns}
    dict_vals_level_three = {k:[] for k in df_level_3.columns}


    #define dict to hold weights per month
    global weight_month_left_L3, weight_month_right_L3

    weight_month_left_L3 =\
            {k : {stn : [] for stn in df_level_3.columns} for k in months}
    weight_month_right_L3 =\
        {k : {stn : [] for stn in df_level_3.columns} for k in months}

    #select values above threshold from upper df
    df_level_3 = df_level_3[df_level_3 > R_threshhold]

    for station in df_level_3.columns:

        #drop nan from station values of upper level df
        df_level_3[station].dropna(axis=0, inplace=True)

        #make sure no more nans in df
        assert df_level_3[station].isnull().sum() == 0,\
            print('still nans in DF level 2')

        #go through station index df upper level
        for idx in df_level_3[station].index:

            #check which time it is, affects disaggregation
            if idx.hour == 6:

                '''disaggregate to sub-int (0-3; 3-6)'''

                #extract value in upper level df
                val_level_3_1 = df_level_3[station].loc[idx]

                #add original values to dict
                dict_vals_level_three[station].append(val_level_3_1)

                '''values LEFT in level 2 sub_interval 1'''

                #end idx (03:00:00) (sum sub-interval left 1)
                idx3_1_0 = idx - pd.\
                    Timedelta(seconds=time_delta_180m_right)
                assert idx3_1_0.hour == 3, 'error locating 1st idx'

                #extract value sub interval left one in df3
                volume_left_l3_0 = df_level_4[station].loc[idx3_1_0]

                #check if value is nan, if so replace it
                if math.isnan(volume_left_l3_0): volume_left_l3_0 = 1e-36

                '''values RIGHT in level 2 sub_interval 1'''

                #end idx (6:00:00), (sum sub-interval right 1)
                idx3_2_1 = idx
                assert idx3_2_1.hour == 6, 'error locating 2nd idx'

                #extract value sub interval right one in df3
                volume_right_l3_1 = df_level_4.loc[idx3_2_1, station]

                #check if value is nan, if so replace it
                if math.isnan(volume_right_l3_1): volume_right_l3_1 = 1e-36

                #assert that both sub-interval volumes sum to upper interval
                assert np.abs((volume_right_l3_1+volume_left_l3_0)\
                              - val_level_3_1) < 1e-5

                ''' calculate weights W1 and W2 sub int 1 and 2 '''

                #calculate ratio volumes left, W1=R1/R
                rainfall_volume_ratio_left_3_0 =\
                    volume_left_l3_0 / val_level_3_1

                #calculate ratio volumes right, W2=R2/R
                rainfall_volume_ratio_right_3_0 =\
                    volume_right_l3_1/val_level_3_1

                #append weights to dict left
                weights_left_L3[station].\
                    append(rainfall_volume_ratio_left_3_0)

                #append weights to dict right
                weights_right_L3[station].\
                    append(rainfall_volume_ratio_right_3_0)

                #append weights to month_dict_left
                weight_month_left_L3[idx3_1_0.month][station].\
                    append(rainfall_volume_ratio_left_3_0)

               #append weights to month_dict_right
                weight_month_right_L3[idx3_2_1.month][station].\
                    append(rainfall_volume_ratio_right_3_0)

            elif idx.hour == 12:

                '''disaggregate to sub int (6-9, 9-12)'''

                #extract value level 2_1(volume in upper df, second int)
                val_level_3_2 = df_level_3[station].loc[idx]

                #add original values to dict
                dict_vals_level_three[station].append(val_level_3_2)

                '''values LEFT in level 2 sub_intervals 2'''

                # idx (12:00:00),
                idx3_2_1 = idx - pd.\
                    Timedelta(seconds=time_delta_180m_right)
                assert idx3_2_1.hour == 9. , 'error locating 3rd idx'

                #extract values of sub interval 2 left
                volume_left_l3_2_0 = df_level_4[station].loc[idx3_2_1]

                #check if value is nan, if so replace it
                if math.isnan(volume_left_l3_2_0): volume_left_l3_2_0 = 1e-36

                '''values RIGHT in level 2 sub_interval 2'''

                #idx (00:00:00)
                idx3_2_2 = idx
                assert idx3_2_2.hour == 12.,'error locating 4th idx'

                #extract value sub interval 2 right
                volume_right_l3_2_1 = df_level_4[station].loc[idx3_2_2]

                #check if value is nan, if so replace it
                if math.isnan(volume_right_l3_2_1): volume_right_l3_2_1 = 1e-36

                #assert that both sub-interval volumes sum to upper interval
                assert np.abs((volume_right_l3_2_1+volume_left_l3_2_0)\
                              - val_level_3_2) < 0.01

                '''calculate Volumes ratio and append to dict weights'''

                #calculate ratio volume left, W1=R1/R
                rainfall_volume_ratio_left_1 =\
                    volume_left_l3_2_0 / val_level_3_2

                #calculate ratio volume right, w2=R2/R
                rainfall_volume_ratio_right_1 =\
                    volume_right_l3_2_1 / val_level_3_2


                #append weight to dict left
                weights_left_L3[station].\
                    append(rainfall_volume_ratio_left_1)

                #append weight to dict right
                weights_right_L3[station].\
                    append(rainfall_volume_ratio_right_1)

                #append weights to month_dict_left
                weight_month_left_L3[idx3_2_1.month][station].\
                    append(rainfall_volume_ratio_left_1)

               #append weights to month_dict_right
                weight_month_right_L3[idx3_2_2.month][station].\
                    append(rainfall_volume_ratio_right_1)

            elif idx.hour == 18:

                '''disaggregate to sub int (12-15, 15-18)'''

                #extract value level 2_1(volume in upper df, second int)
                val_level_3_3 = df_level_3[station].loc[idx]

                #add original values to dict
                dict_vals_level_three[station].append(val_level_3_3)

                '''values LEFT in level 2 sub_intervals 2'''

                # idx (15:00:00),
                idx3_2_1 = idx - pd.\
                    Timedelta(seconds=time_delta_180m_right)
                assert idx3_2_1.hour == 15. , 'error locating 3rd idx'

                #extract values of sub interval 2 left
                volume_left_l3_3_0 = df_level_4[station].loc[idx3_2_1]

                #check if value is nan, if so replace it
                if math.isnan(volume_left_l3_3_0): volume_left_l3_3_0 = 1e-36

                '''values RIGHT in level 2 sub_interval 2'''

                #idx (00:00:00)
                idx3_2_2 = idx
                assert idx3_2_2.hour == 18.,'error locating 4th idx'

                #extract value sub interval 2 right
                volume_right_l3_3_1 = df_level_4[station].loc[idx3_2_2]

                #check if value is nan, if so replace it
                if math.isnan(volume_right_l3_3_1): volume_right_l3_3_1 = 1e-36

                #assert that both sub-interval volumes sum to upper interval
                assert np.abs((volume_right_l3_3_1+volume_left_l3_3_0)\
                              - val_level_3_3) < 0.01

                '''calculate Volumes ratio and append to dict weights'''

                #calculate ratio volume left, W1=R1/R
                rainfall_volume_ratio_left_3 =\
                    volume_left_l3_3_0 / val_level_3_3

                #calculate ratio volume right, w2=R2/R
                rainfall_volume_ratio_right_3 =\
                    volume_right_l3_3_1 / val_level_3_3

                #append weight to dict left
                weights_left_L3[station].\
                    append(rainfall_volume_ratio_left_3)

                #append weight to dict right
                weights_right_L3[station].\
                    append(rainfall_volume_ratio_right_3)

                #append weights to month_dict_left
                weight_month_left_L3[idx3_2_1.month][station].\
                    append(rainfall_volume_ratio_left_3)

               #append weights to month_dict_right
                weight_month_right_L3[idx3_2_2.month][station].\
                    append(rainfall_volume_ratio_right_3)

            elif idx.hour == 0:

                '''disaggregate to sub int (18-21, 21-24)'''

                #extract value level 2_1(volume in upper df, second int)
                val_level_3_4 = df_level_3[station].loc[idx]

                #add original values to dict
                dict_vals_level_three[station].append(val_level_3_4)

                '''values LEFT in level 2 sub_intervals 2'''

                # idx (21:00:00),
                idx3_2_1 = idx - pd.\
                    Timedelta(seconds=time_delta_180m_right)
                assert idx3_2_1.hour == 21. , 'error locating 3rd idx'

                #extract values of sub interval 2 left
                volume_left_l3_4_0 = df_level_4[station].loc[idx3_2_1]

                #check if value is nan, if so replace it
                if math.isnan(volume_left_l3_4_0): volume_left_l3_4_0 = 1e-36

                '''values RIGHT in level 2 sub_interval 2'''

                #idx (00:00:00)
                idx3_2_2 = idx
                assert idx3_2_2.hour == 0.,'error locating 4th idx'

                #extract value sub interval 2 right
                volume_right_l3_4_1 = df_level_4[station].loc[idx3_2_2]

                #check if value is nan, if so replace it
                if math.isnan(volume_right_l3_4_1): volume_right_l3_4_1 = 1e-36

                #assert that both sub-interval volumes sum to upper interval
                assert np.abs((volume_right_l3_4_1+volume_left_l3_4_0)\
                              - val_level_3_4) < 0.01

                '''calculate Volumes ratio and append to dict weights'''

                #calculate ratio volume left, W1=R1/R
                rainfall_volume_ratio_left_4 =\
                    volume_left_l3_4_0 / val_level_3_4

                #calculate ratio volume right, w2=R2/R
                rainfall_volume_ratio_right_4 =\
                    volume_right_l3_4_1 / val_level_3_4

                #append weight to dict left
                weights_left_L3[station].\
                    append(rainfall_volume_ratio_left_4)

                #append weight to dict right
                weights_right_L3[station].\
                    append(rainfall_volume_ratio_right_4)

                #append weights to month_dict_left
                weight_month_left_L3[idx3_2_1.month][station].\
                    append(rainfall_volume_ratio_left_4)

               #append weights to month_dict_right
                weight_month_right_L3[idx3_2_2.month][station].\
                    append(rainfall_volume_ratio_right_4)

    return (weights_left_L3, weights_right_L3,
            weight_month_left_L3, weight_month_right_L3,
            dict_vals_level_three)

#call fct with df level 2 and 3
cascadeWeightslevel_3(df_60m_to_360m, df_60m_to_180m, threshhold)
print('done with LEVEL3 weights, writing weights L1, L2 and L3 to dfs')

#==============================================================================
# save results to dfs
#==============================================================================

#save weights as df for every level and sub_int
def saveWeights_DF(weights_left_dict, weights_right_dict,
                   orig_vals_dict, cascade_level):

    out_dir_level_two = os.path.join(out_dir, (r'%s' % cascade_level))
    if not os.path.exists(out_dir_level_two): os.mkdir(out_dir_level_two)

    df = {k:[] for k in weights_left_dict.keys()}

    for stn in weights_left_dict.keys():

        #create new df for each stn and each sub intervals section
        df1 = pd.DataFrame()

        #each sub interval weights
        df1[stn + ' Sub Int Left'] = weights_left_dict[stn]
        df1[stn + ' Sub Int Right'] = weights_right_dict[stn]
        df1[stn + ' Original Volume'] = orig_vals_dict[stn]

        df1.to_csv(os.path.join(out_dir_level_two,
                                'weights %s df %s.csv'\
                                % (cascade_level, stn)),
                    sep=df_sep,
                    index_label=stn)
        df[stn].append(df1)
    return df

#call fct to save weights level 1 and level 2
dfs_L1 = saveWeights_DF(weights_left_L1, weights_right_L1,
                        dict_vals_level_one, cascade_level_1)

dfs_L2 = saveWeights_DF(weights_left_L2, weights_right_L2,
                        dict_vals_level_two, cascade_level_2)

dfs_L3 = saveWeights_DF(weights_left_L3, weights_right_L3,
                        dict_vals_level_three, cascade_level_3)

print('done with saving weights , calculating prob that W=0 or W=1')
#==============================================================================
# find Probability P01 that W=0 or W=1
#==============================================================================

def probP01(weights_dict, dict_orig_vals, min_nbr_of_W, cascade_level):

    '''
    Idea: find probability W=0 or W=1 from sampled Ws for each
          station and cascade level.
          MDRC parameter one --> P01

    input:  dict: stn: [weights]
            dict: stn: orig values
            int: min nbr of Ws to consider

    ouput:  dict: stn: [P0]
            dict: stn:[P1]
    '''

    #create dict w = 0 and w = 1 and R volumes(maybe 4 later)
    dict_W0 = {k:[] for k in weights_dict.keys()}
    dict_W1 = {k:[] for k in weights_dict.keys()}

    dict_R_W0 = {k:[] for k in weights_dict.keys()}
    dict_R_W1 = {k:[] for k in weights_dict.keys()}

    #create df to hold results
    global df_probs

    #create dict P(w=0) or P(w=1)
    dict_P0 = {k:[] for k in weights_dict.keys()}
    dict_P1 = {k:[] for k in weights_dict.keys()}

    #iterate through weights per station
    for stn in weights_dict.keys():

        #check if enough weights per station
        if len(weights_dict[stn]) > min_nbr_of_W:

            for val in weights_dict[stn]:
                #W0
                if val == 0.:
                    dict_W0[stn].append(1)
                    dict_R_W0[stn].append(dict_orig_vals[stn])
                #W1
                elif val == 1.:
                    dict_W1[stn].append(1)
                    dict_R_W1[stn].append(dict_orig_vals[stn])
        #P0
        if len(dict_W0[stn]) > min_nbr_of_W:
            dict_P0[stn] = len(dict_W0[stn])/len(weights_dict[stn])
        elif len(dict_W0[stn]) <= min_nbr_of_W:
            dict_P0[stn] = -999 #if not enough vals
        #P1
        if len(dict_W1[stn]) >= min_nbr_of_W:
            dict_P1[stn] = len(dict_W1[stn])/len(weights_dict[stn])
        elif len(dict_W1[stn]) <= min_nbr_of_W:
            dict_P1[stn] = -999 #if not enough vals

    #df to hold output; stations as idx
    df_probs = pd.DataFrame(index=dict_P0.keys())

    #go through each stn values
    for stn in dict_P0.keys(): #save to df

        #write values to df
        df_probs.loc[stn, 'P0'] = dict_P0[stn]
        df_probs.loc[stn, 'P1'] = dict_P1[stn]

    #P01 = P0 + P1, what is of our interest
    for stn in df_probs.index:

        df_probs.loc[stn, 'P01'] = df_probs.loc[stn, 'P0']\
                                    + df_probs.loc[stn, 'P1']

    df_probs.to_csv(os.path.join(out_dir, r'Prob W P01 %s.csv'\
                                 % cascade_level),
                    sep=df_sep)

    return df_probs

#call fct to get P0, P1, P01 for each level
df_prob_p01_L1 = probP01(weights_left_L1, dict_vals_level_one,
                         min_nbr_vals, cascade_level_1)

df_prob_p01_L2 = probP01(weights_left_L2, dict_vals_level_two,
                         min_nbr_vals, cascade_level_2)

df_prob_p01_L3 = probP01(weights_left_L3, dict_vals_level_three,
                         min_nbr_vals, cascade_level_3)

print('done with calculating P01 per stn, calculating P01 per month')

#==============================================================================
# find prob P(w=0) or P(w=1) for every month
#==============================================================================
def probWeightsMonth(weight_dict_month, min_nbr_values, cascade_level):

    '''
    Idea: find seasonal effect on P01, used in analysing

    input:  dict with keys as months
            values are dict, station as keys, and W as values
    output: P0 and P1 per month
    '''

    #make dict a df, easier to work with
    d = pd.DataFrame.from_dict(weight_dict_month)

    #define dict to hold weights per month
    weights_month_P1 = {k:[] for k in months}
    weights_month_P0 = {k:[] for k in months}

    #define dict to hold prob per month
    global prob_month_P1
    prob_month_P1 = {k:[] for k in months}
    prob_month_P0 = {k:[] for k in months}

    #iterate for every month
    for month in d.columns:

        ct = 0. #count vals/month

        #go for every stn in df month
        for stn in d[month].index:

            # count vals per month
            ct += len(d[month][stn])

            #go through stn W vals
            for val in d.loc[stn, month]:

                #get P(W=1) per month
                if val == 1.:
                    weights_month_P1[month].append(val)

                #get P(W=0) per month
                if val == 0.:
                    weights_month_P0[month].append(val)

        #frequency P01 per month
        if len((weights_month_P1[month])) >= min_nbr_values:
            prob_month_P1[month].append(len(weights_month_P1[month]) / ct)

        if len((weights_month_P0[month])) >= min_nbr_values:
            prob_month_P0[month].append(len(weights_month_P0[month]) / ct)

        #if no P01 value then delete it
        if len((prob_month_P1[month])) < 1: del prob_month_P1[month]
        if len((prob_month_P0[month])) < 1: del prob_month_P0[month]

    #df to hold ouput
    global df_probs_month
    df_probs_month = pd.DataFrame(index=months)

    #go through values every month and append to df
    for month in df_probs_month.index:

        if month in prob_month_P0.keys():
            df_probs_month.loc[month, 'P0 per Month'] =\
                    prob_month_P0[month][0]

        if month in prob_month_P1.keys():
            df_probs_month.loc[month, 'P1 per Month'] =\
                    prob_month_P1[month][0]

    #calculate P01 per station and save to df
    df_probs_month['P01 per month'] = df_probs_month['P0 per Month']\
                                        + df_probs_month['P1 per Month']
    #save df
    df_probs_month.to_csv(os.path.join(out_dir,
                                   'P1 P0 per month %s.csv' %cascade_level),
                          sep=df_sep)
    return df_probs_month

##call fct W per month per stn
P01_per_month_L1 = probWeightsMonth(weight_month_left_L1,
                                    min_nbr_vals, cascade_level_1)

P01_per_month_L2 = probWeightsMonth(weight_month_left_L2,
                                    min_nbr_vals, cascade_level_2)

P01_per_month_L3 = probWeightsMonth(weight_month_left_L3,
                                    min_nbr_vals, cascade_level_3)
print('done with calculating P0 and P1 per month, fitting a beta pdf to Ws')

#==============================================================================
# beta distribution for 0<W<1, max likelihood to find parameters
#==============================================================================
'''
the beta distribution is assumed to best represent the distribution of the
weights of each level, the function has two parameters (alfa, beta) that has to
be identified in each level.
It is also possible to use a beta distribution with one parameter, a symmetric
distribution with one parameter beta.
maximazing the likelihood function is used to find beta
'''

def maxLikelihood(dict_weights, cascade_level):

    '''
        Idea: fit a symmetric beta fct to sampled weights
              MDRC parameter two: beta

        input: dict weights values
        output: df__fitted_parameters_result
    '''

    #define df to hold parameters result per station as ouput
    global df_parameters_out
    df_parameters_out = pd.DataFrame(index=dict_weights.keys())

    #call the beta distribution function from scipy
    beta_dist_fct = stats.beta

    #define constraint when using symmetric beta_dist_fct: alfa=beta
    cons = {'type' : 'eq', 'fun': lambda x: x[1] - x[0]}

    #constrain beta values >1 otherwise won't be an envelop of the data
    bounds = [(None, None), (None, None)] #change this to improve fit

    #start going through each staion sampled weights
    for station in dict_weights.keys():

            #get values as np array
            values = np.array([dict_weights[station]])

            #select values 0< W <1 (beta_dist_fct:0<x<1)
            nonnull_w_values = values[(values != 0) & (values != 1)]

            if len(nonnull_w_values) >= min_nbr_vals: #if enough Ws

                #calculate log beta_pdf(w, alfa, beta)
                _logpdf = lambda x, alfa, _beta:\
                            beta_dist_fct(a=alfa, b=_beta).logpdf(x)
                vec = np.vectorize(_logpdf) #easier to work with these vals

                #def fct to calculate sum(Log(beta_pdf(x)))
                def unliklihood(scale):
                    alfa = scale[0]
                    _beta = scale[1]
                    e = vec(nonnull_w_values, alfa, _beta)
                    return -np.sum(e) #negative bcz minimise not maximising

                #optimise fct to find parameters
                res = optimize.minimize(unliklihood,
                                        x0=[2.2, 2.2], #choice of intial param
                                        bounds=bounds, #should be in bounds
                                        method='SLSQP',
                                        constraints=cons,
                                        tol=1e-20)

                #save parameters to df_prameters_out
                df_parameters_out.loc[station, 'alfa'] = res.x[0]
                df_parameters_out.loc[station, 'beta'] = res.x[1]

            else: # if not enough weights for a station
                df_parameters_out.loc[station, 'alfa'] = -999
                df_parameters_out.loc[station, 'beta'] = -999

            #save df
            df_parameters_out.to_csv(os.path.join(out_dir,
                                          'bounded maximum likelihood %s.csv'\
                                          %cascade_level),
                                     sep=df_sep)
    return df_parameters_out

#call fct to find beta parameter level one and two
df_beta_params_L1 = maxLikelihood(weights_left_L1, cascade_level_1)
df_beta_params_L2 = maxLikelihood(weights_left_L2, cascade_level_2)
df_beta_params_L3 = maxLikelihood(weights_left_L3, cascade_level_3)

print('done with fitting beta to weights, finding unbounded model params')

#==============================================================================
# UNBOUNDED MODEL
#==============================================================================
'''
Idea: introduce volume dependancy of the P01 parameter on rainfall R
the unbounded model models the relationship between R and P01 using logistic
regression. (bardossy).
P01 = 1- 1/(1+e^-Z01)
Z01 = a + b.log10R
a and b are coefficients to be estimated using the log-maxlikelihood
'''

def volumeDependacy(in_df_w_stn_dict, cascade_level):

    #define out_dir for df_values_out of log Likelihood fct
    out_dir_unboud_model = os.path.join(out_dir,
                                        (r'%s P01 volume dependancy'\
                                         %cascade_level))

    if not os.path.exists(out_dir_unboud_model):
        os.mkdir(out_dir_unboud_model)

    #define out_dir for df_likelihood_params result
    out_dir_level_one_params = os.path.join(out_dir_unboud_model,
                                            (r'%s log_regress params'\
                                             %cascade_level))

    if not os.path.exists(out_dir_level_one_params):
        os.mkdir(out_dir_level_one_params)

    '''
    FIRST model the dependance of R on P01
    define classes for R vals, for every
    class find the P01 from observed Ws
    plot P01 vs R mean of each class
    use the maximum likelihood and the
    proposed logistic regeression model to
    find the parameters of the conceptual
    model that should reflect the dependency
    between P01 and R values
    '''

    #these dfs will hold output
    global df_dependancy, df_likelihood, res

    #new df to hold results of likelihood of logRegression params a and b
    df_dependancy = pd.DataFrame(index=in_df_w_stn_dict.keys())

    #perform for every station
    for stn in in_df_w_stn_dict.keys():

        #get stn values, bcz saved as a list in dfs_L1 = [df1, df2, ...]
        station_df = in_df_w_stn_dict[stn][0]

        #select R vals and weights (W1)
        R_vals = station_df['%s Original Volume' % stn]
        w1_vals = station_df['%s Sub Int Left' % stn]

        #new df to hold calculation of likelihood
        df_likelihood = pd.DataFrame(dtype=float)

        # add R vals and W vals to new df_likelihood
        df_likelihood['R vals'] = R_vals
        df_likelihood['W1 vals'] = w1_vals

        #go through W vals, if 0 or 1 relace with 0 in new col W01
        for idx, w_val in zip(df_likelihood.index,
                               df_likelihood['W1 vals'].values):
            if w_val == 0.0 or w_val == 1.0:
                df_likelihood.loc[idx, 'W 01'] = 0.
            else:
                df_likelihood.loc[idx, 'W 01'] = 1.

        #if W01=0 use log(f(R)) else use log(1-f(R))
        def build_likelihood_cols(df_likelihood, a, b):

            ''' fct introduced to calculate likelihood '''

            #calculate f(R)
            def logRegression(r_vals):
                return  np.array([1 - 1 / (1 + np.exp(\
                    - (np.array([a + b * np.log10(r_vals)]))))])

            #go through W01 vals and calculate cols of likelihood
            for idx, r_val, w_val_ in zip(df_likelihood.index,
                                          df_likelihood['R vals'].values,
                                          df_likelihood['W 01'].values):

                # if W=0 or W=1 usef log(f(precipitation))
                if w_val_ == 0.:
                    df_likelihood.loc[idx, 'log(L(teta))'] =\
                    (np.log(logRegression(r_val)))

                # if 0<W<1 use log(1-f(precipitation))
                elif w_val_ == 1.:
                    df_likelihood.loc[idx, 'log(L(teta))'] =\
                    np.log((1-logRegression(r_val)))

            #return the values of log(likelihood)
            values = df_likelihood['log(L(teta))'].values
            return values

        #what to minimise, cal fct on vals and optmise it to find params of fct
        def unliklihood2(scale):
            a1 = scale[0]
            b1 = scale[1]
            #will return the values log(likelihood), minimise the sum
            e = build_likelihood_cols(df_likelihood, a1, b1)
            return -np.sum((e))

        #result of optimisation
        res = optimize.minimize(unliklihood2,
                                x0=[-1., 1.9], #adjust initial vals, inbounds
                                method='SLSQP',
                                bounds=((-5,5), (1.5,10)), #adjust bounds
                                tol=1e-15,
                                options={'maxiter':10000.})

        #save results to df, will be used for plotting
        df_likelihood.to_csv(os.path.join(out_dir_unboud_model,
                                          'volume dependace of P01 %s.csv'\
                                          %stn),
                             sep=df_sep,
                             index_label=stn)

        #extract params result to df_dependancy
        df_dependancy.loc[stn, 'a'] = res.x[0]
        df_dependancy.loc[stn, 'b'] = res.x[1]

    #save df_dependancy, contains for each stn, values of a and b
    df_dependancy.to_csv(os.path.join(out_dir_level_one_params,
                                     'loglikehood params.csv'),
                         sep=df_sep,
                         index_label=stn)

    return df_dependancy

#call fct volume dependence level 1 and level 2
df_logRegress_params_L1_ = volumeDependacy(dfs_L1, cascade_level_1)
df_logRegress_params_L2 = volumeDependacy(dfs_L2, cascade_level_2)
df_logRegress_params_L3 = volumeDependacy(dfs_L3, cascade_level_3)

print('done with params of the unbounded model, proceeding to evaluation')

#==============================================================================
# Model EVALUATION
#==============================================================================
#raise Exception

def logRegression(r_vals, a, b):
    # this fct is called when getting P01 for R
    return  np.array([1 - 1 / (1 + np.exp(\
                - (np.array([a + b * np.log10(r_vals)]))))])

'''
First compare histograms observed and simulated rainfall
find and compare all values of R99 (frequency of extremes)
find and compare means of all R > R99 (magnitude of extremes)
evaluate frequencies and magnitudes

HOW:
    at every level L, for every R >= threshold:
    randomly assign for the first sub-interval W1 = 0 or W1 != 0 based on P01
    and if W1 != 0, sample W1 from Px, the fitted beta dist, with Beta param,
    R1 = W1*R and R2 = (1-W1)*R

    use input df_60min, and input for level L disaggregation is the model
    output of level L-1 disaggregation. So, read upper_df, sample weights
    based on P01 and Px, simulate Rvalues at 30min timestamps, use new 30min
    data as input to the lowest level, resample W based on P01 and Px, simulate
    R values that are at 15min timestamps.
    Compare the simulated 30min and 15min data to original values and compare
    frequency and magnitude of extremes

    First check results of baseline model, P01 not volume dependent
    Second use unbounded model to include dependency P01 and R
'''

def assignWvals(df_rain_upper_level,
                df_P01_probabilities_L1,
                df_beta_params_l1,
                df_logRegress_params_L1,
                df_P01_probabilities_L2_1,
                df_beta_params_L2_1,
                df_logRegress_params_L2_1,
                df_P01_probabilities_L3,
                df_beta_params_L3,
                df_logRegress_params_L3,
                nbr_realisation, R_threshhold):

    '''
    input:  df upper level
            df_param_P01_L1
            df_param_beta_L1
            df_logReg_param_L1
            df_param_P01_L2
            df_param_beta_L2
            df_logReg_param_L2
            nbr_simulations
            rainfall threshold

    output: simulated rainfall
            level_1: 30min
            level_2: 15min
    '''
    #out dir, results L1
    out_dir_model_eval_1 = os.path.join(out_dir,
                '%s model evaluation' %(cascade_level_1))
    if not os.path.exists(out_dir_model_eval_1): os.mkdir(out_dir_model_eval_1)

    #out dir, results L2
    out_dir_model_eval_2 = os.path.join(out_dir,
                '%s model evaluation' %(cascade_level_2))
    if not os.path.exists(out_dir_model_eval_2): os.mkdir(out_dir_model_eval_2)

    #out dir, results L3
    out_dir_model_eval_3 = os.path.join(out_dir,
                '%s model evaluation' %(cascade_level_3))
    if not os.path.exists(out_dir_model_eval_3): os.mkdir(out_dir_model_eval_3)

    #output dictionaries to read later when needed
    global dict_out_L1, dict_out_L2, dict_out_L3, result_simulations

    #dictionaries to hold outputs all stations
    dict_out_L1 = {k:[] for k in df_rain_upper_level.columns}
    dict_out_L2 = {k:[] for k in df_rain_upper_level.columns}
    dict_out_L3 = {k:[] for k in df_rain_upper_level.columns}

    #select values above threshold from upper df, what to disaggregate
    df_rain_upper_level = df_rain_upper_level\
                            [df_rain_upper_level > R_threshhold]

    #start iterating through stations
    for station in df_rain_upper_level.columns:

        print('simulating for :', station)

        #new df for every station simulations
        df_output_L1 = pd.DataFrame()

        #df2 to hold ouput lower level simulations
        df_output_L2 = pd.DataFrame()

        #df3 to hold ouput lower level simulations
        df_output_L3 = pd.DataFrame()

        #drop nan from station values level 1
        df_rain_upper_level[station].dropna(axis=0, inplace=True)

        assert df_rain_upper_level[station].isnull().sum() == 0,\
            print('still nans in DF level 1')

#==============================================================================
# go through index df level 1
#==============================================================================
        for idx in df_rain_upper_level[station].index:

            #extract volume R in level 1, what to disaggregate
            val_level_1 = df_rain_upper_level[station].loc[idx]

            #end idx sub int one (12:00:00)
            idx1_1 = idx - pd.\
                Timedelta(seconds=time_delta_720m_right)
            assert idx1_1.hour == 12, 'evaluation, L1 locate 1st idx'

            #end idx sub int two (00:00:00)
            idx2_2 = idx
            assert idx2_2.hour == 0, 'evaluation,L1 locate 2st idx'

            #idx sub-int 1 in L2 360min
            idx_l2_1 = idx1_1 - pd.\
                Timedelta(seconds=time_delta_360m_right)
            assert idx_l2_1.hour == 6, 'evaluation,L2 locate 1st idx'

            #idx sub-int 2 in L2 360min
            idx_l2_2 = idx1_1
            assert idx_l2_2.hour == 12, 'evaluation,L2 locate 2st idx'

            #idx sub-int 3 in L2 360min
            idx_l2_3 = idx2_2- pd.\
                Timedelta(seconds=time_delta_360m_right)
            assert idx_l2_3.hour == 18, 'evaluation,L2 locate 3rd idx'

            #idx sub-int 4 in L2 360min
            idx_l2_4 = idx2_2
            assert idx_l2_4.hour == 0, 'evaluation,L2 locate 4th idx'

            #idx sub-int 1 in L3 180min
            idx_l3_1 = idx_l2_1 - pd.\
                Timedelta(seconds=time_delta_180m_right)
            assert idx_l3_1.hour == 3, 'evaluation,L3 locate 1st idx'

            #idx sub-int 2 in L3 180min
            idx_l3_2 = idx_l2_1
            assert idx_l3_2.hour == 6, 'evaluation,L3 locate 2st idx'

            #idx sub-int 3 in L3 180min
            idx_l3_3 = idx_l2_2 - pd.\
                Timedelta(seconds=time_delta_180m_right)
            assert idx_l3_3.hour == 9, 'evaluation,L3 locate 3rd idx'

            #idx sub-int 4 in L3 180min
            idx_l3_4 = idx_l2_2
            assert idx_l3_4.hour == 12, 'evaluation,L3 locate 4th idx'

            #idx sub-int 5 in L3 180min
            idx_l3_5 = idx_l2_3 - pd.\
                Timedelta(seconds=time_delta_180m_right)
            assert idx_l3_5.hour == 15, 'evaluation,L3 locate 5th idx'

            #idx sub-int 5 in L3 180min
            idx_l3_6 = idx_l2_3
            assert idx_l3_6.hour == 18, 'evaluation,L3 locate 6th idx'

            idx_l3_7 = idx_l2_4 - pd.\
                Timedelta(seconds=time_delta_180m_right)
            assert idx_l3_7.hour == 21, 'evaluation,L3 locate 7th idx'

            idx_l3_8 = idx_l2_4
            assert idx_l3_8.hour == 0, 'evaluation,L3 locate 8th idx'

#==============================================================================
# baseline model level one
#==============================================================================

            #read P01 and beta values for station baseline model
            p01 = df_P01_probabilities_L1.loc[station, 'P01']
            beta_val_L1 = df_beta_params_l1.loc[station, 'beta']

            #generate a random var and compare it to P01
            r1 = np.random.uniform()
            rv1 = np.random.uniform()

            #if below P01
            if r1 <= p01 :

                if rv1 <= 0.5 :
                    w1 = 0. #assign W1=0 W2=1
                elif rv1 > 0.5:
                    w1 = 1.

            elif r1 > p01:
                #use px to sample W
                w1 = np.random.beta(beta_val_L1, beta_val_L1, 1)

            #use value of w1 to calculate R1 and R2
            R1 = w1 * val_level_1
            R2 = (1-w1) * val_level_1

            #store simulated vals in df_out
            df_output_L1.loc\
                [idx1_1, 'baseline rainfall %s'%cascade_level_1] = R1

            df_output_L1.loc\
                [idx2_2, 'baseline rainfall %s'%cascade_level_1] = R2

#==============================================================================
# unbounded model level one
#==============================================================================

            #read P01 and beta values for station unbounded model
            a = df_logRegress_params_L1.loc[station, 'a']
            b = df_logRegress_params_L1.loc[station, 'b']

            #calculate f(R) unbounded model, P01=fct(R)
            p01_unbounded = logRegression(val_level_1, a, b)

            #generate a random var and compare it to P01
            r2 = np.random.uniform()
            rv2 = np.random.uniform()

            if r2 <= p01_unbounded :

                if rv2 <= 0.5 :
                    w1_u = 0. #assign W1=0 , w2=1
                elif rv2 > 0.5:
                    w1_u = 1.

            elif r2 > p01_unbounded:
                #use px to sample W
                w1_u = np.random.beta(beta_val_L1, beta_val_L1, 1)

            #use value of w1 to calculate R1 and R2
            R1_u = w1_u * val_level_1
            R2_u = (1-w1_u) * val_level_1

            #store simulated vals in df_out
            df_output_L1.loc\
            [idx1_1, 'unbounded rainfall %s'%cascade_level_1] = R1_u

            df_output_L1.loc\
            [idx2_2, 'unbounded rainfall %s'%cascade_level_1] = R2_u

#==============================================================================
# LEVEL TWO
#==============================================================================
            '''Input lower level 6hrs is output this level 12hrs'''
#==============================================================================
# baseline model L2
#==============================================================================

            #read P01 and beta values for station baseline model
            p01_2_1 = df_P01_probabilities_L2_1.loc[station, 'P01']
            beta_val_2_1 = df_beta_params_L2_1.loc[station, 'beta']

            #generate a random var and compare it to P01
            r3 = np.random.uniform()
            rv3 = np.random.uniform()

            if r3 <= p01_2_1:
                if rv3 <= 0.5 :
                    w1_2_1 = 0. #assign W1=0 , w2=1
                elif rv3 > 0.5:
                    w1_2_1 = 1.

            elif r3 > p01_2_1:

                #use px to sample W
                w1_2_1 = np.random.beta(beta_val_2_1, beta_val_2_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R1 >= R_threshhold_L2: #check again
            R1_2_1 = w1_2_1 * R1
            R2_2_1 = (1-w1_2_1) * R1

            #store simulated vals in df_out
            df_output_L2.loc\
                    [idx_l2_1, 'baseline rainfall %s'%cascade_level_2]\
                    = R1_2_1

            df_output_L2.loc\
                    [idx_l2_2, 'baseline rainfall %s'%cascade_level_2]\
                    = R2_2_1

            #generate a random var and compare it to P01
            r4 = np.random.uniform()
            rv4 = np.random.uniform()

            if r4 <= p01_2_1:

                if rv4 <= 0.5 :
                    w1_2_2 = 0. #assign W1=0 , w2=1
                elif rv4 > 0.5:
                    w1_2_2 = 1.

            elif r4 > p01_2_1:

                #use px to sample W
                w1_2_2 = np.random.beta(beta_val_2_1, beta_val_2_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R2 > R_threshhold_L2 :

            R1_2_2 = w1_2_2 * R2
            R2_2_2 = (1-w1_2_2) * R2

            #store simulated vals in df_out
            df_output_L2.loc\
            [idx_l2_3, 'baseline rainfall %s'%cascade_level_2] = R1_2_2

            df_output_L2.loc\
            [idx_l2_4, 'baseline rainfall %s'%cascade_level_2] = R2_2_2

#==============================================================================
# unbounded model
#==============================================================================
            #read P01 and beta values for station unbounded model

            a2_1 = df_logRegress_params_L2_1.loc[station, 'a']
            b2_1 = df_logRegress_params_L2_1.loc[station, 'b']

            #calculate f(R) unbounded model, P01=fct(R)
            p01_unbounded_2_1 = logRegression(R1_u, a2_1, b2_1)

            #generate a random var and compare it to P01
            r5 = np.random.uniform()
            rv5 = np.random.uniform()

            if r5 <= p01_unbounded_2_1 :

                if rv5 <= 0.5 :
                    w1_u_2_1 = 0. #assign W1=0 , w2=1
                elif rv5 > 0.5:
                    w1_u_2_1 = 1.

            elif r5 > p01_unbounded_2_1:

                #use px to sample W
                w1_u_2_1 = np.random.beta(beta_val_2_1, beta_val_2_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R1_u > R_threshhold_L2:

            R1_u_2_1 = w1_u_2_1 * R1_u
            R2_u_2_1 = (1-w1_u_2_1) * R1_u

            #store simulated vals in df_out
            df_output_L2.loc\
            [idx_l2_1, 'unbounded rainfall %s'%cascade_level_2]\
            = R1_u_2_1

            df_output_L2.loc\
            [idx_l2_2, 'unbounded rainfall %s'%cascade_level_2]\
            = R2_u_2_1

#### check again which is the input: baseline or unbounded
            p01_unbounded_2_2 = logRegression(R2_u, a2_1, b2_1)

            #generate a random var and compare it to P01
            r6 = np.random.uniform()
            rv6 = np.random.uniform()

            assert r6 != r5 != r4 != r3 != r2 != r1, 'same R.V used in level 2'

            if r6 <= p01_unbounded_2_2 :

                if rv6 <= 0.5 :
                    w1_u_2_2 = 0. #assign W1=0 , w2=1
                elif rv6 > 0.5:
                    w1_u_2_2 = 1.
            elif r6 > p01_unbounded_2_2:

                #use px to sample W
                w1_u_2_2 = np.random.beta(beta_val_2_1, beta_val_2_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R2_u > R_threshhold_L2:
            R1_u_2_2 = w1_u_2_2 * R2_u
            R2_u_2_2 = (1-w1_u_2_2) * R2_u

            #store simulated vals in df_out
            df_output_L2.loc\
                [idx_l2_3, 'unbounded rainfall %s'%cascade_level_2]\
                = R1_u_2_2
            df_output_L2.loc\
                [idx_l2_4, 'unbounded rainfall %s'%cascade_level_2]\
                = R2_u_2_2
#==============================================================================
# LEVEL THREE
#==============================================================================
            '''Input lower level 3hrs is output this level 6hrs'''
#==============================================================================
# baseline model L3
#==============================================================================

            #read P01 and beta values for station baseline model
            p01_3_1 = df_P01_probabilities_L3.loc[station, 'P01']
            beta_val_3_1 = df_beta_params_L3.loc[station, 'beta']

            #generate a random var and compare it to P01
            r7 = np.random.uniform()
            rv7 = np.random.uniform()

            if r7 <= p01_3_1:
                if rv7 <= 0.5 :
                    w1_3_1 = 0. #assign W1=0 , w2=1
                elif rv7 > 0.5:
                    w1_3_1 = 1.

            elif r7 > p01_3_1:

                #use px to sample W
                w1_3_1 = np.random.beta(beta_val_3_1, beta_val_3_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R1 >= R_threshhold_L2: #check again
            R1_3_1 = w1_3_1 * R1_2_1
            R2_3_1 = (1-w1_3_1) * R1_2_1

            #store simulated vals in df_out
            df_output_L3.loc\
                    [idx_l3_1, 'baseline rainfall %s'%cascade_level_3]\
                    = R1_3_1

            df_output_L3.loc\
                    [idx_l3_2, 'baseline rainfall %s'%cascade_level_3]\
                    = R2_3_1

            #generate a random var and compare it to P01
            r8 = np.random.uniform()
            rv8 = np.random.uniform()

            if r8 <= p01_3_1:

                if rv8 <= 0.5 :
                    w1_3_2 = 0. #assign W1=0 , w2=1
                elif rv8 > 0.5:
                    w1_3_2 = 1.

            elif r8 > p01_3_1:

                #use px to sample W
                w1_3_2 = np.random.beta(beta_val_3_1, beta_val_3_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R2 > R_threshhold_L2 :

            R1_3_2 = w1_3_2 * R2_2_1
            R2_3_2 = (1-w1_3_2) * R2_2_1

            #store simulated vals in df_out
            df_output_L3.loc\
            [idx_l3_3, 'baseline rainfall %s'%cascade_level_3] = R1_3_2

            df_output_L3.loc\
            [idx_l3_4, 'baseline rainfall %s'%cascade_level_3] = R2_3_2

            #generate a random var and compare it to P01
            r9 = np.random.uniform()
            rv9 = np.random.uniform()

            if r9 <= p01_3_1:

                if rv9 <= 0.5 :
                    w1_3_3 = 0. #assign W1=0 , w2=1
                elif rv9 > 0.5:
                    w1_3_3 = 1.

            elif r9 > p01_3_1:

                #use px to sample W
                w1_3_3 = np.random.beta(beta_val_3_1, beta_val_3_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R2 > R_threshhold_L2 :

            R1_3_3 = w1_3_3 * R1_2_2
            R2_3_3 = (1-w1_3_3) * R1_2_2

            #store simulated vals in df_out
            df_output_L3.loc\
            [idx_l3_5, 'baseline rainfall %s'%cascade_level_3] = R1_3_3

            df_output_L3.loc\
            [idx_l3_6, 'baseline rainfall %s'%cascade_level_3] = R2_3_3

            #generate a random var and compare it to P01
            r10 = np.random.uniform()
            rv10 = np.random.uniform()

            if r10 <= p01_3_1:

                if rv10 <= 0.5 :
                    w1_3_4 = 0. #assign W1=0 , w2=1
                elif rv10 > 0.5:
                    w1_3_4 = 1.

            elif r10 > p01_3_1:

                #use px to sample W
                w1_3_4 = np.random.beta(beta_val_3_1, beta_val_3_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R2 > R_threshhold_L2 :

            R1_3_4 = w1_3_4 * R2_2_2
            R2_3_4 = (1-w1_3_4) * R2_2_2

            #store simulated vals in df_out
            df_output_L3.loc\
            [idx_l3_7, 'baseline rainfall %s'%cascade_level_3] = R1_3_4

            df_output_L3.loc\
            [idx_l3_8, 'baseline rainfall %s'%cascade_level_3] = R2_3_4
#==============================================================================
# unbounded model
#==============================================================================
            #read P01 and beta values for station unbounded model

            a3_1 = df_logRegress_params_L3.loc[station, 'a']
            b3_1 = df_logRegress_params_L3.loc[station, 'b']

            #calculate f(R) unbounded model, P01=fct(R)
            p01_unbounded_3_1 = logRegression(R1_u_2_1, a3_1, b3_1)

            #generate a random var and compare it to P01
            r11 = np.random.uniform()
            rv11 = np.random.uniform()

            if r11 <= p01_unbounded_3_1 :

                if rv11 <= 0.5 :
                    w1_u_3_1 = 0. #assign W1=0 , w2=1
                elif rv11 > 0.5:
                    w1_u_3_1 = 1.

            elif r11 > p01_unbounded_3_1:

                #use px to sample W
                w1_u_3_1 = np.random.beta(beta_val_3_1, beta_val_3_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R1_u > R_threshhold_L2:

            R1_u_3_1 = w1_u_3_1 * R1_u_2_1
            R2_u_3_1 = (1-w1_u_3_1) * R1_u_2_1

            #store simulated vals in df_out
            df_output_L3.loc\
            [idx_l3_1, 'unbounded rainfall %s'%cascade_level_3]\
            = R1_u_3_1

            df_output_L3.loc\
            [idx_l3_2, 'unbounded rainfall %s'%cascade_level_3]\
            = R2_u_3_1

#### check again which is the input: baseline or unbounded
            p01_unbounded_3_2 = logRegression(R2_u_2_1, a3_1, b3_1)

            #generate a random var and compare it to P01
            r12 = np.random.uniform()
            rv12 = np.random.uniform()

            assert r12 != r11 != r10 != r9 != r8 != r7\
                    , 'same R.V used in level 3'

            if r12 <= p01_unbounded_3_2 :

                if rv12 <= 0.5 :
                    w1_u_3_2 = 0. #assign W1=0 , w2=1
                elif rv12 > 0.5:
                    w1_u_3_2 = 1.
            elif r12 > p01_unbounded_3_2:

                #use px to sample W
                w1_u_3_2 = np.random.beta(beta_val_3_1, beta_val_3_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R2_u > R_threshhold_L2:
            R1_u_3_2 = w1_u_3_2 * R2_u_2_1
            R2_u_3_2 = (1-w1_u_3_2) * R2_u_2_1

            #store simulated vals in df_out
            df_output_L3.loc\
                [idx_l3_3, 'unbounded rainfall %s'%cascade_level_3]\
                = R1_u_3_2
            df_output_L3.loc\
                [idx_l3_4, 'unbounded rainfall %s'%cascade_level_3]\
                = R2_u_3_2

            p01_unbounded_3_3 = logRegression(R1_u_2_2, a3_1, b3_1)

            #generate a random var and compare it to P01
            r13 = np.random.uniform()
            rv13 = np.random.uniform()

            assert r13 != r12 != r11 != r10 != r9 != r8\
                    , 'same R.V used in level 3'

            if r13 <= p01_unbounded_3_3 :

                if rv13 <= 0.5 :
                    w1_u_3_3 = 0. #assign W1=0 , w2=1

                elif rv13 > 0.5:
                    w1_u_3_3 = 1.

            elif r13 > p01_unbounded_3_3:

                #use px to sample W
                w1_u_3_3 = np.random.beta(beta_val_3_1, beta_val_3_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R2_u > R_threshhold_L2:
            R1_u_3_3 = w1_u_3_3 * R1_u_2_2
            R2_u_3_3 = (1-w1_u_3_3) * R1_u_2_2

            #store simulated vals in df_out
            df_output_L3.loc\
                [idx_l3_5, 'unbounded rainfall %s'%cascade_level_3]\
                = R1_u_3_3
            df_output_L3.loc\
                [idx_l3_6, 'unbounded rainfall %s'%cascade_level_3]\
                = R2_u_3_3

            p01_unbounded_3_4 = logRegression(R2_u_2_2, a3_1, b3_1)

            #generate a random var and compare it to P01
            r14 = np.random.uniform()
            rv14 = np.random.uniform()

            assert r14 != r13 != r12 != r9 != r8 != r7\
                    , 'same R.V used in level 3'

            if r14 <= p01_unbounded_3_4 :

                if rv14 <= 0.5 :
                    w1_u_3_4 = 0. #assign W1=0 , w2=1
                elif rv14 > 0.5:
                    w1_u_3_4 = 1.
            elif r14 > p01_unbounded_3_4:

                #use px to sample W
                w1_u_3_4 = np.random.beta(beta_val_3_1, beta_val_3_1, 1)

            #use value of w1 to calculate R1 and R2
#                if R2_u > R_threshhold_L2:
            R1_u_3_4 = w1_u_3_4 * R2_u_2_2
            R2_u_3_4 = (1-w1_u_3_4) * R2_u_2_2

            #store simulated vals in df_out
            df_output_L3.loc\
                [idx_l3_7, 'unbounded rainfall %s'%cascade_level_3]\
                = R1_u_3_4
            df_output_L3.loc\
                [idx_l3_8, 'unbounded rainfall %s'%cascade_level_3]\
                = R2_u_3_4

        #save df output
        df_output_L1.to_csv(os.path.join(out_dir_model_eval_1,
                                  'model results %s %d.csv' %(station,
                                                          nbr_realisation)),
                             sep=df_sep,
                             index_label=station)
        #save station results in dict output
        dict_out_L1[station] = df_output_L1

        #save df
        df_output_L2.to_csv(os.path.join(out_dir_model_eval_2,
                                     'model results %s %d.csv' %(station,
                                     nbr_realisation)),
                         sep=df_sep,
                         index_label=station)

        #save station results in dict output
        dict_out_L2[station] = df_output_L2

        #save df
        df_output_L3.to_csv(os.path.join(out_dir_model_eval_3,
                                     'model results %s %d.csv' %(station,
                                     nbr_realisation)),
                         sep=df_sep,
                         index_label=station)

        #save station results in dict output
        dict_out_L3[station] = df_output_L3

    result_simulations = [dict_out_L1, dict_out_L2, dict_out_L3]
    return  result_simulations

#define initial simulation nbr
simulation_nbr = 1

while simulation_nbr <= 3 : #upper limit to define nbr of simulations
    simulation_nbr += 1

    evaluation_L1 = assignWvals(df_60_to_1440m,
                            df_prob_p01_L1,
                            df_beta_params_L1,
                            df_logRegress_params_L1_,
                            df_prob_p01_L2,
                            df_beta_params_L2,
                            df_logRegress_params_L2,
                            df_prob_p01_L3,
                            df_beta_params_L3,
                            df_logRegress_params_L3,
                            simulation_nbr,
                            threshhold)
print('done with model evaluation, proceeding to Lorenz Curves')
#raise Exception

#==============================================================================
# Lorenz Curve: fct for Original data, fct for Simulated data
#==============================================================================
''' idea: find Irregularity of rainfall distribution
            calculate the cummulative percentage of precipitation contributed
            by the cummulative percentage of days when precipitaiton occured
    method: first sort rainfall values
            define threshhold of wet vs dry hours : mm/hr
            define rainfall amount for class intervals (mm)

    code: determine percentage of rain contributed by hours failling in each
          class and compare observed values and model output

          if the daily precipitation falls within a class interval i,
          the number of rainy days in this class, n, is added to index i
          and the amount of rain, xi, is totaled

'''

def buildLorenzCurve_Original(in_ppt_df, cascade_level):

    ''' input: precipitation ORIGINAL dfs_L1 , cascade level
        output: new df with values of Loren curve 'X', 'Y'
                read those in plotting script and plot them
    '''
    out_dir_ = os.path.join(out_dir,
                        '%s Lorenz curves original' %cascade_level)
    if not os.path.exists(out_dir_): os.mkdir(out_dir_)

    #slect data above threshhold check AGAIN
    in_ppt_df = in_ppt_df[in_ppt_df>threshhold]

    for station in in_ppt_df.columns:
#        print(station)

        #drop nans from station's data
        in_ppt_df[station].dropna(inplace=True)

        #new list to hold rounded values of station
        rainfall_values = []
        for val in in_ppt_df[station]:
            rainfall_values.append(round(val,2))

        #sort rainfall values ad get unique values
        rainfall_values_sorted = (np.\
                      sort(rainfall_values, kind='mergesort'))
        rainfall_values_unique = np.unique(rainfall_values_sorted)

        #calculate total rainfall volume and total nbr of values
        total_rainfall = float(np.sum(rainfall_values))
        total_values = len(rainfall_values)

        #find how values are classed
        nbr_vals_per_class = []
        for th in rainfall_values_unique:
            nbr_vals_per_class.append((rainfall_values<=th).sum())

        #nbr of occurence per rainfall per class
        rainfall_occurences = Counter(rainfall_values_sorted)

        #new df , index=rainfall values, colums= occurences
        df_rain_occurences = pd.DataFrame.\
            from_dict(rainfall_occurences, orient='index')

        #change column name to be 'Ni', nbr of occurences per Rainfall value
        df_rain_occurences.rename(columns={0:'Ni'}, inplace=True)

        #new column to hold accumulated values per class
        df_rain_occurences['sum Ni'] = nbr_vals_per_class
        #new column to hold rainfall contribution per class
        df_rain_occurences['Pi'] = df_rain_occurences.index\
                                    * df_rain_occurences['Ni']
        #new column to hold accumulated precipitation per class
        df_rain_occurences['sum Pi'] = df_rain_occurences['Pi'] .cumsum()

        #find contributions in frequency : occurences / total
        df_rain_occurences['X'] = df_rain_occurences['sum Ni'] / total_values

        #find contributions each class compared to total
        df_rain_occurences['Y'] = df_rain_occurences['sum Pi'] / total_rainfall

        #save df, what to plot is columns X and Y
        df_rain_occurences.to_csv(os.path.join(out_dir_,
          'Lorenz curve %s Observed values %s.csv' %(station,cascade_level)),
                sep=df_sep)
    return
#call fct to find lorenz curves
lorenz_L1 = buildLorenzCurve_Original(df_60_to_720m, cascade_level_1)
lorenz_L2 = buildLorenzCurve_Original(df_60m_to_360m, cascade_level_2)
lorenz_L3 = buildLorenzCurve_Original(df_60m_to_180m, cascade_level_3)

print('done with Lorenz curves of observed data, proveeding to simulated data')
#==============================================================================
#
#==============================================================================

def buildLorenzCurve_Simulations(in_sim_dfs_L1_dict, cascade_level):

    ''' input: precipitation SIMULATED dict{ stn: basline, unbouded}
                cascade level
        output: new df with values of Loren curve 'X', 'Y'
                read those in plotting script and plot them
    '''
    out_dir_2 = os.path.join(out_dir,
                        '%s Lorenz curves simulations' %cascade_level)

    if not os.path.exists(out_dir_2): os.mkdir(out_dir_2)

    base_name = 'baseline rainfall %s' %cascade_level
    unbound_name = 'unbounded rainfall %s' %cascade_level
#    print(base_name, unbound_name)

    if  cascade_level == cascade_level_1:

#        print(cascade_level)

        #extract results L1 from simulated vals
        sim_Result = in_sim_dfs_L1_dict[0]

        #start going through stations
        for station in sim_Result.keys():

            #extract baseline values
            baseline_values = sim_Result[station][base_name]

            #extract unbouded model values
            unbounded_values = sim_Result[station][unbound_name]

            #add values to a list
            values_simulations = [baseline_values, unbounded_values]

            #start going through simulated values
            for i, model_results in enumerate(values_simulations):

                #new list to hold model values of station
                rainfall_values = []
                for val in model_results:
                    rainfall_values.append(round(val,2))

                #sort rainfall values ad get unique values
                rainfall_values_sorted = (np.\
                              sort(rainfall_values, kind='mergesort'))
                rainfall_values_unique = np.unique(rainfall_values_sorted)

                #find total rainfall volume
                total_rainfall = float(np.sum(rainfall_values))

                #find total nbr of values per station
                total_values = len(rainfall_values)

                #new list to hold values per class
                nbr_vals_per_class = []
                #append values in each class, using unique R values
                for th in rainfall_values_unique:
                    nbr_vals_per_class.append((rainfall_values<=th).sum())

                #count nbr of occurence per rainfall
                rainfall_occurences = Counter(rainfall_values_sorted)

                #new df , index=rainfall values, colums= occurences
                df_rain_occurences = pd.DataFrame.\
                    from_dict(rainfall_occurences, orient='index')
                df_rain_occurences.rename(columns={0:'Ni'}, inplace=True)
                #get nbr or rainfall per class
                df_rain_occurences['sum Ni'] = nbr_vals_per_class

                df_rain_occurences['Pi'] = df_rain_occurences.index\
                                            * df_rain_occurences['Ni']
                df_rain_occurences['sum Pi'] =\
                    df_rain_occurences['Pi'].cumsum()

                df_rain_occurences['X'] =\
                    df_rain_occurences['sum Ni'] / total_values

                df_rain_occurences['Y'] =\
                    df_rain_occurences['sum Pi'] / total_rainfall

                if i == 0: model_name = 'baseline'
                elif i== 1: model_name = 'unbounded'

                df_rain_occurences.to_csv(os.path.join(out_dir_2,
                           'Lorenz curve %s %s %s.csv' %(station,
                                            model_name,
                                            cascade_level)), sep=df_sep)

    elif  cascade_level == cascade_level_2:

        sim_Result_ = in_sim_dfs_L1_dict[1]

        for station in sim_Result_.keys():

            baseline_values = sim_Result_[station][base_name]

            unbounded_values = sim_Result_[station][unbound_name]

            values_simulations = [baseline_values, unbounded_values]

            for i, model_results in enumerate(values_simulations):

                #new list to hold model values of station
                rainfall_values = []
                for val in model_results:
                    rainfall_values.append(round(val,2))

                #sort rainfall values ad get unique values
                rainfall_values_sorted = (np.\
                              sort(rainfall_values, kind='mergesort'))
                rainfall_values_unique = np.unique(rainfall_values_sorted)

                total_rainfall = float(np.sum(rainfall_values))
                total_values = len(rainfall_values)

                nbr_vals_per_class = []

                for th in rainfall_values_unique:
                    nbr_vals_per_class.append((rainfall_values<=th).sum())

                #nbr of occurence per rainfall
                rainfall_occurences = Counter(rainfall_values_sorted)

                #new df , index=rainfall values, colums= occurences
                df_rain_occurences = pd.DataFrame.\
                    from_dict(rainfall_occurences, orient='index')
                df_rain_occurences.rename(columns={0:'Ni'}, inplace=True)

                df_rain_occurences['sum Ni'] = nbr_vals_per_class

                df_rain_occurences['Pi'] = df_rain_occurences.index\
                                            * df_rain_occurences['Ni']

                df_rain_occurences['sum Pi'] =\
                    df_rain_occurences['Pi'] .cumsum()

                df_rain_occurences['X'] =\
                    df_rain_occurences['sum Ni'] / total_values

                df_rain_occurences['Y'] =\
                    df_rain_occurences['sum Pi'] / total_rainfall

                if i == 0: model_name = 'baseline'
                elif i== 1: model_name = 'unbounded'

                df_rain_occurences.to_csv(os.path.join(out_dir_2,
                                   'Lorenz curve %s %s %s.csv' %(station,
                                            model_name,
                                            cascade_level)),
                    sep=df_sep)

    elif  cascade_level == cascade_level_3:

        sim_Result_ = in_sim_dfs_L1_dict[2]

        for station in sim_Result_.keys():

            baseline_values = sim_Result_[station][base_name]

            unbounded_values = sim_Result_[station][unbound_name]

            values_simulations = [baseline_values, unbounded_values]

            for i, model_results in enumerate(values_simulations):

                #new list to hold model values of station
                rainfall_values = []
                for val in model_results:
                    rainfall_values.append(round(val,2))

                #sort rainfall values ad get unique values
                rainfall_values_sorted = (np.\
                              sort(rainfall_values, kind='mergesort'))
                rainfall_values_unique = np.unique(rainfall_values_sorted)

                total_rainfall = float(np.sum(rainfall_values))
                total_values = len(rainfall_values)

                nbr_vals_per_class = []

                for th in rainfall_values_unique:
                    nbr_vals_per_class.append((rainfall_values<=th).sum())

                #nbr of occurence per rainfall
                rainfall_occurences = Counter(rainfall_values_sorted)

                #new df , index=rainfall values, colums= occurences
                df_rain_occurences = pd.DataFrame.\
                    from_dict(rainfall_occurences, orient='index')
                df_rain_occurences.rename(columns={0:'Ni'}, inplace=True)

                df_rain_occurences['sum Ni'] = nbr_vals_per_class

                df_rain_occurences['Pi'] = df_rain_occurences.index\
                                            * df_rain_occurences['Ni']

                df_rain_occurences['sum Pi'] =\
                    df_rain_occurences['Pi'] .cumsum()

                df_rain_occurences['X'] =\
                    df_rain_occurences['sum Ni'] / total_values

                df_rain_occurences['Y'] =\
                    df_rain_occurences['sum Pi'] / total_rainfall

                if i == 0: model_name = 'baseline'
                elif i== 1: model_name = 'unbounded'

                df_rain_occurences.to_csv(os.path.join(out_dir_2,
                                   'Lorenz curve %s %s %s.csv' %(station,
                                            model_name,
                                            cascade_level)),
                    sep=df_sep)

    return

lorenz_sim_L1= buildLorenzCurve_Simulations(evaluation_L1, cascade_level_1)
lorenz_sim_L2 = buildLorenzCurve_Simulations(evaluation_L1, cascade_level_2)
lorenz_sim_L3 = buildLorenzCurve_Simulations(evaluation_L1, cascade_level_3)

#==============================================================================
#
#==============================================================================
'''
1. DATA INTEGRITY CHECK ()
2. Accuracy check , quantify errors, why the model fails?
3. long term effect of the algorithm, feed back loop
'''
STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))
