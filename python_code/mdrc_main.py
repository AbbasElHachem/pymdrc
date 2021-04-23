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


import os
import timeit
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from scipy.stats import beta
from scipy.special import gamma as gammaf
from scipy.optimize import minimize


class CascadeModel():

    def __init__(self, df, cascade_levels=None):
        self.df = df
        self.cascade_levels = cascade_levels
        return

    def find_init_temp_agg(self, df):
        temp_agg = (df.index[1] - df.index[0])
        if temp_agg.days == 1:
            self.temp_agg = 1440
        else:
            self.temp_agg = temp_agg.seconds / 60

        return self.temp_agg

    def resampleDf(self, df, agg,
                   closed='right', label='right',
                   shift=False, leave_nan=True,
                   label_shift=None,
                   temp_shift=0,
                   max_nan=0):
        """
        Purpose: Aggregate precipitation data

        Parameters:
        -----------
        Df: Pandas DataFrame Object
            Data set
        agg: string
            Aggregation 'M':Monthly 'D': Daily,
             'H': Hourly, 'Min': Minutely
        closed: string
            'left' or 'right' defines the aggregation interval
        label: string
            'left' or 'right' defines the related timestamp
        shift: boolean, optional
            Shift the values by 6 hours 
            according to the dwd daily station.
            Only valid for aggregation into daily aggregations
            True, data is aggregated from 06:00 - 06:00
            False, data is aggregated from 00:00 - 00:00
            Default is False

        temp_shift: shift the data based on 
        timestamps (+- 0 to 5), default: 0

        label_shift: shift time label by certain values (used for timezones)

        leave_nan: boolean, optional
            True, if the nan values should remain
             in the aggregated data set.
            False, if the nan values should be 
            treated as zero values for the
            aggregation. Default is True

        Remark:
        -------
            If the timestamp is at the end of the timeperiod:

            Input: daily        Output: daily+
                >> closed='right', label='right'

            Input: subdaily     Output: subdaily
                >> closed='right', label='right'

            Input: subdaily     Output: daily
                >> closed='right', label='left'

            Input: subdaily     Output: monthly
                >> closed='right', label='right'


            ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !
            ! ! Always check, if aggregation is correct ! !
            ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !



        """

        if shift == True:
            df_copy = df.copy()
            if agg != 'D' or agg != '1440min':
                raise Exception(
                    'Shift can only be applied to daily aggregations')
            df = df.shift(-6, 'H')

        # To respect the nan values
        if leave_nan == True:
            # for max_nan == 0, the code runs faster if implemented as follows
            if max_nan == 0:
                # print('Resampling')
                # Fill the nan values with values very great negative values and later
                # get the out again, if the sum is still negative
                df = df.fillna(-100000000000.)
                df_agg = df.resample(agg,
                                     closed=closed,
                                     label=label,
                                     offset=temp_shift,
                                     # offset or origin new argument
                                     loffset=label_shift).sum()
                # Replace negative values with nan values
                df_agg.values[df_agg.values[:] < 0.] = np.nan
            else:
                df_agg = df.resample(rule=agg,
                                     closed=closed,
                                     label=label,
                                     base=temp_shift,
                                     loffset=label_shift).sum()
                # find data with nan in original aggregation
                g_agg = df.groupby(pd.Grouper(freq=agg,
                                              closed=closed,
                                              label=label))
                n_nan_agg = g_agg.aggregate(lambda x: pd.isnull(x).sum())

                # set aggregated data to nan if more than max_nan values occur in the
                # data to be aggregated
                filter_nan = (n_nan_agg > max_nan)
                df_agg[filter_nan] = np.nan

        elif leave_nan == False:
            df_agg = df.resample(agg,
                                 closed=closed,
                                 label=label,
                                 base=temp_shift,
                                 loffset=label_shift).sum()
        if shift == True:
            df = df_copy
        return df_agg
    pass


if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    main_dir = os.path.join(
        r'X:\staff\elhachem\ClimXtreme\04_analysis\08_cascade_model')
    os.chdir(main_dir)

    StationPath = os.path.join(main_dir, r'P00003_1min_data.csv')
    assert os.path.exists(StationPath)

    # read df
    StationData_row = pd.read_csv(StationPath, sep=';',
                                  index_col=0,
                                  engine='c')

    StationData_row.index = pd.to_datetime(
        StationData_row.index, format="%Y-%m-%d %H:%M:%S")

    main_class = CascadeModel(df=StationData_row)

    input_agg = main_class.find_init_temp_agg(StationData_row)
    resample_df = main_class.resampleDf(StationData_row,
                                        '60min')

    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
