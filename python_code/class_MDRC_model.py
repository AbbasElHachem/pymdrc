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
import numpy as np
# from numpy.linalg import inv
import pandas as pd
# import tarfile
import os
import timeit
# import datetime
# from datetime import timedelta
# from datetime import datetime
import time
# import glob
# import pickle
import matplotlib.pyplot as plt


# from dateutil.relativedelta import relativedelta
# from scipy.stats import norm
# import calendar
# from pandas.tseries.frequencies import to_offset
from scipy.stats import beta
from scipy.special import gamma as gammaf
from scipy.optimize import minimize
# from sklearn import linear_model
# import math

main_dir = os.path.join(
    r'X:\staff\elhachem\ClimXtreme\04_analysis\08_cascade_model')
os.chdir(main_dir)

StationPath = os.path.join(main_dir, r'oneStationReutlingen.txt')
StationPath = (
    r"X:\hiwi\ElHachem\Jochen\CascadeModel\oneStationReutlingen.csv")
assert StationPath

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program


# =============================================================================
#  data process
# =============================================================================

# divided by 10
StationData_row = pd.read_csv(StationPath, sep=';', header=None, names=[
                              'value'], index_col=0) / 10


StationData_row.index = pd.to_datetime(
    StationData_row.index, format="%Y-%m-%d %H:%M:%S")
# extreme value
#StationData_row[StationData_row.value >10]

# try to let the value bigger than 10 times divided by 10 again
StationData_row[StationData_row.value > 10] = np.nan


threshold = 0.3
# =============================================================================
#  Casade - W value and P01
# =============================================================================
# %%
# threshold 0.3


def grouppe(df, timestep):
    df_unter = df.copy()
    df_unter = df.resample(timestep, label='right').sum()
    return df_unter


S_60min = grouppe(StationData_row, '1H')
S_30min = grouppe(StationData_row, '30T')
S_15min = grouppe(StationData_row, '15T')

# try try

# the length sould be two times larger

#


def fulfill_er(df_oben, df_unten, timestep_unten):

    if 2 * len(df_oben) - len(df_unten) == 1:
        df = df_unten.copy()
        df.loc[df.index[0] -
               pd.offsets.Minute(timestep_unten), 'value'] = df_oben.value[0] - df.value[0]
        df = df.sort_index()

    elif 2 * len(df_oben) - len(df_unten) == 3:
        df = df_unten.copy()
        # third
        df.loc[df.index[0] -
               pd.offsets.Minute(timestep_unten), 'value'] = df_oben.value[1] - df.value[0]
        df = df.sort_index()

    #second and one
    for i in range(0, 2):
        df.loc[df.index[0] -
               pd.offsets.Minute(timestep_unten), 'value'] = df_oben.value[0] / 2
        df = df.sort_index()
    else:
        pass

    return df


#
#
# S_30min2 = fulfill_er(S_60min, S_30min, 30)
# S_15min2 = fulfill_er(S_30min, S_15min, 15)

# check the sum


def sum_check(df1, df2):
    if abs(df1.value.sum() - df2.value.sum()) <= 10**-4:
        pass
    else:
        print('error')


sum_check(StationData_row, S_60min)
sum_check(StationData_row, S_30min)
sum_check(StationData_row, S_15min)


# def fulfill_er(df, df_oben, df_timestep):
#    df = df.combine_first(df_oben)
#    df = df.resample(df_timestep).ffill()
#    return df

#S_30min = fulfill_er(S_30min, S_60min, '30T')
#S_15min = fulfill_er(S_15min, S_30min, '15T')


# calculate w1 and w2
#※ think about how to set threshold

# def para_W1(df_oben, df_unten):
#    W1 = df_unten.iloc[np.arange(len(df_oben)) * 2].value / df_oben.value
# S_60min.iloc[:]
#    return W1
#W1_1er = para_W1(S_60min,S_30min)
#W1_2er = para_W1(S_30min,S_15min)


def para_W2(df_oben, df_unten, pcp_thr):
    # pcp_thr = 0.1
    df_oben_copy = df_oben.copy(deep=True)
    df_unten_copy = df_unten.copy(deep=True)

    # get_all_pcp_abv_thr
    df_oben_abv_thr = df_oben_copy[df_oben_copy >= pcp_thr].dropna()

    shift_freq = (df_unten_copy.index[1] - df_unten_copy.index[0])

    index_w1 = df_unten_copy.index.intersection(
        df_oben_abv_thr.index - shift_freq)
    index_w2 = df_unten_copy.index.intersection(
        df_oben_abv_thr.index)

    index_oben_w1 = index_w1 + shift_freq
    w1 = (df_unten_copy.loc[index_w1, :] /
          df_oben_abv_thr.loc[index_oben_w1, :].values)
    w2 = df_unten_copy.loc[index_w2, :] / df_oben_abv_thr.values
    # assert np.all(w1.values + w2.values) == 1
    return w1


W2_1er = para_W2(S_60min, S_30min, pcp_thr=threshold)
W2_2er = para_W2(S_30min, S_15min, threshold)


def cascade_sort(W):
    cats = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
            'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    months = pd.Categorical(W.index.strftime(
        '%b'), categories=cats, ordered=True)
    W_sort = W.groupby([months, W.index.time]).mean().unstack()
    return W_sort.mean(axis=1)


W2_1er_sort = cascade_sort(W2_1er)
W2_2er_sort = cascade_sort(W2_2er)


def valueP01(W2_1er):
    p0 = W2_1er[W2_1er == 0].dropna().size / W2_1er.size
    p1 = W2_1er[W2_1er == 1].dropna().size / W2_1er.size

    p01 = p0 + p1
    return p01
# use W2
# threshold set here (0.3)
# than separate the P01 and P andere to calculate


# def valueP012(df_oben, df_unten):
# dic = {'value_unten': df_unten.iloc[np.arange(
# len(df_oben)) * 2 + 1].value, 'value_above': df_oben.value}
# df = pd.DataFrame(dic)
# df2 = df[df.value_above >= threshold]
# w = df2.value_unten / df2.value_above
# P01 = len(w[(w == 0) | (w == 1)]) / len(w)
# return P01


P01_1er = valueP01(W2_1er)
P01_2er = valueP01(W2_2er)

#
# def valueP01_monthly(df_oben, df_unten):
#
# dic = {'value_unten': df_unten.iloc[np.arange(
# len(df_oben)) * 2 + 1].value, 'value_above': df_oben.value}
# df = pd.DataFrame(dic)
# df2 = df[df.value_above >= threshold]
# w = df2.value_unten / df2.value_above
#
# month_list = []
# for i in range(0, 12):
# w_mon = w.loc[(w.index.month == (i + 1))]
# P01 = len(w_mon[(w_mon == 0) | (w_mon == 1)]) / len(w_mon)
# month_list.append(P01)
# del w_mon
# return month_list


def valueP01_monthly(W2_1er):

    month_list = []
    for i in range(0, 12):
        w_mon = W2_1er.loc[(W2_1er.index.month == (i + 1))]
        P01 = len(w_mon[(w_mon == 0) | (w_mon == 1)
                        ].dropna()) / len(w_mon)
        month_list.append(P01)

    return month_list


P01_mon1er = valueP01_monthly(W2_1er)
P01_mon2er = valueP01_monthly(W2_2er)

# plot w = 0 | w = 1
plt.ioff()
plt.figure(1)
plt.scatter(np.arange(1, 12 + 1), P01_mon1er)
plt.scatter(np.arange(1, 12 + 1), P01_mon2er)
plt.title(r'$P_0$' + r'$_1$' + '(Monthly)', fontsize=20)
plt.xlabel('Month', fontsize=15)
plt.ylabel('Percentage', fontsize=15)
plt.xlim((0.5, 12.5))
plt.ylim((0, 1))
plt.legend(['1st layer', '2nd layer'])
plt.grid(True)
plt.show()

# plot  0 < w < 1


# def W_innerhalb(w_layer, station_oben, station_unten):
# df = pd.concat([w_layer, station_unten.iloc[np.arange(
# len(station_oben)) * 2 + 1], station_oben.value], axis=1)
# df.columns = ['percent', 'amount', 'threshold_check']
# w_in = df[df.threshold_check >= threshold]
# w_in = w_in[(w_in.percent > 0) & (w_in.percent < 1)]
# plt.plot(w_in.percent, w_in.amount, 'ro')
# plt.xlabel('W value')
# plt.ylabel('Amount')
# plt.show()

def W_innerhalb(W2_1er, df_oben, pcp_thr):
    df_oben_abv_thr = df_oben[df_oben >= pcp_thr].dropna()
    # assert len(df_oben_abv_thr.index) == len(W2_1er.index)
    shift_freq = (df_oben.index[1] - df_oben.index[0]) / 2
    idx_weights_in_01 = W2_1er[
        (W2_1er > 0) & (W2_1er < 1)].dropna().index
    idx_pcp_for_w = idx_weights_in_01 + shift_freq
    w_in = W2_1er[(W2_1er > 0) & (W2_1er < 1)].dropna()
    pcp_abv = df_oben_abv_thr.loc[idx_pcp_for_w, :].dropna()
    plt.plot(w_in.values, pcp_abv.values, 'ro')
    plt.xlabel('W value')
    plt.ylabel('Amount')
    plt.show()


plt.figure(2)

W_innerhalb(W2_1er, S_60min, threshold)
plt.figure(3)
W_innerhalb(W2_2er, S_30min, threshold)
# =============================================================================
#  sorting the number of W  and create a excel table
# =============================================================================
# %%
# equal to 0 or 1


def W_month_number01(W2_1er, number):
    listnumber = []
    for i in range(1, 13):

        w_mon = W2_1er.loc[(W2_1er.index.month == (i))]
        w_nbr = len(w_mon[(w_mon == number)].dropna())
        listnumber.append(w_nbr)
    return listnumber


W21_M0 = W_month_number01(W2_1er, 0)
W22_M0 = W_month_number01(W2_2er, 0)

W21_M1 = W_month_number01(W2_1er, 1)
W22_M1 = W_month_number01(W2_2er, 1)

# between 0 and 1


def W_month_numberin(W2_1er):
    listnumber = []
    for i in range(1, 13):
        w_mon = W2_1er.loc[(W2_1er.index.month == (i))]
        w_nbr = len(
            w_mon[(w_mon == 0) | (w_mon == 1)].dropna())
        listnumber.append(w_nbr)
    return listnumber


W21_Min = W_month_numberin(W2_1er)
W22_Min = W_month_numberin(W2_2er)

# =============================================================================
#  Beta function (Beta value) and plot
# =============================================================================
# %%
# symmetric apha equals to beta -> one parameter
# maximum likelihood


class betafit:

    # def dfcreate(self, w_layer, station_oben, station_unten):
        # df = pd.concat([w_layer, station_unten.iloc[np.arange(0,
                                                              # len(station_oben)) * 2], station_oben.value], axis=1)
        # df.columns = ['percent', 'amount', 'threshold_check']
        #
    # # choose the data with constraint ( > threshold , 0 < w < 1)
        # w_in = df[df.threshold_check >= threshold]
        # w_in = w_in[(w_in.percent != 1) & (w_in.percent != 0)]
        # return w_in
    def dfcreate(self, W2_1er, df_oben, pcp_thr):
        # df_oben_abv_thr = df_oben[df_oben >= pcp_thr].dropna()
        # assert len(df_oben_abv_thr.index) == len(W2_1er.index)
        # shift_freq = (df_oben.index[1] - df_oben.index[0]) / 2
        # idx_weights_in_01 = W2_1er[
            # (W2_1er > 0) & (W2_1er < 1)].dropna().index
        # idx_pcp_for_w = idx_weights_in_01 + shift_freq
        w_in = W2_1er[(W2_1er > 0) & (W2_1er < 1)].dropna()
        # pcp_abv = df_oben_abv_thr.loc[idx_pcp_for_w, :].dropna()

        df = pd.DataFrame(index=w_in.index)
        df['percent'] = w_in.values
        # df['amount_oben'] = pcp_abv
        return df
    # Optimization

    def obj_logbetafun(self, x, sign=1.0):

        df_beta_ob = df_beta.percent.copy()
        summa = - np.log(beta.pdf(df_beta_ob, x[0], x[1])).sum()  # min sum
        return summa

    #print (result_beta.x)
    #---------------------------------------------
    # plot --> 2 teilen (1.) histgram (2.) fit line
    #---------------------------------------------
    def betafun(self, x, a, b):
        return gammaf(a + b) / gammaf(a) / gammaf(b) * x**(a - 1) * (1 - x)**(b - 1)

    def plot_fitbeta(self, df):
        fig = plt.figure(4)
        ax = fig.add_subplot(111)
        ax.scatter(df.percent, bf.betafun(
            df.percent, result_beta.x[0], result_beta.x[1]),
            c='r', zorder=1)

        # center labels in histogram plot
        # def bins_labels(bins, **kwargs):
        #    bin_w = (max(bins) - min(bins)) / (len(bins) - 1)
        #    plt.xticks(np.arange(min(bins)+bin_w/2, max(bins), bin_w), bins, **kwargs)
        #    plt.xlim(0, 1)

        #bins_number = 50
        #bins = np.arange(0 + 1/bins_number , 1, 1/bins_number)
        ax.hist(df.percent, bins=50, color='blue',
                align='mid', density=True, zorder=0)
        #bins_labels(bins, fontsize=10)

        ax.text(0.7, 2.4, r'1 Layer', fontsize=15)
        ax.text(0.7, 2.2, r'$\beta$ = %.2f' % result_beta.x[0], fontsize=15)

        plt.xlabel('W value', fontsize=16)
        plt.ylabel('Probability density function', fontsize=16)
        plt.grid(True)

        plt.show()


bf = betafit()
df_beta = bf.dfcreate(W2_2er, S_30min, pcp_thr=0.)
#df_beta = bf.dfcreate(W2_2er, S_30min ,S_15min)

cons = {'type': 'eq',
        'fun': lambda x: np.array(x[0] - x[1])}

result_beta = minimize(bf.obj_logbetafun, [
                       2, 2], constraints=cons,
                       method='SLSQP', options={'disp': True})

bf.plot_fitbeta(df_beta)

# =============================================================================
#  histogram plot and show the missing data
# =============================================================================
# %%
StationData = pd.DataFrame()
# hourly
StationData = StationData_row.reindex(pd.date_range(StationData_row.index.floor('D').min(),
                                                    StationData_row.index.ceil('D').max(), freq='H'))[:-1]
# daily
#StationData = StationData.resample('D', label='right').sum()

StationData['active'], StationData['inactive'] = \
    (StationData.value >= 0).astype(int), - \
    (~(StationData.value >= 0)).astype(int)

plt.ioff()
f, (ax11, ax12) = plt.subplots(2, 1, sharex=True,
                               gridspec_kw={'height_ratios': [3, 1]})

ax11.plot(StationData.index, StationData.value, c='g')
ax11.set_ylabel('Amount', fontsize=14)
for (v, c) in [(1, 'b'), (0, 'r')]:
    ax12.scatter(StationData.index[StationData.active == v],
                 StationData.active[StationData.active == v], s=5, c=c)

ax12.set_ylim([-0.5, 1.5])
ax12.yaxis.set_visible(False)
ax11.set_title('Rainfall - hourly', fontsize=16)
plt.xlabel('Time', fontsize=14)
plt.tight_layout()
ax11.grid(True), ax12.grid(True)
plt.show()

# =============================================================================
#  log regression - maximum likelihood and plot   #無法完成 - 只有一站
# =============================================================================
# maximum likelihood - log
# remember the w2 is used not w1


def create_df_w_pcp(w_df, df_oben, df_unen):
    df = pd.DataFrame(index=w_df.index,
                      data=w_df.values, columns=['percent'])
    df['amount_unten'] = df_unen.loc[w_df.index, :]
    df['amount_oben'] = df_oben.loc[w_df.index + (
        df_unen.index[1] -
        df_unen.index[0]), :].values
    return df


# %%(1)----------------
df_1 = create_df_w_pcp(W2_2er, S_30min, S_15min)
# df = df[(df.percent <= 1) & (df.percent >= 0)]


def makesure(df):
    df[df.amount_oben < 0] = 0
    return df


df = makesure(df_1)
# Optimization
# 分成2部分   一個是P01， 一個是1 - P01


def obj_logfun(x, sign=1.0):

    df_logRegress_01 = df[(df.percent == 0) | (
        df.percent == 1)].amount_oben.copy()
    df_logRegress_inner = df[(df.percent < 1) & (
        df.percent > 0)].amount_oben.copy()

    Z01_01 = x[0] + x[1] * np.log(df_logRegress_01)
    Z01_inner = x[0] + x[1] * np.log(df_logRegress_inner)

    summa1_01 = - np.log(
        1 - 1 / (1 + np.exp(-Z01_01))
    ).sum()  # min sum
    summa1_inner = - np.log(
        1 / (1 + np.exp(-Z01_inner))
    ).sum()  # min sum

    return summa1_01 + summa1_inner


#cons =((-5,5), (-5,5))
result_log = minimize(obj_logfun, [2, 2]  # , bounds = cons
                      , method='SLSQP', options={'disp': True})


def logfun(x, a, b):
    Z01 = a + b * np.log(x)
    return 1 - 1 / (1 + np.exp(-Z01))


#(2)----------------
increment = 0.02
min_number = 30

# close to left
intervals = np.arange(df.amount_oben.min(),
                      df.amount_oben.max(),
                      (df.amount_oben.max() - df.amount_oben.min()
                       ) * increment
                      )

df['categories'] = pd.cut(df.amount_oben  # .amount_oben.sort_values(ascending=True)
                          , bins=intervals, labels=False)

# check if the number is enough
# if not enough, then just ignore it
# calculate the P01 in each interval (by classified label)
# y = P01, the x = mean of R of each interval


def interval_P01(data):
    P01_list = []

    for i in np.arange(0, data.categories.max() + 1):
        dfs = data[data.categories == i]
        if len(dfs) >= min_number:
            P01 = len(dfs[(dfs.percent == 0) | (dfs.percent == 1)]) / len(dfs)
        else:
            pass

        P01_list.append(P01)

    return P01_list


P01_class = interval_P01(df)


def meanR_class(data):
    meanR_list = []

    for i in np.arange(0, data.categories.max() + 1):
        dfs = data[data.categories == i]
        if len(dfs) >= min_number:
            meanR_list.append(dfs.amount_oben.mean())
        else:
            meanR_list.append(np.nan)

    return meanR_list


meanR = meanR_class(df)

# plot it
fig = plt.figure(6)

x_value = df.amount_oben.sort_values(ascending=True)
plt.plot(np.log(x_value), logfun(x_value, result_log.x[0], result_log.x[1]))
#,color='blue') #, normed=True)#, zorder=0)
plt.scatter(np.log(meanR), P01_class, c='r', zorder=1)

plt.title('L = 1, dt = 30mins', fontsize=20)
plt.xlabel(r'$Log_1$' + r'$_0$' + r'$R$', fontsize=15)
plt.ylabel(r'$P_0$' + r'$_1$', fontsize=15)
plt.grid(True)

plt.show()

# %% use the parameters to build the models


class simulatemodeing:

    threshold = 0.3
    # building and select data

    def datamodel_BC(self, w_layer, data_oben, data_unten):
        df = pd.concat([w_layer, data_unten.iloc[np.arange(0,
                                                           len(data_oben)) * 2], data_oben.value], axis=1)
        # choose the data with constraint ( > threshold , 0 < w < 1)
        # the amount unten hier beduetet zweite.
        df.columns = ['P01', 'amount_unten', 'amount_oben']
        df = df[df.amount_oben >= threshold]
        df = df[(df.P01 <= 1) & (df.P01 >= 0)]
        return df

    def datamodel_DP(self, data_oben, data_unten):
        df = pd.concat(
            [data_unten.iloc[np.arange(len(data_oben)) * 2 + 1], data_oben.value], axis=1)
        # choose the data with constraint ( > threshold , 0 < w < 1)
        # the amount unten hier beduetet zweite.
        df.columns = ['amount_unten', 'amount_oben']
        df = df[df.amount_oben >= threshold]
        # calculate P01 with parameters a and b caculated before
        Z01 = result_log.x[0] + result_log.x[1] * np.log(df.amount_oben)
        df['P01'] = 1 - (1 / (1 + np.exp(-Z01)))
        return df

    def makesure(self, datafr):
        datafr[datafr.amount_oben < 0] = 0
        return datafr

    def df_RV(self, datafr):
        datafr['RV'] = np.random.rand(len(datafr), 1)
        return datafr

    def W_simulation(self, datafr):
        # the R_simulation there is R2
        datafr['W2_simu'] = datafr.RV.copy()

        for i in np.arange(0, len(datafr)):
            RV2 = 0
            if datafr.P01[i] >= datafr.RV[i]:
                # generate RV2
                RV2 = np.random.rand(1)

                if RV2[0] >= 0.5:
                    datafr.W2_simu[i] = 1
                # elif RV2[0] < 0.5:
                else:
                    datafr.W2_simu[i] = 0
                del RV2

            # elif df2.P01[i] < df2.RV[i]:
            #    df2.R2_simu[i] = df2.RV[i]
            else:
                pass

        return datafr

    def R_simulation(self, datafr):
        datafr['R2'] = datafr.W2_simu * datafr.amount_oben
        datafr['R1'] = (1 - datafr.W2_simu) * datafr.amount_oben
        return datafr

  # basic model
cas = simulatemodeing()
df_BCmodel = cas.datamodel_BC(W2_1er, S_60min, S_30min)
df_BCmodel = cas.makesure(df_BCmodel)
df_BCmodel = cas.df_RV(df_BCmodel)
df_BCmodel = cas.W_simulation(df_BCmodel)
df_BCmodel = cas.R_simulation(df_BCmodel)

##
# choose 2016 autumn season (789)
##
df_BCmodel = df_BCmodel[df_BCmodel.index.year == 2016]
# df_BCmodel = df_BCmodel[(df_BCmodel.index.month == 7) |
#                        (df_BCmodel.index.month == 8) |
#                         (df_BCmodel.index.month == 9)]

fig = plt.figure(7)
plt.plot(  # df_model.index, df_model.amount_oben, 'b--'
    df_BCmodel.index, df_BCmodel.amount_unten, 'r*', df_BCmodel.index, df_BCmodel.R1, 'gx')
plt.title('R2', fontsize=20)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Rainfall (mm)', fontsize=15)
plt.legend(['Historical Data', 'Basic Model'])
plt.grid(True)
plt.show()

# dependent model
df_DPmodel = cas.datamodel_DP(S_60min, S_30min)
df_DPmodel = cas.makesure(df_DPmodel)
df_DPmodel = cas.df_RV(df_DPmodel)
df_DPmodel = cas.W_simulation(df_DPmodel)
df_DPmodel = cas.R_simulation(df_DPmodel)

##
# choose 2016 autumn season (789)
##
df_DPmodel = df_DPmodel[df_DPmodel.index.year == 2016]
# df_DPmodel = df_DPmodel[(df_DPmodel.index.month == 7) |
#                        (df_DPmodel.index.month == 8) |
#                         (df_DPmodel.index.month == 9)]

# plot
fig = plt.figure(8)
plt.plot(  # df_model.index, df_model.amount_oben, 'b--'
    df_DPmodel.index, df_DPmodel.amount_unten, 'r*', df_DPmodel.index, df_DPmodel.R1, 'gx')
plt.title('R2', fontsize=20)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Rainfall (mm)', fontsize=15)
plt.legend(['Historical Data', 'Dependent Model'])
plt.grid(True)
plt.show()


fig = plt.figure(9)
plt.plot(df_DPmodel.index, df_DPmodel.amount_unten, 'r*', df_BCmodel.index,
         df_BCmodel.R1, 'bx', df_DPmodel.index, df_DPmodel.R1, 'go')
plt.title('R2', fontsize=20)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Rainfall (mm)', fontsize=15)
plt.legend(['Historical Data', 'Basic Model', 'Dependent Model'])
plt.grid(True)
plt.show()

# Error figure
fig = plt.figure(10)
plt.plot(df_DPmodel.index, df_DPmodel.amount_unten - df_BCmodel.R1,
         'r*', df_BCmodel.index, df_DPmodel.amount_unten - df_DPmodel.R1, 'bx')
plt.title('Basic Model v.s. Dependent Model', fontsize=20)
plt.xlabel('Time', fontsize=15)
plt.ylabel('Error (mm)', fontsize=15)
plt.legend(['Basic Model', 'Dependent Model'])
plt.grid(True)
plt.show()

# sum of absolute error
#sum(abs(df_DPmodel.amount_unten - df_DPmodel.R1))
#sum(abs(df_DPmodel.amount_unten - df_BCmodel.R1))
# %%
STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))


# lorenz curves
