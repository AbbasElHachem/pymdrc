# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas, IWS
"""
from networkx.algorithms import threshold

'''
This Script is for a Multiplicative Discrete Random Cascade Model (MDRC)

Temporal Rainfall Disaggregation 
'''


import os
import timeit
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import optimize

from scipy import stats
from scipy.stats import beta as beta_dist_fct
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

    def plot_station_data(self, df_to_plot, temp_agg):
        StationData = df_to_plot.copy(deep=True)

        # hourly

        if temp_agg < '600min':
            StationData = StationData_row.reindex(
                pd.date_range(StationData_row.index.floor('D').min(),
                              StationData_row.index.ceil('D').max(),
                              freq='H'))[:-1]

        StationData['active'], StationData['inactive'] = \
            (StationData.values >= 0).astype(int), - \
            (~(StationData.values >= 0)).astype(int)

        plt.ioff()
        _, (ax11, ax12) = plt.subplots(
            2, 1, figsize=(12, 8), sharex=True,
            gridspec_kw={'height_ratios': [3, 1]})

        ax11.plot(StationData.index,
                  StationData.iloc[:, 0].values, c='g')
        ax11.set_ylabel('Pcp [mm/%s]' % temp_agg, fontsize=14)
        for (v, c) in [(1, 'b'), (0, 'r')]:
            ax12.scatter(
                StationData.index[StationData.active == v],
                StationData.active[StationData.active == v],
                s=5, c=c)

        ax12.set_ylabel('Acitve / Inactive', fontsize=12)
        ax12.set_ylim([-0.5, 1.5])
        ax12.yaxis.set_visible(False)
        ax11.set_title('Rainfall - %s' % temp_agg,
                       fontsize=16)
        plt.xlabel('Time', fontsize=14)
        plt.tight_layout()
        ax11.grid(True), ax12.grid(True)
        plt.savefig('station_data_%s.png' % temp_agg)
        plt.close()
        return

    def create_agg_levels(self):
        '''
        go up cascade levels using a branching number of 2
        untill desired number of cascades is reached
        '''
        df = self.df
        cascade_levels = self.cascade_levels
        input_agg = self.find_init_temp_agg(df)
        upper_level = int(input_agg)
        agg_levels = [upper_level]
        for _ in range(1, cascade_levels + 1):
            upper_level = int(2 * (upper_level))
            agg_levels.append(upper_level)
            # print(upper_level)
        agg_levels_top_down = agg_levels[::-1]
        return agg_levels_top_down

    def sum_check(self, df1, df2):
        if abs(df1.values.sum() - df2.values.sum()) <= 10**-4:
            pass
        else:
            print('error')
        return

    def resampleDf(self, agg,
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
        df = self.df

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

        self.sum_check(df_agg, df)

        return df_agg

    def calc_weights_level(self, df_oben, df_unten, pcp_thr):
        ''' find weights per level'''
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
        return w1, w2

    def mean_w_month(self, weights_level):
        '''find mean w values per month '''
        cats = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        months = pd.Categorical(weights_level.index.strftime(
            '%b'), categories=cats, ordered=True)
        W_sort = weights_level.groupby(
            [months, weights_level.index.time]).mean().unstack()
        return W_sort.mean(axis=1)

    def valueP01(self, weights_level):
        ''' calculate P01 per level 

            this values reflects the probability of rainfall
            from upper level falling completely in one of the
            sub-intervals in the lower level
        '''
        p0 = (weights_level[weights_level == 0].dropna().size /
              weights_level.size)
        p1 = (weights_level[weights_level == 1].dropna().size /
              weights_level.size)

        p01 = p0 + p1
        return p01

    def valueP01_monthly(self, weights_level):
        ''' group P01 for every month seperately
        this incorporates the yearly variation
         '''
        P01_month_list = []
        for i in range(0, 12):
            w_mon = weights_level.loc[
                (weights_level.index.month == (i + 1))]
            P01 = len(w_mon[(w_mon == 0) | (w_mon == 1)
                            ].dropna()) / len(w_mon)
            P01_month_list.append(P01)

        return P01_month_list

    def plot_P01_monthly(self, P01_month,
                         agg_upper_level,
                         agg_lower_level):
        '''scatter plot P01 for every month '''
        # plot w = 0 | w = 1
        plt.ioff()
        plt.figure(figsize=(12, 8))
        plt.scatter(np.arange(1, 12 + 1), P01_month)
        plt.title(r'$P01$' + '(Monthly)' +
                  '%s - %s' % (agg_upper_level, agg_lower_level),
                  fontsize=20)
        plt.xlabel('Month', fontsize=15)
        plt.ylabel('P01', fontsize=15)
        plt.xlim((0.5, 12.5))
        plt.ylim((0, 1))
        plt.legend(['Upper agg:%s' % agg_upper_level,
                    'Lower agg:%s' % agg_lower_level])
        plt.grid(True)
        plt.savefig('P01_monthly_%s_%s.png'
                    % (agg_upper_level, agg_lower_level))
        plt.close()

        return

    def W_innerhalb(self, weights_level, df_oben, pcp_thr,
                    agg_upper_level, agg_lower_level):
        ''' Scatter plot of the W1 weights per level'''
        df_oben_abv_thr = df_oben[df_oben >= pcp_thr].dropna()
        # assert len(df_oben_abv_thr.index) == len(W2_1er.index)
        shift_freq = (df_oben.index[1] - df_oben.index[0]) / 2
        idx_weights_in_01 = weights_level[
            (weights_level > 0) & (weights_level < 1)].dropna().index
        idx_pcp_for_w = idx_weights_in_01 + shift_freq
        w_in = weights_level[
            (weights_level > 0) & (weights_level < 1)].dropna()
        pcp_abv = df_oben_abv_thr.loc[idx_pcp_for_w, :].dropna()
        plt.ioff()
        plt.figure(figsize=(12, 8))
        plt.plot(w_in.values, pcp_abv.values, 'ro')
        plt.xlabel('W value')
        plt.ylabel('Pcp [mm/%s]' % agg_upper_level)
        plt.grid()
        plt.savefig('W1_level_%s_%s.png'
                    % (agg_upper_level, agg_lower_level))
        plt.close()
        return

    def create_df_w_pcp(self, w_df, df_oben, df_unten):
        df = pd.DataFrame(index=w_df.index,
                          data=w_df.values, columns=['percent'])
        df['amount_unten'] = df_unten.loc[w_df.index, :]
        df['amount_oben'] = df_oben.loc[w_df.index + (
            df_unten.index[1] -
            df_unten.index[0]), :].values

        return df

    def makesure(self, df):
        df[df.amount_oben < 0] = 0
        return df

    def obj_logfun(self, x):

        df_logRegress_01 = df[(df.percent == 0) | (
            df.percent == 1)].amount_oben.copy()
        df_logRegress_inner = df[(df.percent < 1) & (
            df.percent > 0)].amount_oben.copy()

        Z01_01 = x[0] + x[1] * np.log(df_logRegress_01)
        Z01_inner = x[0] + x[1] * np.log(df_logRegress_inner)

        summa1_01 = - np.log(1 - 1 / (1 + np.exp(-Z01_01))).sum()
        summa1_inner = - np.log(1 / (1 + np.exp(-Z01_inner))).sum()

        return summa1_01 + summa1_inner

    def logfun(self, x, a, b):
        Z01 = a + b * np.log(x)
        return 1 - 1 / (1 + np.exp(-Z01))

    def create_intervals_p01(self, df):
        intervals = np.arange(df.amount_oben.min(),
                              df.amount_oben.max(),
                              (df.amount_oben.max() -
                               df.amount_oben.min()
                               ) * increment)

        df['categories'] = pd.cut(df.amount_oben,
                                  bins=intervals, labels=False)

        return df

    def interval_P01(self, df, min_number=30):
        P01_list = []

        for i in np.arange(0, df.categories.max() + 1):
            dfs = df[df.categories == i]
            if len(dfs) >= min_number:
                P01 = len(dfs[(dfs.percent == 0) |
                              (dfs.percent == 1)]) / len(dfs)
            else:
                pass

            P01_list.append(P01)

        return P01_list

    def meanR_class(self, df, min_number=30):
        meanR_list = []

        for i in np.arange(0, df.categories.max() + 1):
            dfs = df[df.categories == i]
            if len(dfs) >= min_number:
                meanR_list.append(dfs.amount_oben.mean())
            else:
                meanR_list.append(np.nan)

        return meanR_list

    def plot_log_reg_P01(self, df, agg_upper_level, agg_lower_level):
        # plot it
        plt.figure(figsize=(12, 8))

        x_value = df.amount_oben.sort_values(ascending=True)
        plt.plot(np.log(x_value), self.logfun(
            x_value, result_log.x[0], result_log.x[1]),
            label='Model')
        #,color='blue') #, normed=True)#, zorder=0)
        plt.scatter(np.log(meanR), P01_class, c='r', zorder=1,
                    label='Obsv')

        plt.title('LogRegr, %s-%s' % (agg_upper_level,
                                      agg_lower_level), fontsize=20)
        plt.xlabel(r'$Log_1$' + r'$_0$' + r'$R$', fontsize=15)
        plt.ylabel(r'$P_0$' + r'$_1$', fontsize=15)
        plt.grid(True)
        plt.legend(loc=0)
        plt.savefig('P01_LogReg_%s_%s.png'
                    % (agg_upper_level, agg_lower_level))
        plt.close()
#=======================================================================
#
#=======================================================================


class betafit:

    def dfcreate(self, W2_1er):

        w_in = W2_1er[(W2_1er > 0) & (W2_1er < 1)].dropna()
        # pcp_abv = df_oben_abv_thr.loc[idx_pcp_for_w, :].dropna()
        df = pd.DataFrame(index=w_in.index)
        df['percent'] = w_in.values

        return df
    # Optimization
    # df_beta = self.dfcreate()

    def obj_logbetafun(self, x):

        df_beta_ob = df_beta.percent.copy()
        summa = - np.log(
            beta_dist_fct.pdf(
                df_beta_ob, x[0], x[1])).sum()  # min sum
        return summa

    def betafun(self, x, a, b):
        return (gammaf(a + b) / gammaf(a) / gammaf(b) *
                x**(a - 1) * (1 - x)**(b - 1))

    def plot_fitbeta(self, df, agg_upper_level, agg_lower_level):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.scatter(df.percent, bf.betafun(
            df.percent, result_beta.x[0], result_beta.x[1]),
            c='r', zorder=1)

        #bins_number = 50
        #bins = np.arange(0 + 1/bins_number , 1, 1/bins_number)
        ax.hist(df.percent, bins=50, color='blue',
                align='mid',
                density=True, zorder=0)
        #bins_labels(bins, fontsize=10)

        plt.title(r'$\beta$ = %.2f'
                  r' %s-%s' % (result_beta.x[0],
                               agg_upper_level, agg_lower_level),
                  fontsize=14)
        ax.text(0.7, 2.2, r'$\beta$ = %.2f' %
                result_beta.x[0], fontsize=15)

        plt.xlabel('W values ]0, 1[', fontsize=16)
        plt.ylabel('Probability density function', fontsize=16)
        plt.grid(True, alpha=0.5)
        plt.savefig('W_beta_fct_%s_%s.png'
                    % (agg_upper_level, agg_lower_level))
        plt.close()

#=======================================================================
#
#=======================================================================


class simulatemodeing:

    def valueP01_monthly(self, w_df):

        month_list = []
        for i in range(0, 12):
            w_mon = w_df.loc[(w_df.index.month == (i + 1))]
            P01 = len(w_mon[(w_mon == 0) | (w_mon == 1)
                            ].dropna()) / len(w_mon)
            month_list.append(P01)

        return month_list

    def datamodel_BC(self, w_df, df_oben, df_unen, threshold):
        df_oben = df_oben[df_oben >= threshold]
        df = pd.DataFrame(index=w_df.index,
                          data=w_df.values, columns=['percent'])
        df['amount_unten'] = df_unen.loc[w_df.index, :]
        df['amount_oben'] = df_oben.loc[w_df.index + (
            df_unen.index[1] -
            df_unen.index[0]), :].values
        df = df[df.amount_oben >= threshold]
        df = df[(df.percent <= 1) & (df.percent >= 0)]
        return df

    def model_BC_P01(self, df_data_BC):
        P01_val_month = self.valueP01_monthly(df_data_BC.percent)
        for month in range(0, 12):
            idx_month = df_data_BC.loc[df_data_BC.index.month ==
                                       month + 1, :].index
            df_data_BC.loc[idx_month, 'P01'] = P01_val_month[month]
        return df_data_BC

    def datamodel_DP(self, data_oben, data_unten):
        df = pd.concat(
            [data_unten.iloc[np.arange(0,
                                       len(data_oben)) * 2],
             data_oben.value], axis=1)
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

    def df_RV2(self, datafr):
        datafr['RV2'] = np.random.rand(len(datafr), 1)
        return datafr

    def assign_W01_P01(self, datafr):
        # these values will be either 0 or 1
        idx_p01 = np.where(datafr.RV <= datafr.P01)[0]
        df_BCmodel_01 = datafr.iloc[idx_p01, :]
        idx_w0 = df_BCmodel_01.iloc[
            np.where(df_BCmodel_01.RV2 <= 0.5)[0], :].index
        idx_w1 = df_BCmodel_01.iloc[
            np.where(df_BCmodel_01.RV2 > 0.5)[0], :].index

        datafr.loc[idx_w0, 'W0'] = 0
        datafr.loc[idx_w0, 'W1'] = 1

        datafr.loc[idx_w1, 'W0'] = 1
        datafr.loc[idx_w1, 'W1'] = 0
        return datafr

    def assign_W01_beta(self, datafr):

        idx_w01 = datafr.iloc[
            np.where(datafr.RV > datafr.P01)[0], :].index
        datafr.loc[idx_w01, 'W0'] = np.random.beta(
            result_beta.x[0], result_beta.x[1],
            size=len(idx_w01))

        datafr.loc[idx_w01, 'W1'] = (
            1 - datafr.loc[idx_w01, 'W0'])
        return datafr

    def R_simulation(self, datafr):

        datafr = self.assign_W01_P01(datafr)
        datafr = self.assign_W01_beta(datafr)
        df_final = pd.DataFrame(index=datafr.index)
        df_final['R1'] = datafr.loc[
            :, 'amount_oben'] * datafr.loc[:, 'W0']
        df_final['R2'] = datafr.loc[
            :, 'amount_oben'] * datafr.loc[:, 'W1']
        return df_final


#=======================================================================
#
#=======================================================================
if __name__ == '__main__':
    print('#### Started on %s ####\n' % time.asctime())
    START = timeit.default_timer()

    main_dir = os.path.join(
        r'X:\staff\elhachem\GitHub\pymdrc\test_data_results')
    os.chdir(main_dir)

    StationPath = os.path.join(main_dir, r'P00003_60min_data_1995_2011.csv')
    assert os.path.exists(StationPath)

    # read df
    StationData_row = pd.read_csv(StationPath, sep=';',
                                  index_col=0,
                                  engine='c')

    StationData_row.index = pd.to_datetime(
        StationData_row.index, format="%Y-%m-%d %H:%M:%S")

    cascade_levels = 5

    main_class = CascadeModel(
        df=StationData_row, cascade_levels=cascade_levels)

    ll = main_class.create_agg_levels()

    trace_rainfall = 0.3

    initial_beta_vls = [2.2, 2.2]  # if symmetric beta
    beta_bounds = [(2, 5.), (2, 5.)]

    # create dataframes for all aggregation frequencies
    cons = {'type': 'eq',
            'fun': lambda x: np.array(x[0] - x[1])}

    # change this to improve fit of logRegression (2params a, b)

    increment = 0.02
    min_number = 30

    a_b_intial_vls = [-1., 1.9]
    a_b_params_bounds = [(None, None), (1.8, None)]

    for _agg in ll:
        upper_level_agg = str(_agg) + 'Min'
        lower_level_agg = str(int(_agg / 2)) + 'Min'
        # 60-120-240-480-720-1440
        df_upper_level = main_class.resampleDf(upper_level_agg)
        df_lower_level = main_class.resampleDf(lower_level_agg)

        main_class.plot_station_data(df_to_plot=df_upper_level,
                                     temp_agg=upper_level_agg)

        w1, w2 = main_class.calc_weights_level(
            df_oben=df_upper_level,
            df_unten=df_lower_level,
            pcp_thr=trace_rainfall)

        P01_level = main_class.valueP01(weights_level=w1)

        P01_level_monthly = main_class.valueP01_monthly(weights_level=w1)

        main_class.plot_P01_monthly(P01_month=P01_level_monthly,
                                    agg_upper_level=upper_level_agg,
                                    agg_lower_level=lower_level_agg)
        main_class.W_innerhalb(weights_level=w1,
                               df_oben=df_upper_level,
                               pcp_thr=trace_rainfall,
                               agg_upper_level=upper_level_agg,
                               agg_lower_level=lower_level_agg)

        # b1, b2 = main_class.fitBetaWeights(weights_level=w1)

        # main_class.plot_fitbeta(w1)
        bf = betafit()
        df_beta = bf.dfcreate(w1)
        #df_beta = bf.dfcreate(W2_2er, S_30min ,S_15min)

        result_beta = minimize(bf.obj_logbetafun,
                               initial_beta_vls,
                               constraints=cons,
                               method='SLSQP',
                               options={'disp': True})

        bf.plot_fitbeta(df_beta,
                        agg_upper_level=upper_level_agg,
                        agg_lower_level=lower_level_agg)

        df = main_class.create_df_w_pcp(w1, df_oben=df_upper_level,
                                        df_unten=df_lower_level)
        df = main_class.makesure(df)

        result_log = minimize(
            main_class.obj_logfun,
            [2, 2],
            method='SLSQP', options={'disp': True})

        df_incr_p01 = main_class.create_intervals_p01(df)
        P01_class = main_class.interval_P01(df)
        meanR = main_class.meanR_class(df)

        main_class.plot_log_reg_P01(
            df,
            agg_upper_level=upper_level_agg,
            agg_lower_level=lower_level_agg)

        cas = simulatemodeing()
        df_BCmodel = cas.datamodel_BC(w1, df_upper_level,
                                      df_lower_level,
                                      threshold=trace_rainfall)
        df_BCmodel = cas.makesure(df_BCmodel)
        df_BCmodel = cas.model_BC_P01(df_BCmodel)
        df_BCmodel = cas.df_RV(df_BCmodel)
        df_BCmodel = cas.df_RV2(df_BCmodel)
        df_BCmodel = cas.R_simulation(df_BCmodel)
        break
    STOP = timeit.default_timer()
    print(('\n#### Done with everything on %s.\nTotal run time was'
           ' about %0.4f seconds ####' % (time.asctime(), STOP - START)))
