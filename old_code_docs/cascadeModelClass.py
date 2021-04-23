# -*- coding: utf-8 -*-
"""
Finalized on % 01.06.2018

@author: EL Hachem Abbas, IWS
Institut fuer Wasser- und Umweltsystemmodellierung - IWS
"""
from collections import Counter
import os
import time
import timeit

from scipy import optimize
from scipy import stats

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


plt.ioff()

'''
This Script is for a Multiplicative Discrete Random Cascade Model (MDRC)

First read rainfall data on a fine resolution (5min or less)
and aggregate the data to higher time frequencies:
(Upper Level: 60min; Middle Level: 30min; Lower Level: 15min).

Second disaggregating the data through a mutiplicative Discrete Random Cascade
model (MDRC) (60min->30min->15min), at the end, rainfall on a finer resolution
will be simulated using a higher time frequency data.

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
and using the maximum likelihood method the parameter ÃŸ is estimated
for every level and every station.
The MDRC has two parameters P01 and ÃŸ per level.

Simulating:
once parameters for every level and every station have been estimated
use the parameters to simulate for every rainfall value in cascade level zero
above the threshold a two values for the corresponding sub intervals,
using randmoly generated weigths based on P01 value
and Beta distreibution function, find R1=W1*R0 and R2=(1-W1)*R0
create a new dataframe for the output,
this is done for every model and every level and station, define the wanted
number of simulations (each simulation is a timeseries)

Lorenz Curves
read the results of one simulation for the different stations and cascade level
calculate the Lorenz curves for the observed and simulated data

plot the results in a different script (plotting_results_cascade_model.py)

Litterature for better understanding everything:

Mcintyre, Neil & BÃ¡rdossy, AndrÃ¡s. (2017).
Using Probable Maximum Precipitation to Bound the Disaggregation of Rainfall.
Water. 9. 496. 10.3390/w9070496.

EL Hachem, Abbas. (2018).
Application of a Cascade Model for Temporal Rainfall Disaggregation.
10.13140/RG.2.2.15707.26404.
https://www.researchgate.net/project/
Application-of-a-Cascade-Model-for-Temporal-Rainfall-Disaggregation
'''


class CascadeModel:

    def readDf(self, df_file, sep, date_fmt, idx_col, divide_factor, r_max):
        ''' read on df with defined sep and date format and idx  col'''
        df_out = pd.read_csv(df_file, sep=sep, index_col=idx_col,
                             encoding='utf-8') / divide_factor
        if date_fmt is not None:
            try:
                df_out.index = pd.to_datetime(df_out.index, format=date_fmt)
            except Exception as msg:
                print(msg)
                df_out.index = pd.to_datetime(df_out.index, format=date_fmt,
                                              errors='coerce')

        df_out = df_out[df_out < r_max]
        return df_out

    def resampleDf(self, data_frame, temp_freq, temp_shift, out_save_dir,
                   df_sep_, fillnan=False, df_save_name=None):
        ''' sample DF based on freq and time shift '''

        df_ = data_frame.copy()
        df_res = df_.resample(temp_freq, label='right', closed='right',
                              base=temp_shift).sum()
        if fillnan:
            df_res.fillna(value=0, inplace=True)

        if df_save_name is not None:
            df_res.to_csv(os.path.join(out_save_dir, df_save_name),
                          sep=df_sep_)
        return df_res

    def createOutDir(self, out_dir_all, dir_name_str):
        '''create out dir, using dir_name, folder_name, cascade_level'''
        out_dir_ = os.path.join(out_dir_all, dir_name_str)
        if not os.path.exists(out_dir_):
            os.mkdir(out_dir_)
        return out_dir_

    def neglectedData(self, df_level_1, R_threshhold, out_save_dir,
                      df_sep_, save_df_output=True):
        ''' input: ppt df and threhhold
            output: count df vals b4 and after use of threshhold
        '''
        # create dicts to hold output
        stn_values_count_ratio = {k: [] for k in df_level_1.columns}

        # initial values count per station
        for station in df_level_1.columns:
            df_stn = df_level_1[station].dropna(axis=0)
            df_abv_thr = df_stn[df_stn > R_threshhold]
            stn_values_count_ratio[station].append(len(df_stn.values))
            stn_values_count_ratio[station].append(len(df_abv_thr.values))

            stn_values_count_ratio[station].append(
                np.round(100 * (len(df_abv_thr.values) /
                                len(df_stn.values)), 2))

        # save dict to data frame and save results
        df_output = pd.DataFrame.from_dict(stn_values_count_ratio,
                                           orient='index')
        # adjust columns name
        df_output.rename(columns={0: 'df all values',
                                  1: 'df abv thr',
                                  2: 'ratio abv thr'}, inplace=True)

        if save_df_output:
            # extract time stamp of df, used when saving df
            df_time_frequency = df_level_1.index.freqstr

            # save df
            df_output.to_csv(os.path.join(out_save_dir,
                                          'df_%s_vls_abv__thr_of_%0.2f_mm_.csv'
                                          % df_time_frequency, R_threshhold),
                             sep=df_sep_)
        print('done calculating remaning data after use of threshold')

        return df_output

    def calculateWeightsL1(self, df_L0, df_L1,
                           R_threshhold, time_delta_L1):
        ''' fct to calculate and save weights cascade level one
            done for every station appart, since data differs '''
        out_save_dir = self.createOutDir(out_dir, r'%s' % cascade_level_1)
        weights_stn_dictL1 = {stn: [] for stn in df_L0.columns}
        dict_dfs_ws = {stn: [] for stn in df_L0.columns}
        for station in df_L0.columns:

            # drop nan from station values level 1
            df_abv_thr = df_L0[station][df_L0[station] > R_threshhold]
            orig_vals = df_abv_thr.dropna(axis=0).values.ravel()

            idx_intv_one = df_abv_thr.index - \
                pd.Timedelta(seconds=time_delta_L1)

            vals_intv_one = df_L1[station].loc[idx_intv_one].values.ravel()
            vals_intv_two = df_L1[station].loc[df_abv_thr.index].values.ravel()

            weights_intv_one = vals_intv_one / orig_vals
            weights_intv_two = vals_intv_two / orig_vals
            assert weights_intv_one.any() + weights_intv_two.any() == 1

            df_w_vals_L1 = pd.DataFrame(
                data=np.round(np.vstack(weights_intv_one), 2),
                index=idx_intv_one, columns=[station + ' Sub Int Left'])
            df_w_vals_L1[station + ' Sub Int Right'] = np.round(
                np.vstack(weights_intv_two), 2)
            df_w_vals_L1[station + ' Original Volume'] = np.vstack(orig_vals)
            df_w_vals_L1.to_csv(os.path.join(out_save_dir,
                                             'weights_%s_df_%s.csv'
                                             % (cascade_level_1,
                                                station)),
                                sep=df_sep)
            weights_stn_dictL1[station] = weights_intv_one
            dict_dfs_ws[station] = df_w_vals_L1
        return weights_stn_dictL1, dict_dfs_ws

    def calculateWeightsL2(self, df_L1, df_L2,
                           R_threshhold, time_delta_L2):
        ''' fct to calculate and save weights cascade level two
            done for every station appart, since data differs '''
        out_save_dir = self.createOutDir(out_dir, r'%s' % cascade_level_2)
        weights_stn_dictL2 = {stn: [] for stn in df_L1.columns}
        dict_dfs_ws = {stn: [] for stn in df_L1.columns}
        for station in df_L1.columns:

            # drop nan from station values level 1
            df_abv_thr = df_L1[station][df_L1[station] > R_threshhold]
            df_abv_thr.dropna(axis=0, inplace=True)

            df_abv_thr_left = df_abv_thr[df_abv_thr.index.minute == 30]
            df_abv_thr_right = df_abv_thr[df_abv_thr.index.minute == 0]

            vals_intv_one_1 = df_L2[station].loc[df_abv_thr_left.index -
                                                 pd.Timedelta(
                                                     seconds=time_delta_L2)
                                                 ].values.ravel()

            vals_intv_one_2 = df_L2[station].loc[
                df_abv_thr_left.index].values.ravel()

            vals_intv_two_1 = df_L2[station].loc[df_abv_thr_right.index -
                                                 pd.Timedelta(
                                                     seconds=time_delta_L2)
                                                 ].values.ravel()
            vals_intv_two_2 = df_L2[station].loc[
                df_abv_thr_right.index].values.ravel()

            weights_intv_one_L2 = np.hstack((vals_intv_one_1 / df_abv_thr.loc[
                df_abv_thr_left.index].values.ravel(),
                vals_intv_two_1 / df_abv_thr.loc[
                df_abv_thr_right.index].values.ravel()))

            weights_intv_two_L2 = np.hstack((vals_intv_one_2 / df_abv_thr.loc[
                df_abv_thr_left.index].values.ravel(),
                vals_intv_two_2 / df_abv_thr.loc[
                df_abv_thr_right.index].values.ravel()))

            assert weights_intv_one_L2.any() + weights_intv_two_L2.any() == 1

            df_w_vals_L2 = pd.DataFrame(
                data=df_abv_thr.values,
                columns=[station + ' Original Volume'],
                index=df_abv_thr.index)
            df_w_vals_L2[station + ' Sub Int Right'] = np.round(
                np.vstack(weights_intv_two_L2), 2)
            df_w_vals_L2[station + ' Sub Int Left'] = np.round(np.vstack(
                weights_intv_one_L2), 2)

            df_w_vals_L2.to_csv(os.path.join(out_save_dir,
                                             'weights_%s_df_%s.csv' %
                                             (cascade_level_2, station)),
                                sep=df_sep)
            weights_stn_dictL2[station] = weights_intv_one_L2
            dict_dfs_ws[station] = df_w_vals_L2
        return weights_stn_dictL2, dict_dfs_ws

    def calculateP01fromWeights(self, dict_weights,
                                min_w_vals, cascade_level):
        '''
        Idea: find probability W=0 or W=1 from sampled Ws for each
              station and cascade level.
              MDRC parameter one --> P01
        input:  dict: stn: [weights]
                dict: stn: orig values
                int: min nbr of Ws to consider
        ouput:  df_p01
        '''
        df_p01_ = pd.DataFrame(index=dict_weights.keys())
        for stn_ in dict_weights.keys():
            weights_left = dict_weights[stn_]
            if len(weights_left[weights_left == 0]) >= min_w_vals:

                p0 = len(weights_left[weights_left == 0]) / len(weights_left)
            else:
                p0 = -999
            if len(weights_left[weights_left == 1]) >= min_w_vals:
                p1 = len(weights_left[weights_left == 1]) / len(weights_left)
            else:
                p1 = -999
            if (len(weights_left[weights_left == 0]) or
                    len(weights_left[weights_left == 1])) >= min_w_vals:
                p01 = len(weights_left[(weights_left == 0) | (
                    weights_left == 1)]) / len(weights_left)
            else:
                p01 = np.nan()
            assert np.isclose(p0 + p1, p01, 1e-4)
            df_p01_.loc[stn_, 'P0'] = p0
            df_p01_.loc[stn_, 'P1'] = p1
            df_p01_.loc[stn_, 'P01'] = p01
        df_p01_.to_csv(os.path.join(out_dir,
                                    'Prob W P01 %s.csv' %
                                    (cascade_level)),
                       sep=df_sep)

        return df_p01_

    def weightsSortMonth(self, dict_dfs_weights, cascade_level):
        ''' find weights for every month and calculate P0, P1, P01 per month'''
        dict_w_month = {m: [] for m in range(1, 13)}
        dict_p01_month = {stn: [] for stn in dict_dfs_weights.keys()}
        for stn in dict_dfs_weights.keys():
            df_p01_month = pd.DataFrame(index=dict_w_month.keys())
            w_left = dict_dfs_weights[stn]['%s Sub Int Left' % stn]
            [dict_w_month[month].append(w_left.loc[idx])
             for month in dict_w_month
             for idx in w_left.index if month == idx.month]
            for m in dict_w_month.keys():
                df_p01_month.loc[m, 'P0'] = len([k for k in dict_w_month[m]
                                                 if k == 0])\
                    / len(dict_w_month[m])
                df_p01_month.loc[m, 'P1'] = len([k for k in dict_w_month[m]
                                                 if k == 1])\
                    / len(dict_w_month[m])
                df_p01_month.loc[m, 'P01'] = len([k for k in dict_w_month[m]
                                                  if (k == 0 or k == 1)])\
                    / len(dict_w_month[m])
                df_p01_month.to_csv(os.path.join(out_dir,
                                                 '%s P1 P0 per month %s.csv'
                                                 % (stn, cascade_level)),
                                    sep=df_sep)
            dict_p01_month[stn].append(df_p01_month)
        return dict_p01_month

    def fitBetaWeights(self, dict_weights, cascade_level,
                       initial_parm, bounds, constrains):
        '''
        Idea: fit a symmetric beta fct to sampled weights
              MDRC parameter two: beta

        input: dict weights values
        output: df__fitted_parameters_result
        '''

        # define df to hold parameters result per station as ouput
        df_parameters_out = pd.DataFrame(index=dict_weights.keys())

        # call the beta distribution function from scipy
        beta_dist_fct = stats.beta

        # start going through each staion sampled weights
        for station in dict_weights.keys():

                # get values as np array
            values = np.array([dict_weights[station]])

            # select values 0< W <1 (beta_dist_fct:0<x<1)
            nonnull_w_values = values[(values != 0) & (values != 1)]

            if len(nonnull_w_values) >= min_nbr_vals:  # if enough Ws

                    # calculate log beta_pdf(w, alfa, beta)
                def _logpdf(x, alfa, _beta): return \
                    beta_dist_fct(a=alfa, b=_beta).logpdf(x)

                vec = np.vectorize(_logpdf)  # easier to work with these vals

                # def fct to calculate sum(Log(beta_pdf(x)))
                def unliklihood(scale):
                    alfa = scale[0]
                    _beta = scale[1]
                    e = vec(nonnull_w_values, alfa, _beta)
                    return -np.sum(e)  # negative bcz minimise not maximising

                # optimise fct to find parameters
                res = optimize.minimize(unliklihood,
                                        x0=initial_parm,  # intial param
                                        bounds=bounds,  # should be in bounds
                                        method='SLSQP',
                                        constraints=constrains,
                                        tol=1e-20)

                # save parameters to df_prameters_out
                df_parameters_out.loc[station, 'alfa'] = res.x[0]
                df_parameters_out.loc[station, 'beta'] = res.x[1]

            else:  # if not enough weights for a station
                df_parameters_out.loc[station, 'alfa'] = -999
                df_parameters_out.loc[station, 'beta'] = -999

            # save df
            df_parameters_out.to_csv(os.path.join(out_dir,
                                                  ('bounded maximum'
                                                   'likelihood %s.csv')
                                                  % cascade_level),
                                     sep=df_sep)
        print('done with fitting beta to weights, finding unbouded model next')
        return df_parameters_out

    def logRegression(self, r_vals, a, b):
        '''this fct is called when getting P01 for R'''
        return np.array([1 - 1 / (1 + np.exp(
            -(np.array([a + b * np.log10(r_vals)]))))])

    def build_likelihood_cols(self, df_likelihood, a, b):
        ''' if W01=0 use log(f(R)) else use log(1-f(R))
             fct introduced to calculate likelihood '''

        # go through W01 vals and calculate cols of likelihood
        for idx, r_val, w_val_ in zip(df_likelihood.index,
                                      df_likelihood['R vals'].values,
                                      df_likelihood['W 01'].values):

            # if W=0 or W=1 usef log(f(precipitation))
            if w_val_ == 0.:
                df_likelihood.loc[idx, 'log(L(teta))'] = \
                    (np.log(self.logRegression(r_val, a, b)))

            # if 0<W<1 use log(1-f(precipitation))
            elif w_val_ == 1.:
                df_likelihood.loc[idx, 'log(L(teta))'] = \
                    np.log((1 - self.logRegression(r_val, a, b)))

        # return the values of log(likelihood)
        values = df_likelihood['log(L(teta))'].values
        return values

    def unliklihood2(self, scale):
        '''what to minimise, cal fct on vals and
            optmise it to find params of fct'''
        a1 = scale[0]
        b1 = scale[1]
        # will return the values log(likelihood), minimise the sum
        e = self.build_likelihood_cols(df_likelihood, a1, b1)
        return -np.sum((e))

    def volumeDependacy(self, in_df_w_stn_dict, cascade_level,
                        initial_vls_a_b, bounds_a_b):
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
        global df_likelihood, res
        # define out_dir for df_values_out of log Likelihood fct
        out_dir_unboud_model = self.createOutDir(out_dir,
                                                 r'%s P01 volume dependancy'
                                                 % cascade_level)

        # define out_dir for df_likelihood_params result
        out_dir_level_one_params = self.createOutDir(out_dir_unboud_model,
                                                     r'%s log_regress params'
                                                     % cascade_level)

        # new df to hold results of likelihood of logRegression params a and b
        df_dependancy = pd.DataFrame(index=in_df_w_stn_dict.keys())

        # perform for every station
        for stn in in_df_w_stn_dict.keys():

            # get stn values, bcz saved as a list in dfs_L1 = [df1, df2, ...]
            station_df = in_df_w_stn_dict[stn]

            # select R vals and weights (W1)
            R_vals = station_df['%s Original Volume' % stn]
            w1_vals = station_df['%s Sub Int Left' % stn]

            # new df to hold calculation of likelihood
            df_likelihood = pd.DataFrame(dtype=float)

            # add R vals and W vals to new df_likelihood
            df_likelihood['R vals'] = R_vals
            df_likelihood['W1 vals'] = w1_vals

            # go through W vals, if 0 or 1 relace with 0 in new col W01
            for idx, w_val in zip(df_likelihood.index,
                                  df_likelihood['W1 vals'].values):
                if w_val == 0.0 or w_val == 1.0:
                    df_likelihood.loc[idx, 'W 01'] = 0.
                else:
                    df_likelihood.loc[idx, 'W 01'] = 1.

            # result of optimisation
            res = optimize.minimize(self.unliklihood2,
                                    x0=initial_vls_a_b,
                                    method='SLSQP',
                                    bounds=bounds_a_b,
                                    tol=1e-15,
                                    options={'maxiter': 10000.})

            # save results to df, will be used for plotting
            df_likelihood.to_csv(os.path.join(out_dir_unboud_model,
                                              'volume dependace of P01 %s.csv'
                                              % stn),
                                 sep=df_sep,
                                 index_label=stn)

            # extract params result to df_dependancy
            df_dependancy.loc[stn, 'a'] = res.x[0]
            df_dependancy.loc[stn, 'b'] = res.x[1]

        # save df_dependancy, contains for each stn, values of a and b
        df_dependancy.to_csv(os.path.join(out_dir_level_one_params,
                                          'loglikehood params.csv'),
                             sep=df_sep,
                             index_label=stn)
        print('done with params of the unbounded model,\
              proceeding to evaluation')

        return df_dependancy

    def assignWvals(self, df_rain_upper_level,
                    df_P01_probabilities_L1,
                    df_beta_params_L1,
                    df_logRegress_params_L1,
                    df_P01_probabilities_L2_1,
                    df_beta_params_L2_1,
                    df_logRegress_params_L2_1,
                    nbr_realisation, R_threshhold,
                    time_delta_level_one,
                    time_delta_level_two):
        '''
        First compare histograms observed and simulated rainfall
        find and compare all values of R99 (frequency of extremes)
        find and compare means of all R > R99 (magnitude of extremes)
        evaluate frequencies and magnitudes

        HOW:
            at every level L, for every R >= threshold:
            randomly assign for the first sub-interval W1 = 0 or W1 != 0
            based on P01
            and if W1 != 0, sample W1 from Px, the fitted beta dist,
            with Beta param,
            R1 = W1*R and R2 = (1-W1)*R

            use input df_60min, and input for level L
            disaggregation is the model
            output of level L-1 disaggregation. So, read upper_df,
            sample weights
            based on P01 and Px, simulate Rvalues at 30min timestamps,
            use the new 30min data as input to the lowest level,
            resample W based on P01 and Px, simulate
            R values that are at 15min timestamps.
            Compare the simulated 30min and 15min data to original values
            and compare
            frequency and magnitude of extremes

            First check results of baseline model, P01 not volume dependent
            Second use unbounded model to include dependency P01 and R
        '''

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
        # out dir, results L1

        out_dir_model_eval_1 = self.createOutDir(out_dir,
                                                 '%s model evaluation'
                                                 % (cascade_level_1))

        # out dir, results L2
        out_dir_model_eval_2 = self.createOutDir(out_dir,
                                                 '%s model evaluation'
                                                 % (cascade_level_2))

        # dictionaries to hold outputs all stations
        dict_out_L1 = {k: [] for k in df_rain_upper_level.columns}
        dict_out_L2 = {k: [] for k in df_rain_upper_level.columns}

        # select values above threshold from upper df, what to disaggregate
        df_rain_upper_level = df_rain_upper_level[
            df_rain_upper_level > R_threshhold]
        # start iterating through stations
        for station in df_rain_upper_level.columns:

            print('simulating for :', station)

            # new df for every station simulations
            df_output_L1 = pd.DataFrame()

            # df2 to hold ouput lower level simulations
            df_output_L2 = pd.DataFrame()

            # drop nan from station values level 1
            df_rain_upper_level[station].dropna(axis=0, inplace=True)

            assert df_rain_upper_level[station].isnull().sum() == 0, \
                print('still zero vals in DF level 1')

    # =========================================================================
    # go through index df level 1
    # =========================================================================

            for idx in df_rain_upper_level[station].index:

                # extract volume R in level 1, what to disaggregate
                val_level_1 = df_rain_upper_level[station].loc[idx]

                # end idx sub int one (00:30:00)
                idx1_1 = idx - pd.\
                    Timedelta(seconds=time_delta_level_one)
    #            assert idx1_1.minute == 30, 'evaluation, L1 locate 1st idx'

                # end idx sub int two (01:00:00)
                idx2_2 = idx
    #            assert idx2_2.minute == 0, 'evaluation,L1 locate 2st idx'

                # idx sub-int 1 in L2 15min
                idx_l2_1 = idx1_1 - pd.\
                    Timedelta(seconds=time_delta_level_two)
    #            assert idx_l2_1.minute == 15, 'evaluation,L2 locate 1st idx'

                # idx sub-int 2 in L2 30min
                idx_l2_2 = idx1_1
    #            assert idx_l2_2.minute == 30, 'evaluation,L2 locate 2st idx'

                # idx sub-int 3 in L2 45min
                idx_l2_3 = idx2_2 - pd.\
                    Timedelta(seconds=time_delta_level_two)
    #            assert idx_l2_3.minute == 45, 'evaluation,L2 locate 3rd idx'

                # idx sub-int 4 in L2 60min
                idx_l2_4 = idx2_2
    #            assert idx_l2_4.minute == 0, 'evaluation,L2 locate 4th idx'

    # =========================================================================
    # baseline model level one
    # =========================================================================

                # read P01 and beta values for station baseline model
                p01 = df_P01_probabilities_L1.loc[station, 'P01']
                beta_val_L1 = df_beta_params_L1.loc[station, 'beta']

                # generate a random var and compare it to P01
                r1 = np.random.uniform()
                rv1 = np.random.uniform()
                # if below P01
                if r1 <= p01:
                    if rv1 <= 0.5:
                        w1 = 0.  # assign W1=0 W2=1
                    elif rv1 > 0.5:
                        w1 = 1.

                elif r1 > p01:
                    # use px to sample W
                    w1 = np.random.beta(beta_val_L1, beta_val_L1, 1)

                # use value of w1 to calculate R1 and R2
                R1 = w1 * val_level_1
                R2 = (1 - w1) * val_level_1

    #            print(R1, R2)
                # store simulated vals in df_out
                df_output_L1.loc[idx1_1,
                                 'baseline rainfall %s' % cascade_level_1] = R1

                df_output_L1.loc[idx2_2,
                                 'baseline rainfall %s' % cascade_level_1] = R2

    # =========================================================================
    # unbounded model level one
    # =========================================================================

                # read P01 and beta values for station unbounded model
                a = df_logRegress_params_L1.loc[station, 'a']
                b = df_logRegress_params_L1.loc[station, 'b']

                # calculate f(R) unbounded model, P01=fct(R)
                p01_unbounded = self.logRegression(val_level_1, a, b)

                # generate a random var and compare it to P01
                r2 = np.random.uniform()
                rv2 = np.random.uniform()

                if r2 <= p01_unbounded:
                    if rv2 <= 0.5:
                        w1_u = 0.  # assign W1=0 , w2=1
                    elif rv2 > 0.5:
                        w1_u = 1.

                elif r2 > p01_unbounded:
                    # use px to sample W
                    w1_u = np.random.beta(beta_val_L1, beta_val_L1, 1)

                # use value of w1 to calculate R1 and R2
                R1_u = w1_u * val_level_1
                R2_u = (1 - w1_u) * val_level_1

                # store simulated vals in df_out
                df_output_L1.loc[idx1_1,
                                 'unbounded rainfall %s'
                                 % cascade_level_1] = R1_u

                df_output_L1.loc[idx2_2,
                                 'unbounded rainfall %s'
                                 % cascade_level_1] = R2_u

    # =========================================================================
    # LEVEL TWO
    # =========================================================================
                '''Input lower level 15min is output this level'''
    # =========================================================================
    # baseline model L2
    # =========================================================================

                # read P01 and beta values for station baseline model
                p01_2_1 = df_P01_probabilities_L2_1.loc[station, 'P01']
                beta_val_2_1 = df_beta_params_L2_1.loc[station, 'beta']

                # generate a random var and compare it to P01
                r3 = np.random.uniform()
                rv3 = np.random.uniform()
                if r3 <= p01_2_1:
                    if rv3 <= 0.5:
                        w1_2_1 = 0.  # assign W1=0 , w2=1
                    elif rv3 > 0.5:
                        w1_2_1 = 1.

                elif r3 > p01_2_1:
                    # use px to sample W
                    w1_2_1 = np.random.beta(beta_val_2_1, beta_val_2_1, 1)

                # use value of w1 to calculate R1 and R2
                R1_2_1 = w1_2_1 * R1
                R2_2_1 = (1 - w1_2_1) * R1

                # store simulated vals in df_out
                df_output_L2.loc[idx_l2_1,
                                 'baseline rainfall %s'
                                 % cascade_level_2] = R1_2_1

                df_output_L2.loc[idx_l2_2,
                                 'baseline rainfall %s'
                                 % cascade_level_2] = R2_2_1

                # generate a random var and compare it to P01
                r4 = np.random.uniform()
                rv4 = np.random.uniform()

                if r4 <= p01_2_1:

                    if rv4 <= 0.5:
                        w1_2_2 = 0.  # assign W1=0 , w2=1
                    elif rv4 > 0.5:
                        w1_2_2 = 1.

                elif r4 > p01_2_1:

                    # use px to sample W
                    w1_2_2 = np.random.beta(beta_val_2_1, beta_val_2_1, 1)

                # use value of w1 to calculate R1 and R2

                R1_2_2 = w1_2_2 * R2
                R2_2_2 = (1 - w1_2_2) * R2

                # store simulated vals in df_out
                df_output_L2.loc[idx_l2_3,
                                 'baseline rainfall %s'
                                 % cascade_level_2] = R1_2_2

                df_output_L2.loc[idx_l2_4,
                                 'baseline rainfall %s'
                                 % cascade_level_2] = R2_2_2

    # =========================================================================
    # unbounded model
    # =========================================================================
                # read P01 and beta values for station unbounded model

                a2_1 = df_logRegress_params_L2_1.loc[station, 'a']
                b2_1 = df_logRegress_params_L2_1.loc[station, 'b']

                # calculate f(R) unbounded model, P01=fct(R)
                p01_unbounded_2_1 = self.logRegression(R1_u, a2_1, b2_1)

                # generate a random var and compare it to P01
                r5 = np.random.uniform()
                rv5 = np.random.uniform()

                if r5 <= p01_unbounded_2_1:

                    if rv5 <= 0.5:
                        w1_u_2_1 = 0.  # assign W1=0 , w2=1
                    elif rv5 > 0.5:
                        w1_u_2_1 = 1.

                elif r5 > p01_unbounded_2_1:

                    # use px to sample W
                    w1_u_2_1 = np.random.beta(beta_val_2_1, beta_val_2_1, 1)

                # use value of w1 to calculate R1 and R2

                R1_u_2_1 = w1_u_2_1 * R1_u
                R2_u_2_1 = (1 - w1_u_2_1) * R1_u

                # store simulated vals in df_out
                df_output_L2.loc[idx_l2_1,
                                 'unbounded rainfall %s'
                                 % cascade_level_2] = R1_u_2_1

                df_output_L2.loc[idx_l2_2,
                                 'unbounded rainfall %s'
                                 % cascade_level_2] = R2_u_2_1

                p01_unbounded_2_2 = self.logRegression(R2_u, a2_1, b2_1)

                # generate a random var and compare it to P01
                r6 = np.random.uniform()
                rv6 = np.random.uniform()

                if r6 <= p01_unbounded_2_2:

                    if rv6 <= 0.5:
                        w1_u_2_2 = 0.  # assign W1=0 , w2=1
                    elif rv6 > 0.5:
                        w1_u_2_2 = 1.

                elif r6 > p01_unbounded_2_2:

                    # use px to sample W
                    w1_u_2_2 = np.random.beta(beta_val_2_1, beta_val_2_1, 1)

                # use value of w1 to calculate R1 and R2
                R1_u_2_2 = w1_u_2_2 * R2_u
                R2_u_2_2 = (1 - w1_u_2_2) * R2_u

                # store simulated vals in df_out
                df_output_L2.loc[idx_l2_3,
                                 'unbounded rainfall %s'
                                 % cascade_level_2] = R1_u_2_2
                df_output_L2.loc[idx_l2_4,
                                 'unbounded rainfall %s'
                                 % cascade_level_2] = R2_u_2_2

            # save df output
            df_output_L1.to_csv(os.path.join(out_dir_model_eval_1,
                                             'model results %s %d.csv'
                                             % (station,
                                                nbr_realisation)),
                                sep=df_sep, float_format='%0.3f')
            # save station results in dict output
            dict_out_L1[station] = df_output_L1

            # save station results in dict output
            df_output_L2.to_csv(os.path.join(out_dir_model_eval_2,
                                             'model results %s %d.csv'
                                             % (station,
                                                nbr_realisation)),
                                sep=df_sep, float_format='%0.3f')

            # save station results in dict output
            dict_out_L2[station] = df_output_L2

            print('done all simulations for station: ', station)
        result_simulations = [dict_out_L1, dict_out_L2]
        return result_simulations

    def buildLorenzCurve(self, in_sim_dfs_L1_dict, orig_df,
                         cascade_level):
        ''' idea: find Irregularity of rainfall distribution
            calculate the cummulative percentage of precipitation contributed
            by the cummulative percentage of days when precipitaiton occured
            method: first sort rainfall values
                define threshhold of wet vs dry hours : mm/hr
                define rainfall amount for class intervals (mm)

            code: determine percentage of rain contributed by hours
                failling in each
              class and compare observed values and model output

              if the daily precipitation falls within a class interval i,
              the number of rainy days in this class, n, is added to index i
              and the amount of rain, xi, is totaled
            '''

        ''' input: precipitation ORIGINAL dfs_L1 , cascade level
            output: new df with values of Loren curve 'X', 'Y'
                    read those in plotting script and plot them
        '''
        out_dir_2 = self.createOutDir(out_dir,
                                      '%s Lorenz curves simulations'
                                      % cascade_level)

        base_name = 'baseline rainfall %s' % cascade_level
        unbound_name = 'unbounded rainfall %s' % cascade_level

        if cascade_level == cascade_level_1:

            # extract results L1 from simulated vals
            sim_Result = in_sim_dfs_L1_dict[0]

            # start going through stations
            for station in sim_Result.keys():

                # extract baseline values
                baseline_values = sim_Result[station][base_name]

                # extract unbouded model values
                unbounded_values = sim_Result[station][unbound_name]

                # add values to a list
                values_simulations = [baseline_values, unbounded_values]

                # start going through simulated values
                for i, model_results in enumerate(values_simulations):

                    # extract from orig df the same simulated dates
                    common_idx = []
                    try:
                        common_idx = orig_df[station].index.\
                            intersection(model_results.index)
                    except Exception as msg:
                        print(msg)
                        continue
                    orig_rain_vals = orig_df[station].loc[common_idx]

                    # new list to hold model values of station
                    rainfall_values = []
                    for val in model_results:
                        rainfall_values.append(round(val, 2))

                    # sort rainfall values ad get unique values
                    rainfall_values_sorted = np.sort(rainfall_values,
                                                     kind='mergesort')
                    orig_rain_vals_sorted = np.sort(orig_rain_vals,
                                                    kind='mergesort')

                    rainfall_values_unique = np.unique(rainfall_values_sorted)
                    orig_rain_vals_unique = np.unique(orig_rain_vals_sorted)

                    # find total rainfall volume
                    total_rainfall = float(np.sum(rainfall_values))
                    total_rainfall_orig = float(np.sum(orig_rain_vals))

                    # find total nbr of values per station
                    total_values = len(rainfall_values)
                    total_values_orig = len(orig_rain_vals)

                    # new list to hold values per class
                    nbr_vals_per_class = []
                    nbr_o_vals_per_class = []

                    # append values in each class, using unique R values
                    for th in rainfall_values_unique:
                        nbr_vals_per_class.append(
                            (rainfall_values <= th).sum())

                    for th_o in orig_rain_vals_unique:
                        nbr_o_vals_per_class.append(
                            (orig_rain_vals <= th_o).sum())

                    # count nbr of occurence per rainfall
                    rainfall_occurences = Counter(rainfall_values_sorted)
                    rainfall_o_occurences = Counter(orig_rain_vals_sorted)

                    # new df , index=rainfall values, colums= occurences
                    df_rain_occurences = pd.DataFrame.\
                        from_dict(rainfall_occurences, orient='index')
                    df_rain_occurences.rename(columns={0: 'Ni'},
                                              inplace=True)

                    df_o_rain_occurences = pd.DataFrame.\
                        from_dict(rainfall_o_occurences, orient='index')
                    df_o_rain_occurences.rename(columns={0: 'Ni O'},
                                                inplace=True)

                    # get nbr or rainfall per class
                    df_rain_occurences['sum Ni'] = nbr_vals_per_class
                    df_o_rain_occurences['sum Ni O'] = nbr_o_vals_per_class

                    df_rain_occurences['Pi'] = df_rain_occurences.index\
                        * df_rain_occurences['Ni']
                    df_o_rain_occurences['Pi O'] = df_o_rain_occurences.index\
                        * df_o_rain_occurences['Ni O']

                    df_rain_occurences['sum Pi'] = \
                        df_rain_occurences['Pi'].cumsum()

                    df_o_rain_occurences['sum Pi O'] = \
                        df_o_rain_occurences['Pi O'].cumsum()

                    df_rain_occurences['X'] = \
                        df_rain_occurences['sum Ni'] / total_values

                    df_o_rain_occurences['X O'] = \
                        df_o_rain_occurences['sum Ni O'] / total_values_orig

                    df_rain_occurences['Y'] = \
                        df_rain_occurences['sum Pi'] / total_rainfall

                    df_o_rain_occurences['Y O'] = \
                        df_o_rain_occurences['sum Pi O'] / total_rainfall_orig

                    if i == 0:
                        model_name = 'baseline'
                    elif i == 1:
                        model_name = 'unbounded'

                    df_rain_occurences.to_csv(os.path.join(out_dir_2,
                                                           ('Lorenz curve'
                                                            '%s %s %s.csv')
                                                           % (station,
                                                              model_name,
                                                              cascade_level)),
                                              sep=df_sep)

                    df_o_rain_occurences.to_csv(os.path.join(out_dir_2,
                                                             ('Orig Lorenz curve'
                                                              '%s %s.csv')
                                                             % (station,
                                                                 cascade_level)),
                                                sep=df_sep)
        elif cascade_level == cascade_level_2:

            sim_Result_ = in_sim_dfs_L1_dict[1]

            for station in sim_Result_.keys():

                baseline_values = sim_Result_[station][base_name]

                unbounded_values = sim_Result_[station][unbound_name]

                values_simulations = [baseline_values, unbounded_values]

                for i, model_results in enumerate(values_simulations):

                    # extract from orig df the same simulated dates
                    common_idx = []
                    try:
                        common_idx = orig_df[station].index.\
                            intersection(model_results.index)
                    except Exception as msg:
                        print(msg)
                        continue
                    orig_rain_vals = orig_df[station].loc[common_idx]

                    # new list to hold model values of station
                    rainfall_values = []
                    for val in model_results:
                        rainfall_values.append(round(val, 2))

                    # sort rainfall values ad get unique values
                    rainfall_values_sorted = (np.
                                              sort(rainfall_values,
                                                   kind='mergesort'))
                    orig_rain_vals_sorted = (np.
                                             sort(orig_rain_vals,
                                                  kind='mergesort'))

                    rainfall_values_unique = np.unique(rainfall_values_sorted)
                    orig_rain_vals_unique = np.unique(orig_rain_vals_sorted)

                    # find total rainfall volume
                    total_rainfall = float(np.sum(rainfall_values))
                    total_rainfall_orig = float(np.sum(orig_rain_vals))

                    # find total nbr of values per station
                    total_values = len(rainfall_values)
                    total_values_orig = len(orig_rain_vals)
                    # new list to hold values per class
                    nbr_vals_per_class = []
                    nbr_o_vals_per_class = []
                    # append values in each class, using unique R values
                    for th in rainfall_values_unique:
                        nbr_vals_per_class.append(
                            (rainfall_values <= th).sum())
                    for th_o in orig_rain_vals_unique:
                        nbr_o_vals_per_class.append(
                            (orig_rain_vals <= th_o).sum())

                    # nbr of occurence per rainfall
                    rainfall_occurences = Counter(rainfall_values_sorted)
                    rainfall_o_occurences = Counter(orig_rain_vals_sorted)

                    # new df , index=rainfall values, colums= occurences
                    df_rain_occurences = pd.DataFrame.\
                        from_dict(rainfall_occurences, orient='index')
                    df_rain_occurences.rename(columns={0: 'Ni'},
                                              inplace=True)

                    df_o_rain_occurences = pd.DataFrame.\
                        from_dict(rainfall_o_occurences, orient='index')
                    df_o_rain_occurences.rename(columns={0: 'Ni O'},
                                                inplace=True)

                    # get nbr or rainfall per class
                    df_rain_occurences['sum Ni'] = nbr_vals_per_class
                    df_o_rain_occurences['sum Ni O'] = nbr_o_vals_per_class

                    df_rain_occurences['Pi'] = df_rain_occurences.index\
                        * df_rain_occurences['Ni']
                    df_o_rain_occurences['Pi O'] = df_o_rain_occurences.index\
                        * df_o_rain_occurences['Ni O']

                    df_rain_occurences['sum Pi'] = \
                        df_rain_occurences['Pi'] .cumsum()
                    df_o_rain_occurences['sum Pi O'] = \
                        df_o_rain_occurences['Pi O'].cumsum()

                    df_rain_occurences['X'] = \
                        df_rain_occurences['sum Ni'] / total_values
                    df_o_rain_occurences['X O'] = \
                        df_o_rain_occurences['sum Ni O'] / total_values_orig

                    df_rain_occurences['Y'] = \
                        df_rain_occurences['sum Pi'] / total_rainfall
                    df_o_rain_occurences['Y O'] = \
                        df_o_rain_occurences['sum Pi O'] / total_rainfall_orig

                    if i == 0:
                        model_name = 'baseline'
                    elif i == 1:
                        model_name = 'unbounded'

                    df_rain_occurences.to_csv(os.path.join(out_dir_2,
                                                           ('Lorenz curve'
                                                            '%s %s %s.csv')
                                                           % (station,
                                                              model_name,
                                                              cascade_level)),
                                              sep=df_sep)
                    df_o_rain_occurences.to_csv(os.path.join(out_dir_2,
                                                             ('Orig Lorenz curve'
                                                              '%s %s.csv')
                                                             % (station,
                                                                 cascade_level)),
                                                sep=df_sep)

        print('done calculating the lorenz curves')


if __name__ == '__main__':

    print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
    START = timeit.default_timer()  # to get the runtime of the program

    # define main and output dir, what to change
    main_dir = (r'X:\hiwi\ElHachem\Jochen\CascadeModel')
    os.chdir(main_dir)

    # out_dir = os.path.join(main_dir, r'Weights')
    out_dir = os.path.join(main_dir, r'Weights')
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # define rainfall data directory
    data_dir = r'X:\hiwi\ElHachem\Jochen\CascadeModel'

    # read 5min data file
    in_5m_df_file = os.path.join(data_dir, 'rr_5min_all.csv')
    # in_1m_df_file = os.path.join(data_dir,
    # 'ppt_data_for_radolan_2005_2018.csv')

    assert os.path.exists(in_5m_df_file), \
        'wrong 5min df file location'

    # =========================================================================
    # what to change when new data
    # =========================================================================
    # for reading df
    df_sep = '\t'
    # date_fotmat = '%Y-%m-%d %H:%M:%s'
    date_fotmat = '%d.%m.%Y %H:%M'

    index_col = 0

    # define volume threshhold R (mm) minimal and maximal per 5min or less
    threshhold = 0.1
    r_max = 100

    # factor for deviding data, incase needed
    ppt_divide_fct = 1

    # for Prob cals
    min_nbr_vals = 5.

    # name of cascade level
    cascade_level_1 = 'Level one'
    cascade_level_2 = 'Level two'

    # define time delta to substract from df index to get values in other df
    time_delta_30min = 1800  # 1800s 30min (don't change)
    time_delta_15min = 900  # s 15min

    # constrain beta values >1 otherwise won't be an envelop of the data
    initial_beta_vls = [2.2, 2.2]  # if symmetric beta
    beta_bounds = [(2, 5.), (2, 5.)]

    # define constraint when using symmetric beta_dist_fct: alfa=beta
    cons = {'type': 'eq', 'fun': lambda x: x[1] - x[0]}

    # change this to improve fit of logRegression (2params a, b)
    a_b_intial_vls = [-1., 1.9]
    a_b_params_bounds = [(None, None), (1.8, None)]

    # def nbr of simulated timeseries
    wanted_simulations = 2
    cascade_model = CascadeModel()

    in_df_5m = cascade_model.readDf(in_5m_df_file, df_sep,
                                    date_fotmat, index_col,
                                    ppt_divide_fct, r_max)

    # drop columns from original df
    in_df_5m.drop(columns=['rr_01', 'rr_02', 'rr_03', 'rr_04',
                           'rr_05', 'rr_06', 'rr_10'],
                  inplace=True)

    df_5m_to_60m = cascade_model.resampleDf(in_df_5m, '60T', 0, out_dir,
                                            df_sep, fillnan=True)
    df_5m_to_30m = cascade_model.resampleDf(in_df_5m, '30T', 0, out_dir,
                                            df_sep, fillnan=True,
                                            df_save_name='resampled 30min.csv')
    df_5m_to_15m = cascade_model.resampleDf(in_df_5m, '15T', 0, out_dir,
                                            df_sep, fillnan=True,
                                            df_save_name='resampled 15min.csv')

    cascade_model.neglectedData(df_5m_to_60m, threshhold,
                                out_dir, df_sep, save_df_output=False)

    cascade_model.neglectedData(df_5m_to_30m, threshhold,
                                out_dir, df_sep, save_df_output=False)

    w_l1, dfw1 = cascade_model.calculateWeightsL1(df_5m_to_60m,
                                                  df_5m_to_30m,
                                                  threshhold,
                                                  time_delta_30min)
    w_l2, dfw2 = cascade_model.calculateWeightsL2(df_5m_to_30m,
                                                  df_5m_to_15m,
                                                  threshhold,
                                                  time_delta_15min)
    p1 = cascade_model.calculateP01fromWeights(w_l1, min_nbr_vals,
                                               cascade_level_1)
    p2 = cascade_model.calculateP01fromWeights(w_l1, min_nbr_vals,
                                               cascade_level_2)

    sd = cascade_model.weightsSortMonth(dfw1, cascade_level_1)
    sd = cascade_model.weightsSortMonth(dfw2, cascade_level_2)

    beta1 = cascade_model.fitBetaWeights(w_l1, cascade_level_1,
                                         initial_beta_vls, beta_bounds,
                                         cons)
    beta2 = cascade_model.fitBetaWeights(w_l2, cascade_level_2,
                                         initial_beta_vls, beta_bounds,
                                         cons)

    lk1 = cascade_model.volumeDependacy(dfw1, cascade_level_1,
                                        a_b_intial_vls,
                                        a_b_params_bounds)

    lk2 = cascade_model.volumeDependacy(dfw2, cascade_level_2,
                                        a_b_intial_vls,
                                        a_b_params_bounds)

    simulation_nbr = 1
    while simulation_nbr <= wanted_simulations:

        simulation_nbr += 1
        print('start with simualtion number: ', simulation_nbr)
        evaluation_L1 = cascade_model.assignWvals(df_5m_to_60m,
                                                  p1,
                                                  beta1,
                                                  lk1,
                                                  p2,
                                                  beta2,
                                                  lk2,
                                                  simulation_nbr,
                                                  threshhold,
                                                  time_delta_30min,
                                                  time_delta_15min)
        print('done with model evaluation, proceeding to Lorenz Curves')

    lorenz_sim_L1 = cascade_model.buildLorenzCurve(evaluation_L1, df_5m_to_30m,
                                                   cascade_level_1)

    lorenz_sim_L2 = cascade_model.buildLorenzCurve(evaluation_L1, df_5m_to_15m,
                                                   cascade_level_2)

    STOP = timeit.default_timer()  # Ending time
    print(('\n\a\a\a Done with everything on %s. Total run time was'
           ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP - START)))
