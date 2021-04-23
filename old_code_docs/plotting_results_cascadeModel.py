# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas, IWS
Institut für Wasser- und Umweltsystemmodellierung - IWS
"""
from scipy.stats import beta, rankdata
from scipy.stats import spearmanr as spr
from scipy.stats import ks_2samp as KOL

import os
import timeit
import time
import matplotlib.dates as md
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fnmatch

''' this script is for plotting the result of cascadeModel.py:

    plot the histograam of the sample weights and plot the fitted beta
    distribution, this function is called for level one and two.

    plot the probability P01 that W=0 or W=1, for each station, each level

    plot the seasonal effect on the probability P01, for each month and level

    plot the logistic regression fitted function vs the classified rainfall
    part of the unbounded model, do it for all stations and each level

    plot the results of the baseline and unbounded model, using the fitted
    parameters, a subplot with original, baseline and unbounded rainfall

    plot the result of the mean of the 100 simulations done for the baseline
    and the unbounded model and compare to original values
'''

plt.ioff()

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer()  # to get the runtime of the program

# what to CHANGE
main_dir = (r'X:\hiwi\ElHachem\Jochen')
os.chdir(main_dir)

# read 5min data file
in_5m_df_file = os.path.join(main_dir, r'CascadeModel\rr_5min_all.csv')


assert in_5m_df_file

# def data dir
in_data_dir = os.path.join(main_dir,
                           r'CascadeModel\Weights')

# =============================================================================
# Level ONE get all needed dfs (no need to change)
# =============================================================================
cascade_level_1 = 'Level one'

# in_df of P0 and P1 per month
in_data_prob_df_file_L1 = os.path.join(in_data_dir,
                                       'P1 P0 per month %s.csv'
                                       % cascade_level_1)
# in_df of fitted beta dist params
in_data_beta_df_file_L1 = os.path.join(in_data_dir,
                                       r'bounded maximum likelihood %s.csv'
                                       % cascade_level_1)
# in_df of P01 per stn
in_data_prob_stn_df_file_L1 = os.path.join(in_data_dir,
                                           r'Prob W P01 %s.csv'
                                           % cascade_level_1)
# in_df of w_vals, R_wals and logRegression cols
unbounded_model_dir_L1 = os.path.join(in_data_dir,
                                      r'%s P01 volume dependancy'
                                      % cascade_level_1)
# in_df of logistic regression params
params_file_L1 = os.path.join(unbounded_model_dir_L1,
                              r'%s log_regress params' % cascade_level_1,
                              r'loglikehood params.csv')

# in_df results of simulation for model evaluation

in_dfs_simulation = os.path.join(in_data_dir,
                                 r'%s model evaluation'
                                 % cascade_level_1)

in_dfs_simulation_01 = os.path.join(in_data_dir,
                                    r'%s model evaluation_'
                                    % cascade_level_1)
# read original values, to compare to model
in_df_30min_orig = os.path.join(in_data_dir, r'resampled 30min.csv')

in_lorenz_df_L1 = os.path.join(in_data_dir,
                               r'%s Lorenz curves original'
                               % cascade_level_1)

# read dfs holding the results of Lorenz curves of simulated values
in_lorenz_df_L1_sim = os.path.join(in_data_dir,
                                   r'%s Lorenz curves simulations'
                                   % cascade_level_1)

# =============================================================================
# Level TWO ONE get all needed dfs (no need to change)
# =============================================================================

cascade_level_2 = 'Level two'

# in_df of P0 and P1 per month
in_data_prob_df_file_L2 = os.path.join(in_data_dir,
                                       'P1 P0 per month %s.csv'
                                       % cascade_level_2)
# in_df of fitted beta dist params
in_data_beta_df_file_L2 = os.path.join(in_data_dir,
                                       r'bounded maximum likelihood %s.csv'
                                       % cascade_level_2)
# in_df of P01 per stn
in_data_prob_stn_df_file_L2 = os.path.join(in_data_dir,
                                           r'Prob W P01 %s.csv'
                                           % cascade_level_2)

# in_df of w_vals, R_wals and logRegression cols
unbounded_model_dir_L2 = os.path.join(in_data_dir,
                                      r'%s P01 volume dependancy'
                                      % cascade_level_2)
# in_df of logistic regression params
params_file_L2 = os.path.join(unbounded_model_dir_L2,
                              r'%s log_regress params' % cascade_level_2,
                              r'loglikehood params.csv')

in_dfs_simulation_2 = os.path.join(in_data_dir,
                                   r'%s model evaluation' % cascade_level_2)
in_dfs_simulation_02 = os.path.join(in_data_dir,
                                    r'%s model evaluation_' % cascade_level_2)

in_df_15min_orig = os.path.join(in_data_dir, r'resampled 15min.csv')

in_lorenz_df_L2 = os.path.join(in_data_dir,
                               r'%s Lorenz curves original' % cascade_level_2)

in_lorenz_df_L2_sim = os.path.join(in_data_dir,
                                   r'%s Lorenz curves simulations'
                                   % cascade_level_2)


# =============================================================================
# create OUT dir and make sure all IN dir are correct
# =============================================================================

# def out_dir to hold plots
out_dir = os.path.join(main_dir,
                       r'Histograms_Weights')
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

out_dir_all = os.path.join(out_dir,
                           'out_subplots')
if not os.path.exists(out_dir_all):
    os.mkdir(out_dir_all)

# make sure all defined directories exist
assert os.path.exists(in_data_dir), 'wrong data DF location'

# LEVEL ONE
assert os.path.exists(in_data_prob_df_file_L1),\
    'wrong data prob location L1'
assert os.path.exists(in_data_prob_stn_df_file_L1),\
    'wrong data stn prob location L1'
assert os.path.exists(in_data_beta_df_file_L1),\
    'wrong data beta location L1'
assert os.path.exists(unbounded_model_dir_L1),\
    'wrong unbounded model DF location L1'
assert os.path.exists(params_file_L1),\
    'wrong params DF location L1'
assert os.path.exists(in_dfs_simulation),\
    'wrong simulation DF location L1'
assert os.path.exists(in_df_30min_orig),\
    'wrong orig DF location L1'

# assert os.path.exists(in_lorenz_df_L1),\
#         'wrong Lorenz Curves original Df L1'

assert os.path.exists(in_lorenz_df_L1_sim),\
    'wrong Lorenz Curves simu Df L1'

# LEVEL TWO ONE
assert os.path.exists(in_data_prob_df_file_L2),\
    'wrong data prob location L2'
assert os.path.exists(in_data_prob_stn_df_file_L2),\
    'wrong data stn prob location L2'
assert os.path.exists(in_data_beta_df_file_L2),\
    'wrong data beta location L2'
assert os.path.exists(unbounded_model_dir_L2),\
    'wrong unbounded model DF location L2'
assert os.path.exists(params_file_L2),\
    'wrong params DF location L2'
assert os.path.exists(in_dfs_simulation_2),\
    'wrong simulation DF location L2'

assert os.path.exists(in_df_15min_orig),\
    'wrong orig DF location L2'

# assert os.path.exists(in_lorenz_df_L2),\
#         'wrong Lorenz Curves original Df L2'
#
assert os.path.exists(in_lorenz_df_L2_sim),\
    'wrong Lorenz Curves Df L2'

# used for plotting and labelling, what to change
wanted_stns_list = ['rr_07',	'rr_09', 'rr_10']

# =============================================================================
# class to break inner loop and execute outer loop when needed
# =============================================================================


class ContinueI(Exception):
    pass

# ============================================================================
# def what to plot
# =============================================================================


rain_weights = False
weights = False
dependency = False
histogram = False
lorenz = False
cdf_max = False

plotP01Station = False
plotP01Month = False

buildDFsim = False
plotCdfSim = False
boxPlot = False
rankedHist = False
shiftedOrigSimValsCorr = False
plotShiftedOrigSimVals = False
plotAllSim = False
# =============================================================================
#
# =============================================================================

# def fig size, will be read in imported module to plot Latex like plots
# fig_size = (17, 21)
fig_size = (30, 20)
dpi = 80
save_format = '.pdf'

font_size_title = 22     # 16
font_size_axis = 20        # 20
font_size_legend = 20    # 20

marker_size = 150     # 80
w = 0.
# define df_seperator
df_sep = '\t'

#df_sep = ';'

date_format = '%Y-%m-%d %H:%M:%S'

# define transparency
transparency = 0.5

# define ratio of bars, and width decrease factor
ratio_bars = 0.045
ratio_decrease_factor = 0.99

# define if normed or not
norme_it = True
if histogram:
    norme_it = False

# define line width
line_width = 0.01

# =============================================================================
#
# =============================================================================


def getFiles(data_dir, cascade_level):
    # create function to get files based on dir and cascade level

    dfs_files = []
    for r, dir_, f in os.walk(os.path.join(data_dir,
                                           r'%s' % cascade_level)):
        for fs in f:
            if fs.endswith('.csv'):
                dfs_files.append(os.path.join(r, fs))
    return dfs_files


# get files for l1
dfs_files_L1 = getFiles(in_data_dir, cascade_level_1)
# get files for l2
dfs_files_L2 = getFiles(in_data_dir, cascade_level_2)

# =============================================================================
#
# =============================================================================


def getHistWeights(in_dfs_data_list, in_df_beta_param, cascade_level):
    '''
    input: weights_dfs , beta_params_df, cascade_level
    output: plots of weights histogram and fitted beta pdf
    '''
    wanted_stns_data = {k: {'fct': [], 'data': []}
                        for k in wanted_stns_list}

    # read beta pdf params results
    df_beta = pd.read_csv(in_df_beta_param, sep=df_sep, index_col=0)

    # go through file and stations
    for df_file in in_dfs_data_list:

        for station in (df_beta.index):

            for stn_wanted in wanted_stns_list:

                if str(station) == stn_wanted and str(station) in df_file:

                    # read each df-file and round values
                    d = pd.read_csv(df_file, sep=df_sep, index_col=0)
                    rounded_df = d
                    # select stn beta distribution params
                    a = df_beta.loc[station, 'alfa']
                    b = df_beta.loc[station, 'beta']

                    for k, col in enumerate(rounded_df.columns):

                        # plot weights sub interval left W1
                        if k == 0:

                            # select weights between 0 and 1 for plot
                            for val in rounded_df[col].values:
                                if val != 0 and val <= 1e-8:  # if low vals
                                    rounded_df[col].replace(val, value=0.,
                                                            inplace=True)

                            rounded_df2 = rounded_df.loc[
                                (rounded_df[col] != 0.0) &
                                (rounded_df[col] != 1.0)]

                            # define bins nbr for histogram of weights

                            bins = np.arange(0., 1.01, 0.045)
                            center = (bins[:-1] + bins[1:]) / 2

                            # plot hist weights 0 < W < 1
                            hist, bins = np.histogram(rounded_df2[col].values,
                                                      bins=bins,
                                                      normed=norme_it)
                            if type(station) is not str:
                                station = str(station)
                            wanted_stns_data[station]['data'].append(
                                [center, hist,
                                 len(rounded_df2[col].index)])

                            wanted_stns_data[station]['fct'].append(
                                [rounded_df2[col].values,
                                 beta.pdf(
                                     rounded_df2[col].values, a, b),
                                 (a, 0)])

    return wanted_stns_data
# =============================================================================
#
# =============================================================================


def plotScatterWeightsRainfall(in_df_data, cascade_level):
    ''' scatter plot rainfall values vs sampled weights'''
    wanted_stns_data = {k: {'fct': [], 'data': []}
                        for k in wanted_stns_list}

    for stn_wanted in wanted_stns_list:
        for df_file in in_df_data:
            if stn_wanted in df_file:
                in_df = pd.read_csv(df_file, sep=df_sep, index_col=0)

                w_vals = in_df['%s Sub Int Left' % stn_wanted].values
                r_vals = in_df['%s Original Volume' %
                               stn_wanted].values
                wanted_stns_data[stn_wanted]['fct'].append(
                    [w_vals, r_vals, 0])
    return wanted_stns_data
# =============================================================================
#
# =============================================================================


def plotdictData(data_dict1, data_dict2,
                 cascade_level1, cascade_level2,
                 xlabel, ylabel, var):

    f, axes = plt.subplots(3, 2,
                           figsize=fig_size, dpi=dpi,
                           sharex='col', sharey='row')

    for j, stn in enumerate(data_dict1.keys()):
        axes[j, 0].set_axisbelow(True)
        axes[j, 0].yaxis.grid(color='gray',
                              linestyle='dashed',
                              linewidth=line_width,
                              alpha=0.2)
        axes[j, 0].xaxis.grid(color='gray',
                              linestyle='dashed',
                              linewidth=line_width,
                              alpha=0.2)
        axes[j, 0].tick_params(axis='y',
                               labelsize=font_size_axis)
        if histogram is False and lorenz is not True\
                and cdf_max is False:
            if rain_weights is True:
                label_ = 'Sampled Weights'
            if weights is True:
                label_ = 'Fitted Beta distribution function'
            if dependency is True:
                label_ = 'Maximum likelihood model'
                data_dict1[stn]['fct'][0][1] =\
                    data_dict1[stn]['fct'][0][1].reshape(
                        data_dict1[stn]['fct'][0][1].shape[-1], )
                data_dict1[stn]['fct'][0][0].sort()
                data_dict1[stn]['fct'][0][1][::-1].sort()

            axes[j, 0].plot(data_dict1[stn]['fct'][0][0],
                            data_dict1[stn]['fct'][0][1],
                            color='r',
                            alpha=0.85,
                            label=label_)
        if rain_weights is True:

            title = (('%s \n%s'
                      % (stn, cascade_level1)))

            in_title = ''
            if stn == 'rr_07':
                xscale, yscale = 0.8, 30
                xscale2, yscale2 = 0.8, 3.75
            if stn == 'rr_09':
                xscale, yscale = 0.8, 24
                xscale2, yscale2 = 0.8, 3.75
            if stn == 'rr_10':
                xscale, yscale = 0.8, 25
                xscale2, yscale2 = 0.8, 2.3
        if weights is True:

            title = (('%s \n%s'
                      % (stn, cascade_level1)))

            in_title = (r'$\beta$=%0.2f'
                        % data_dict1[stn]['fct'][0][2][0])

            xscale, yscale = 0.8, 2.4
            xscale2, yscale2 = 0.8, 2.15

            if stn == 'rr_09':
                xscale, yscale = 0.8, 4.15
                xscale2, yscale2 = 0.8, 3.75
            if stn == 'rr_10':
                xscale, yscale = 0.8, 2.6
                xscale2, yscale2 = 0.8, 2.3
            axes[j, 0].bar(data_dict1[stn]['data'][0][0],
                           data_dict1[stn]['data'][0][1],
                           align='center',
                           width=ratio_bars,
                           alpha=transparency,
                           linewidth=line_width,
                           color='b',
                           label='Observed weights')

        elif dependency is True:

            title = ('%s \n%s'
                     % (stn, cascade_level1))

            in_title = (('a=%.2f \n'
                         'b=%.2f')
                        % (data_dict1[stn]['fct'][0][2][0],
                           data_dict1[stn]['fct'][0][2][1]))
#            in_title2 = (('Count of\n'
#                          '$W=0$ or\n$W=1$:\n'
#                          '%d')
#                         % ((data_dict1[stn]['data'][0][2])))

            xscale, yscale = 0.87, 0.43
            xscale2, yscale2 = 0.87, 0.36

            axes[j, 0].scatter(data_dict1[stn]['data'][0][0],
                               data_dict1[stn]['data'][0][1],
                               color='b',
                               marker='*',
                               alpha=0.85,
                               label='Observed rainfall')
            axes[j, 0].set_ylim([-0.01, 0.5])
            axes[j, 0].set_xlim([-0.5, 1.26])

        elif histogram is True:
            if stn == 'rr_07':
                xscale, yscale = 0.95, 350
                xscale2, yscale2 = 0.95, 345
            if stn == 'rr_09':
                xscale, yscale = 0.95, 245
                xscale2, yscale2 = 0.95, 240
            if stn == 'rr_10':
                xscale, yscale = 0.95, 105
                xscale2, yscale2 = 0.95, 100
            title = (('%s \n%s')
                     % (stn, cascade_level1))

            in_title = ''
            axes[j, 0].bar(data_dict1[stn]['fct'][0][0][0],
                           data_dict1[stn]['fct'][0][0][1],
                           align='center',
                           width=ratio_bars,
                           alpha=0.5,
                           linewidth=line_width,
                           color='blue',
                           label='Original values',
                           edgecolor='darkblue')
            axes[j, 0].bar(data_dict1[stn]['fct'][0][1][0],
                           data_dict1[stn]['fct'][0][1][1],
                           align='center',
                           width=ratio_bars*0.8,
                           alpha=0.7,
                           linewidth=line_width,
                           color='red',
                           label='Basic model',
                           edgecolor='darkred')
            axes[j, 0].bar(data_dict1[stn]['fct'][0][2][0],
                           data_dict1[stn]['fct'][0][2][1],
                           align='center',
                           width=ratio_bars*0.6,
                           alpha=0.5,
                           linewidth=line_width,
                           color='lime',
                           label='Dependent model',
                           edgecolor='darkgreen')

            axes[j, 0].set_axisbelow(True)
            axes[j, 0].yaxis.grid(color='gray',
                                  linestyle='dashed',
                                  linewidth=line_width,
                                  alpha=0.2)
            axes[j, 0].xaxis.grid(color='gray',
                                  linestyle='dashed',
                                  linewidth=line_width,
                                  alpha=0.2)
        elif lorenz is True:

            axes[j, 0].set_ylim([0, 1])
            xscale, yscale = 0.13, 0.85
            xscale2, yscale2 = 0.13, 0.81

            title = (('%s \n%s')
                     % (stn, cascade_level1))

            in_title = ''
            axes[j, 0].scatter(data_dict1[stn]['fct'][0][0][0],
                               data_dict1[stn]['fct'][0][0][1],
                               color='b',
                               alpha=0.7,
                               s=marker_size*0.7,
                               marker='o',
                               label='Observed rainfall')
            axes[j, 0].scatter(data_dict1[stn]['fct'][0][1][0],
                               data_dict1[stn]['fct'][0][1][1],
                               color='r',
                               s=marker_size*0.9,
                               marker='*',
                               alpha=0.5,
                               label='Basic model')
            axes[j, 0].scatter(data_dict1[stn]['fct'][0][2][0],
                               data_dict1[stn]['fct'][0][2][1],
                               color='g',
                               s=marker_size*0.8,
                               marker='+',
                               alpha=0.5,
                               label='Dependent model')

        elif cdf_max is True:
            xscale, yscale = 28.47, 0.08
            xscale2, yscale2 = 28.47, 0.025

            axes[j, 0].set_ylim([0, 1])

            title = (('%s \n%s')
                     % (stn, cascade_level1))

            in_title = ''
            axes[j, 0].plot(data_dict1[stn]['fct'][0][0][0],
                            data_dict1[stn]['fct'][0][0][1],
                            color='b',
                            alpha=0.6,
                            label='Observed rainfall',
                            linewidth=5.)
            axes[j, 0].plot(data_dict1[stn]['fct'][0][1][0],
                            data_dict1[stn]['fct'][0][1][1],
                            color='r',
                            alpha=0.5,
                            label='Basic model',
                            linewidth=5)
            axes[j, 0].plot(data_dict1[stn]['fct'][0][2][0],
                            data_dict1[stn]['fct'][0][2][1],
                            color='g',
                            alpha=0.5,
                            label='Dependent model',
                            linewidth=5.)

            df_kls_test_1.loc['orig_bas', stn] = KOL(
                data_dict1[stn]['fct'][0][0][0],
                data_dict1[stn]['fct'][0][1][0])[0]
            df_kls_test_1.loc['orig_dpt', stn] = KOL(
                data_dict1[stn]['fct'][0][0][0],
                data_dict1[stn]['fct'][0][2][0])[0]
            df_kls_test_1.loc['bas_dpt', stn] = KOL(
                data_dict1[stn]['fct'][0][1][0],
                data_dict1[stn]['fct'][0][2][0])[0]

            axes[j, 0].set_axisbelow(True)
            axes[j, 0].yaxis.grid(color='gray',
                                  linestyle='dashed',
                                  linewidth=line_width,
                                  alpha=0.2)
            axes[j, 0].xaxis.grid(color='gray',
                                  linestyle='dashed',
                                  linewidth=line_width,
                                  alpha=0.2)
            axes[j, 0].set_xlim([np.min(
                [data_dict1[stn]['fct'][0][0][0],
                 data_dict1[stn]['fct'][0][1][0],
                 data_dict1[stn]['fct'][0][2][0]]),
                35])
        axes[j, 0].text(xscale, yscale,
                        title, fontsize=font_size_title)

        axes[j, 0].text(xscale2, yscale2,
                        in_title,
                        fontsize=font_size_title)
        axes[j, 0].set_ylabel(ylabel,
                              fontsize=font_size_axis)
        axes[j, 0].tick_params(axis='y',
                               labelsize=font_size_axis)
        if j == 2:
            axes[j, 0].set_xlabel(xlabel,
                                  fontsize=font_size_axis)
            axes[j, 0].tick_params(axis='x',
                                   labelsize=font_size_axis)
# =============================================================================
#
# =============================================================================
    for k, stn in enumerate(data_dict2.keys()):
        axes[k, 1].set_axisbelow(True)
        axes[k, 1].yaxis.grid(color='gray',
                              linestyle='dashed',
                              linewidth=line_width,
                              alpha=0.2)
        axes[k, 1].xaxis.grid(color='gray',
                              linestyle='dashed',
                              linewidth=line_width,
                              alpha=0.2)
        axes[k, 1].tick_params(axis='y',
                               labelsize=font_size_axis)
        if histogram is False and lorenz is not True\
                and cdf_max is False:

            if rain_weights is True:
                label_ = 'Sampled Weights'
            if weights is True:
                label_ = 'Fitted Beta distribution function'
            if dependency is True:
                label_ = 'Maximum likelihood model'
                data_dict2[stn]['fct'][0][1] =\
                    data_dict2[stn]['fct'][0][1].reshape(
                    data_dict2[stn]['fct'][0][1].shape[-1], )
                data_dict2[stn]['fct'][0][0].sort()
                data_dict2[stn]['fct'][0][1][::-1].sort()
            axes[k, 1].plot(data_dict2[stn]['fct'][0][0],
                            data_dict2[stn]['fct'][0][1],
                            color='r',
                            alpha=0.85,
                            label=label_)

        if rain_weights is True:

            title = (('%s \n%s'
                      % (stn, cascade_level2)))

            in_title = ''
            if stn == 'rr_07':
                xscale, yscale = 0.8, 30
                xscale2, yscale2 = 0.8, 3.75
            if stn == 'rr_09':
                xscale, yscale = 0.8, 24
                xscale2, yscale2 = 0.8, 3.75
            if stn == 'rr_10':
                xscale, yscale = 0.8, 25
                xscale2, yscale2 = 0.8, 2.3
        if weights is True:

            title = (('%s \n%s'
                      % (stn, cascade_level2)))

            in_title = (r'$\beta$=%0.2f'
                        % data_dict2[stn]['fct'][0][2][0])

            xscale, yscale = 0.8, 2.4
            xscale2, yscale2 = 0.8, 2.12

            if stn == 'rr_09':
                xscale, yscale = 0.8, 4.15
                xscale2, yscale2 = 0.8, 3.75
            if stn == 'rr_10':
                xscale, yscale = 0.8, 2.6
                xscale2, yscale2 = 0.8, 2.3
            axes[k, 1].bar(data_dict2[stn]['data'][0][0],
                           data_dict2[stn]['data'][0][1],
                           align='center',
                           width=ratio_bars,
                           alpha=transparency,
                           linewidth=line_width,
                           color='b',
                           label='Observed weights')

        elif dependency is True:

            title = (('%s \n%s')
                     % (stn, cascade_level2))

            in_title = (('a=%.2f \n'
                         'b=%.2f')
                        % (data_dict2[stn]['fct'][0][2][0],
                           data_dict2[stn]['fct'][0][2][1]))
            xscale, yscale = 0.87, 0.43
            xscale2, yscale2 = 0.87, 0.36

            axes[k, 1].scatter(data_dict2[stn]['data'][0][0],
                               data_dict2[stn]['data'][0][1],
                               color='b',
                               marker='*',
                               alpha=0.85,
                               label='Observed Rainfall')
            axes[k, 1].set_ylim([-0.01, 0.5])
            axes[k, 1].set_xlim([-0.5, 1.26])

#                in_title2 = (('Count of\n'
#                              '$W=0$ or\n$W=1$:\n'
#                              '%d')
#                             % ((data_dict2[stn]['data'][0][2])))
        elif histogram is True:
            if stn == 'rr_07':
                xscale, yscale = 0.95, 550
                xscale2, yscale2 = 0.95, 545
            if stn == 'rr_09':
                xscale, yscale = 0.95, 400
                xscale2, yscale2 = 0.95, 375
            if stn == 'rr_10':
                xscale, yscale = 0.95, 136
                xscale2, yscale2 = 0.95, 131

            title = (('%s \n%s')
                     % (stn, cascade_level2))

            in_title = ''

            axes[k, 1].bar(data_dict2[stn]['fct'][0][0][0],
                           data_dict2[stn]['fct'][0][0][1],
                           align='center',
                           width=ratio_bars,
                           alpha=0.5,
                           linewidth=line_width,
                           color='blue',
                           label='Original values',
                           edgecolor='darkblue')
            axes[k, 1].bar(data_dict2[stn]['fct'][0][1][0]-w,
                           data_dict2[stn]['fct'][0][1][1],
                           align='center',
                           width=ratio_bars*0.8,
                           alpha=0.7,
                           linewidth=line_width,
                           color='red',
                           label='Basic model',
                           edgecolor='darkred')
            axes[k, 1].bar(data_dict2[stn]['fct'][0][2][0]+w,
                           data_dict2[stn]['fct'][0][2][1],
                           align='center',
                           width=ratio_bars*0.6,
                           alpha=0.5,
                           linewidth=line_width,
                           color='lime',
                           label='Dependent model',
                           edgecolor='darkgreen')
            axes[k, 1].set_axisbelow(True)
            axes[k, 1].yaxis.grid(color='gray',
                                  linestyle='dashed',
                                  linewidth=line_width,
                                  alpha=0.2)
            axes[k, 1].xaxis.grid(color='gray',
                                  linestyle='dashed',
                                  linewidth=line_width,
                                  alpha=0.2)
        elif lorenz is True:

            axes[k, 1].set_ylim([0, 1])

            xscale, yscale = 0.13, 0.85
            xscale2, yscale2 = 0.13, 0.81

            title = (('%s \n%s')
                     % (stn, cascade_level2))

            in_title = ''

            axes[k, 1].scatter(data_dict2[stn]['fct'][0][0][0],
                               data_dict2[stn]['fct'][0][0][1],
                               color='b',
                               marker='o',
                               alpha=0.7,
                               s=marker_size*0.7,
                               label='Observed rainfall')
            axes[k, 1].scatter(data_dict2[stn]['fct'][0][1][0],
                               data_dict2[stn]['fct'][0][1][1],
                               color='r',
                               marker='*',
                               s=marker_size*0.9,
                               alpha=0.5,
                               label='Basic model')
            axes[k, 1].scatter(data_dict2[stn]['fct'][0][2][0],
                               data_dict2[stn]['fct'][0][2][1],
                               color='g',
                               marker='+',
                               s=marker_size*0.8,
                               alpha=0.5,
                               label='Dependent model')
        elif cdf_max is True:

            axes[k, 1].set_ylim([0, 1])

            xscale, yscale = 25.68, 0.08
            xscale2, yscale2 = 25.68, 0.065
            title = (('%s \n%s')
                     % (stn, cascade_level2))

            in_title = ''

            axes[k, 1].plot(data_dict2[stn]['fct'][0][0][0],
                            data_dict2[stn]['fct'][0][0][1],
                            color='b',
                            alpha=0.6,
                            label='Observed rainfall',
                            linewidth=5)
            axes[k, 1].plot(data_dict2[stn]['fct'][0][1][0],
                            data_dict2[stn]['fct'][0][1][1],
                            color='r',
                            alpha=0.5,
                            label='Basic model',
                            linewidth=5)
            axes[k, 1].plot(data_dict2[stn]['fct'][0][2][0],
                            data_dict2[stn]['fct'][0][2][1],
                            color='g',
                            alpha=0.5,
                            label='Dependent model',
                            linewidth=5)
            axes[k, 1].set_xlim([np.min(
                [data_dict2[stn]['fct'][0][0][0],
                 data_dict2[stn]['fct'][0][1][0],
                 data_dict2[stn]['fct'][0][2][0]]),
                35])
            df_kls_test_2.loc['orig_bas', stn] = KOL(
                data_dict2[stn]['fct'][0][0][0],
                data_dict2[stn]['fct'][0][1][0])[0]
            df_kls_test_2.loc['orig_dpt', stn] = KOL(
                data_dict2[stn]['fct'][0][0][0],
                data_dict2[stn]['fct'][0][2][0])[0]
            df_kls_test_2.loc['bas_dpt', stn] = KOL(
                data_dict2[stn]['fct'][0][1][0],
                data_dict2[stn]['fct'][0][2][0])[0]

        axes[k, 1].text(xscale, yscale,
                        title,
                        fontsize=font_size_title)

        axes[k, 1].text(xscale2, yscale2,
                        in_title,
                        fontsize=font_size_title)

        if k == 2:
            axes[k, 1].set_xlabel(xlabel,
                                  fontsize=font_size_axis)
            axes[k, 1].tick_params(axis='x',
                                   labelsize=font_size_axis)
    plt.legend(bbox_to_anchor=(-1.25, -0.25, 2.25, .0502),
               ncol=4,
               fontsize=font_size_title*1.15,
               mode="expand", borderaxespad=0.)

    plt.subplots_adjust(hspace=0.25, wspace=0.25, top=0.85)

    plt.savefig(os.path.join(out_dir_all,
                             'DE' + var +
                             save_format),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

# =============================================================================
#
# =============================================================================


if weights:
    # call fct to get files and find W for level one and level two
    dict1 = getHistWeights(dfs_files_L1, in_data_beta_df_file_L1,
                           cascade_level_1)
    dict2 = getHistWeights(dfs_files_L2, in_data_beta_df_file_L2,
                           cascade_level_2)

    plotdictData(dict1, dict2,
                 cascade_level_1, cascade_level_2,
                 '0 < W < 1',
                 'Probability density values', 'weights_round')

    print('done plotting hist of weights and fitted beta distribution')
    raise Exception


if rain_weights:
    wr1 = plotScatterWeightsRainfall(dfs_files_L1, cascade_level_1)
    wr2 = plotScatterWeightsRainfall(dfs_files_L2, cascade_level_2)

    plotdictData(wr1, wr2,
                 cascade_level_1, cascade_level_2,
                 '0 $\leq$ W $\leq$ 1',
                 'Oberved Rainfall', 'Rweights2')

    print('done plotting scatter of rainfall and weights ')
    raise Exception

# =============================================================================
# Plot Prob that P(W=0) or P(W=1) per Month
# =============================================================================


def plotProbMonth(prob_df_file1, prob_df_file2):

    out_figs_dir1 = os.path.join(out_dir,
                                 'out_subplots')
    fact = 2
    fig, (ax1, ax2) = plt.subplots(figsize=(60, 30), ncols=2,
                                   dpi=dpi, sharey=True)

    in_prob_df = pd.read_csv(prob_df_file1, sep=df_sep, index_col=0)
    in_prob_df = in_prob_df[in_prob_df >= 0.]
    x = np.array([(in_prob_df.index)])

    ax1.set_xticks(np.linspace(1, 12, 12))

    y_1 = np.array([(in_prob_df['P1 per Month'].values)])
    y_1 = y_1.reshape((x.shape))

    ax1.scatter(x, y_1, c='b', marker='X',
                s=marker_size,
                label=r'P($W_{1}$) = 1')

    y_0 = np.array([(in_prob_df['P0 per Month'].values)])
    y_0 = y_0.reshape((x.shape))

    ax1.scatter(x, y_0, c='r', marker='h',
                s=marker_size,
                label='P($W_{1}$) = 0')

    y_3 = np.array([(in_prob_df['P01 per month'].values)])
    y_3 = y_3.reshape((x.shape))

    ax1.scatter(x, y_3, c='darkgreen', marker='D',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1')
    ax1.yaxis.set_ticks(np.arange(0, 0.32, 0.05))
    ax1.tick_params(axis='x', labelsize=font_size_axis*fact)
    ax1.tick_params(axis='y', labelsize=font_size_axis*fact)

    ax1.set_xlabel(r'Month', fontsize=font_size_axis*2.3)
    ax1.set_ylabel('$P_{01}$', fontsize=font_size_axis*2.3)

    ax1.text(10.6, 0.292, 'Level one',
             fontsize=font_size_title*2.5)
    ax1.yaxis.labelpad = 25
    ax1.xaxis.labelpad = 25
# =============================================================================

    in_prob_df2 = pd.read_csv(prob_df_file2, sep=df_sep, index_col=0)
    in_prob_df2 = in_prob_df2[in_prob_df2 >= 0.]
    x2 = np.array([(in_prob_df2.index)])

    ax2.set_xticks(np.linspace(1, 12, 12))

    y_12 = np.array([(in_prob_df2['P1 per Month'].values)])
    y_12 = y_12.reshape((x2.shape))

    ax2.scatter(x2, y_12, c='b', marker='X',
                s=marker_size,
                label=r'P($W_{1}$) = 1')

    y_02 = np.array([(in_prob_df2['P0 per Month'].values)])
    y_02 = y_02.reshape((x.shape))

    ax2.scatter(x2, y_02, c='r', marker='h',
                s=marker_size,
                label='P($W_{1}$) = 0')

    y_32 = np.array([(in_prob_df2['P01 per month'].values)])
    y_32 = y_32.reshape((x2.shape))

    ax2.scatter(x2, y_32, c='darkgreen', marker='D',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1')

    ax2.set_xlabel(r'Month', fontsize=font_size_axis*2.3)
    ax2.tick_params(axis='x', labelsize=font_size_axis*fact)
    ax2.tick_params(axis='y', labelsize=font_size_axis*fact)
#    ax2.grid(color='k', linestyle='dotted', linewidth=0.01, alpha=0.5)

    ax2.text(10.6, 0.292, 'Level two',
             fontsize=font_size_title*2.5)
    ax2.yaxis.labelpad = 25
    ax2.xaxis.labelpad = 25
    plt.legend(bbox_to_anchor=(-1.05, -0.2, 2.05, .102),
               ncol=4,
               fontsize=font_size_title*2.5,
               mode="expand", borderaxespad=0.)

    plt.subplots_adjust(hspace=0.4, wspace=0.05, top=0.85)

    plt.savefig(os.path.join(out_figs_dir1,
                             r'P01perStationGermany2%s' % save_format),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

    plt.close('all')
    return


# call fct level 1 and level 2
if plotP01Month:
    plotProbMonth(in_data_prob_df_file_L1, in_data_prob_df_file_L2)
    print('done plotting seasonal effect on P01')

# =============================================================================
# Plot Prob that P(W=0) or P(W=1) per Station
# =============================================================================


def probStation(prob_stn_df_file1,
                prob_stn_df_file2):

    out_figs_dir2 = os.path.join(out_dir, 'out_subplots')
    fact = 2
    fig, (ax3, ax4) = plt.subplots(figsize=(60, 30), ncols=2,
                                   dpi=300, sharey=True)
    global in_df

    # read prob df file and select >= 0 values
    in_df = pd.read_csv(prob_stn_df_file1, sep=df_sep, index_col=0)
    in_df = in_df[in_df >= 0.]

    # for labeling x axis by station names
    u, x = np.unique([(in_df.index)], return_inverse=True)
    alpha = 0.85

    # plot P1 values
    y_1 = np.array([(in_df['P1'].values)])
    y_1 = y_1.reshape(x.shape)
    ax3.scatter(x, y_1, c='b', marker='X',
                s=marker_size,
                label=r'P($W_{1}$) = 1', alpha=alpha)

    # plot P0 values
    y_2 = np.array([(in_df['P0'].values)])
    y_2 = y_2.reshape(x.shape)
    ax3.scatter(x, y_2, c='r', marker='h',
                s=marker_size,
                label=r'P($W_{1}$) = 0', alpha=alpha)

    # plot P01 values
    y_3 = np.array([(in_df['P01'].values)])
    y_3 = y_3.reshape(x.shape)
    ax3.scatter(x, y_3, c='darkgreen', marker='D',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1', alpha=alpha)

    ax3.set(xticks=range(len(u)), xticklabels=u)

    ax3.yaxis.set_ticks(np.arange(0, 1, 0.05))

    plt.setp(ax3.get_xticklabels(), rotation='horizontal',
             fontsize=font_size_axis*fact)

    plt.setp(ax3.get_yticklabels(), rotation='horizontal',
             fontsize=font_size_axis*fact)

    ax3.set_ylabel('$P_{01}$', fontsize=font_size_axis*2.3)
    ax3.set_xlabel('Station ID', fontsize=font_size_axis*2.3)

    ax3.set_ylim([0, 1])
    ax3.tick_params(axis='x', labelsize=font_size_axis*2)
    ax3.tick_params(axis='y', labelsize=font_size_axis*2)
    ax3.yaxis.labelpad = 25
    ax3.xaxis.labelpad = 25
# =============================================================================
    # read prob df file and select >= 0 values
    in_df2 = pd.read_csv(prob_stn_df_file2, sep=df_sep, index_col=0)
    in_df2 = in_df2[in_df2 >= 0.]

    # for labeling x axis by station names
    u2, x2 = np.unique([(in_df2.index)], return_inverse=True)

    # plot P1 values
    y_2 = np.array([(in_df2['P1'].values)])
    y_2 = y_2.reshape(x2.shape)
    ax4.scatter(x2, y_2, c='b', marker='X',
                s=marker_size,
                label=r'P($W_{1}$) = 1', alpha=alpha)

    # plot P0 values
    y_22 = np.array([(in_df2['P0'].values)])
    y_22 = y_22.reshape(x.shape)
    ax4.scatter(x2, y_22, c='r', marker='h',
                s=marker_size,
                label=r'P($W_{1}$) = 0', alpha=alpha)

    # plot P01 values
    y_32 = np.array([(in_df2['P01'].values)])
    y_32 = y_32.reshape(x2.shape)
    ax4.scatter(x2, y_32, c='darkgreen', marker='D',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1', alpha=alpha)

    ax4.set(xticks=range(len(u2)), xticklabels=u2)

#    ax4.grid(color='k', linestyle='dotted', linewidth=0.01, alpha=0.5)
    ax4.tick_params(axis='x', labelsize=font_size_axis*2)
    ax4.tick_params(axis='y', labelsize=font_size_axis*2)
#    ax4.text(0.1, 0.305, 'Level two',
#             fontsize=font_size_title*2.5)
    plt.setp(ax4.get_xticklabels(), rotation='horizontal',
             fontsize=font_size_axis*fact)

    plt.setp(ax4.get_yticklabels(), rotation='horizontal',
             fontsize=font_size_axis*fact)

    ax4.set_xlabel('Station ID', fontsize=font_size_axis*2.3)
    ax4.yaxis.labelpad = 25
    ax4.xaxis.labelpad = 25
    plt.legend(bbox_to_anchor=(-1.05, -0.2, 2.05, .102),
               ncol=4,
               fontsize=font_size_title*3,
               mode="expand", borderaxespad=0.)

    plt.subplots_adjust(hspace=0.3, wspace=0.05, top=0.85)
    plt.savefig(os.path.join(out_figs_dir2,
                             r'P01perStationGermany%s' % save_format),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

    plt.close('all')
    return


# call fct Level 1 and Level 2
if plotP01Station:

    probStation(in_data_prob_stn_df_file_L1,
                in_data_prob_stn_df_file_L2)
    print('done plotting P0, P1, P01 for every station')
# =============================================================================
# PLOT Unbounded Model Volume Dependency
# =============================================================================


def getSimFiles(sim_data_dir):
    # get simulated files
    files_sim = []
    for r, dir_, f in os.walk(sim_data_dir):
        for fs in f:
            if fs.endswith('.csv'):
                files_sim.append(os.path.join(r, fs))
    return files_sim


# read df values of R_vals, W_vals and logLikelihood vals L1
dfs_files_P01_L1 = getSimFiles(unbounded_model_dir_L1)
dfs_files_P01_L2 = getSimFiles(unbounded_model_dir_L2)


def logRegression(r_vals, a, b):
    return np.array([1 - 1 / (1 + np.exp(-
                                         (np.array([a + b *
                                                    np.log10(r_vals)]))))])


def volumeDependacyP01_1(in_df_files, in_param_file, cascade_level):

    percentile = 0.02  # divide R values into classes and fill classes with W
    # min_w_nbrs = 20   # when calculating P01, min nbr of W to consider
#    global ds
    wanted_stns_data = {k: {'fct': [], 'data': []}
                        for k in wanted_stns_list}
#    global d_plot, d
    for station in wanted_stns_list:
        if station == 'rr_07':
            percentile = 0.15
            min_w_nbrs = 35
        if station == 'rr_09':
            percentile = 0.2
            min_w_nbrs = 15
        if station == 'rr_10':
            percentile = 0.30
            min_w_nbrs = 15

        for df_file in in_df_files:
            if fnmatch.fnmatch(df_file, '*.csv') and station in df_file:

                # read df_file: R_vals, W_vals, W_vals(0_1), L(teta)
                d = pd.read_csv(df_file, sep=df_sep, index_col=0)
                d.round(2)
                # calculate P01 as freq from observed R, W values
                '''
                    superimposed are the 'observed' values of P01 estimated:
                    by fitting to the observed values of W with in each third
                    percentile of R, plotted against the mean values of R
                    in these ranges.
                    '''
                # new df to plot P01 vs log_10R
                d_plot = pd.DataFrame(
                    index=np.arange(0, len(d.index), 1))

                # new cols for  R vals and W1 vals
                d_plot['R vals'] = d['R vals']
                d_plot['W_01'] = d['W 01']

                # define classes min and max R values
                r_min = min(d_plot['R vals'].values)
                r_max = max(d_plot['R vals'].values)
                # define classes width increase
                k_inc = percentile
                # find needed nbr of classes
                nbr_classes = int((r_max - r_min) / k_inc)

                # new dicts to hold klasses intervals
                klasses = {}

                # new dicts to hold klasses W values
                klassen_w_vals = {}

                # new dicts to hold klasses R values
                klassen_r_vals = {}

                # new dicts to hold klasses W01 values for P01 observed
                w_01 = {}

                # create new classes and lists to hold values
                for i in range(nbr_classes+1):
                    klasses[i] = [round(r_min+i*k_inc, 2),
                                  round(r_min+(1+i)*k_inc, 2)]
                    klassen_w_vals[i] = []
                    klassen_r_vals[i] = []
                    w_01[i] = []

                # go through values
                for val, w_val in zip(d_plot['R vals'].values,
                                      d_plot['W_01'].values):
                    # find Rvals and Wvals per class
                    for klass in klasses.keys():
                            # if R val is in class, append w_val r_val to class
                        if (min(klasses[klass]) <=
                                val <=
                                max(klasses[klass])):

                            klassen_w_vals[klass].append(w_val)
                            klassen_r_vals[klass].append(val)

                # find P01 as frequency per class
                for klass in klassen_w_vals.keys():

                    # if enough values per class
                    if len(klassen_w_vals[klass]) >= min_w_nbrs:
                        ct_ = 0
                        for w_ in klassen_w_vals[klass]:
                            # if w_val = 0, w=0 or w=1 ,
                            # elif w_val=1 then 0<w<1
                            # this is why use P01 = 1-sum(W01)/len(W01)
                            if w_ == 0:
                                ct_ += 1

                        w_01[klass].append(ct_ /
                                           len(klassen_w_vals[klass]))

                        # calculate mean of rainfall values of the class
                        w_01[klass].append(np.mean(np.
                                                   log10(
                                                       klassen_r_vals[klass])))

                # convert dict Class: [P01, Log(Rmean)] to df, Class as idx
                ds = pd.DataFrame.from_dict(w_01, orient='index')
                ds.sort_values(0, ascending=False, inplace=True)
                # count 0<w<1 for plotting it in title
                ct = 0
                for val in d_plot['W_01'].values:
                    if val == 0.:
                        ct += 1

                # plot observed P01, x=mean(log10(R_values)), y=(P01)
                wanted_stns_data[station]['data'].append([ds[1],
                                                          ds[0],
                                                          ct])

                # read df for logRegression parameters
                df_param = pd.read_csv(in_param_file,
                                       sep=df_sep,
                                       index_col=0)

                # x values = log10 R values
                x_vals = np.log10(d_plot['R vals'].values)

                # extract logRegression params from df
                a_ = df_param.loc[station, 'a']
                b_ = df_param.loc[station, 'b']

                # calculate logRegression fct
                y_vals = logRegression(
                    d_plot['R vals'].values, a_, b_)

                # plot x, y values of logRegression
                wanted_stns_data[station]['fct'].append([x_vals,
                                                         y_vals,
                                                         (a_, b_)])

    return wanted_stns_data


def volumeDependacyP01_2(in_df_files, in_param_file, cascade_level):

    # percentile = 0.02  # divide R values into classes and fill classes with W
    # min_w_nbrs = 20   # when calculating P01, min nbr of W to consider
    #    global ds
    wanted_stns_data = {k: {'fct': [], 'data': []}
                        for k in wanted_stns_list}
#    global d_plot, d
    for station in wanted_stns_list:
        if station == 'rr_07':
            percentile = 0.1
            min_w_nbrs = 35
        if station == 'rr_09':
            percentile = 0.45
            min_w_nbrs = 20
        if station == 'rr_10':
            percentile = 0.2
            min_w_nbrs = 10

        for df_file in in_df_files:
            if fnmatch.fnmatch(df_file, '*.csv') and station in df_file:

                # read df_file: R_vals, W_vals, W_vals(0_1), L(teta)
                d = pd.read_csv(df_file, sep=df_sep, index_col=0)
                d.round(2)
                # calculate P01 as freq from observed R, W values
                '''
                    superimposed are the 'observed' values of P01 estimated:
                    by fitting to the observed values of W with in each third
                    percentile of R, plotted against the mean values of R
                    in these ranges.
                    '''
                # new df to plot P01 vs log_10R
                d_plot = pd.DataFrame(
                    index=np.arange(0, len(d.index), 1))

                # new cols for  R vals and W1 vals
                d_plot['R vals'] = d['R vals']
                d_plot['W_01'] = d['W 01']

                # define classes min and max R values
                r_min = min(d_plot['R vals'].values)
                r_max = max(d_plot['R vals'].values)
                # define classes width increase
                k_inc = percentile
                # find needed nbr of classes
                nbr_classes = int((r_max - r_min) / k_inc)

                # new dicts to hold klasses intervals
                klasses = {}

                # new dicts to hold klasses W values
                klassen_w_vals = {}

                # new dicts to hold klasses R values
                klassen_r_vals = {}

                # new dicts to hold klasses W01 values for P01 observed
                w_01 = {}

                # create new classes and lists to hold values
                for i in range(nbr_classes+1):
                    klasses[i] = [round(r_min+i*k_inc, 2),
                                  round(r_min+(1+i)*k_inc, 2)]
                    klassen_w_vals[i] = []
                    klassen_r_vals[i] = []
                    w_01[i] = []

                # go through values
                for val, w_val in zip(d_plot['R vals'].values,
                                      d_plot['W_01'].values):
                    # find Rvals and Wvals per class
                    for klass in klasses.keys():
                        # if R val is in class, append w_val r_val to class
                        if (min(klasses[klass]) <=
                                val <=
                                max(klasses[klass])):

                            klassen_w_vals[klass].append(w_val)
                            klassen_r_vals[klass].append(val)

                # find P01 as frequency per class
                for klass in klassen_w_vals.keys():

                    # if enough values per class
                    if len(klassen_w_vals[klass]) >= min_w_nbrs:
                        ct_ = 0
                        for w_ in klassen_w_vals[klass]:
                            # if w_val = 0, w=0 or w=1 ,
                            # elif w_val=1 then 0<w<1
                            # this is why use P01 = 1-sum(W01)/len(W01)
                            if w_ == 0:
                                ct_ += 1

                        w_01[klass].append(ct_ /
                                           len(klassen_w_vals[klass]))

                        # calculate mean of rainfall values of the class
                        w_01[klass].append(np.mean(np.
                                                   log10(
                                                       klassen_r_vals[klass])))

                # convert dict Class: [P01, Log(Rmean)] to df, Class as idx
                ds = pd.DataFrame.from_dict(w_01, orient='index')
                ds.sort_values(0, ascending=False, inplace=True)
                # count 0<w<1 for plotting it in title
                ct = 0
                for val in d_plot['W_01'].values:
                    if val == 0.:
                        ct += 1

                # plot observed P01, x=mean(log10(R_values)), y=(P01)
                wanted_stns_data[station]['data'].append([ds[1],
                                                          ds[0],
                                                          ct])

                # read df for logRegression parameters
                df_param = pd.read_csv(in_param_file,
                                       sep=df_sep,
                                       index_col=0)

                # implement logRegression fct

                # x values = log10 R values
                x_vals = np.log10(d_plot['R vals'].values)

                # extract logRegression params from df
                a_ = df_param.loc[station, 'a']
                b_ = df_param.loc[station, 'b']

                # calculate logRegression fct
                y_vals = logRegression(
                    d_plot['R vals'].values, a_, b_)

                # plot x, y values of logRegression
                wanted_stns_data[station]['fct'].append([x_vals,
                                                         y_vals,
                                                         (a_, b_)])

    return wanted_stns_data


if dependency:
    # call fct Level one and two

    dictlg1 = volumeDependacyP01_1(dfs_files_P01_L1,
                                   params_file_L1, cascade_level_1)
    dictlg2 = volumeDependacyP01_2(dfs_files_P01_L2,
                                   params_file_L2, cascade_level_2)
    plotdictData(dictlg1, dictlg2,
                 cascade_level_1, cascade_level_2,
                 'log$_{10}$ R', 'P$_{01}$', 'dependency_r')
    print('done plotting the volume dependency of P01')
    raise Exception

# =============================================================================
# Model EVALUATION
# =============================================================================
# get the files, all simulations and just one simulation
dfs_files_sim_01 = getSimFiles(in_dfs_simulation_01)
dfs_files_sim_02 = getSimFiles(in_dfs_simulation_02)

dfs_files_sim = getSimFiles(in_dfs_simulation)
dfs_files_sim_2 = getSimFiles(in_dfs_simulation_2)


def compareHistRain(in_df_orig_vals_file,
                    in_df_simulation_files,
                    cascade_level):
    '''
        input: original precipitaion values
                simuated precipitation values, baseline and unbounded
                cascade level
        output: subplot of baseline and unbounded model vs orig vals
                np.log10(values) is plotted
    '''

    wanted_stns_data = {k: {'fct': [], 'data': []}
                        for k in wanted_stns_list}

    # define bins and centering of bars
    bins2 = np.arange(-0.6, 1.5, 0.05)
    center2 = (bins2[:-1] + bins2[1:]) / 2

    # read df orig values
    in_df_orig_vals = pd.read_csv(in_df_orig_vals_file,
                                  sep=df_sep, index_col=0)

    for station in (in_df_orig_vals.columns):

        for i, df_file in enumerate(in_df_simulation_files):

            for stn_wanted in wanted_stns_list:
                if station == stn_wanted:
                    if station in df_file:

                        # read file as dataframe
                        df_sim_vals = pd.read_csv(df_file,
                                                  sep=df_sep,
                                                  index_col=0)
                        df_sim_vals.round(2)
                        # start going through index of station data
                        tx_int = df_sim_vals.index.intersection(
                            in_df_orig_vals[station].index)
                        df_sim_vals['orig vals'] =\
                            in_df_orig_vals[station].loc[tx_int]

                        # plot hist for original vals
                        hist0, bins =\
                            np.histogram(np.log10(df_sim_vals[
                                'orig vals'].values),
                                bins=bins2, range=(-0.6, 1.5),
                                normed=norme_it)

                        # extract baseline values from simulated file
                        hist1, bins1 = np.histogram(
                                np.log10(df_sim_vals
                                         ['baseline rainfall %s'
                                          % cascade_level].values),
                                bins=bins2,
                                range=(-0.6, 1.5),
                                normed=norme_it)

                        # extract unbounded values from simulated file
                        hist2, bins2 =\
                            np.histogram(np.log10(
                                         df_sim_vals[
                                             'unbounded rainfall %s'
                                             % cascade_level].values),
                                         bins=bins2,
                                         range=(-0.6, 1.5),
                                         normed=norme_it)

                        wanted_stns_data[station][
                            'fct'].append([(center2, hist0),
                                           (center2, hist1),
                                           (center2, hist2)])

    return wanted_stns_data

# call fct


if histogram:
    dictev1 = compareHistRain(in_df_30min_orig, dfs_files_sim_01,
                              cascade_level_1)
    dictev2 = compareHistRain(in_df_15min_orig, dfs_files_sim_02,
                              cascade_level_2)

    plotdictData(dictev1, dictev2,
                 cascade_level_1, cascade_level_2,
                 '$log_{10}$ (R)',
                 'Frequency', 'hists')

    print('done plotting the results of the simulations')

# =============================================================================
#
# =============================================================================


def buildDFSimulations(orig_vals_df, in_df_simulation, cascade_level):
    '''
    idea: build one df from all simulations for each model
    '''

    global in_df_orig, df_all_basic_sims, df_basic_sim

    # read df orig values and make index a datetime obj
    in_df_orig = pd.read_csv(orig_vals_df, sep=df_sep, index_col=0)

    in_df_orig.index = pd.to_datetime(in_df_orig.index,
                                      format=date_format)
    # df to hold original values
    df_orig_vals = pd.DataFrame(columns=in_df_orig.columns)

    for station in wanted_stns_list:
        for i, df_file in enumerate(in_df_simulation):

            if station in df_file:

                    # read simulations file
                df_sim_vals = pd.read_csv(df_file,
                                          sep=df_sep,
                                          index_col=0)

                df_sim_vals.index = pd.to_datetime(df_sim_vals.index,
                                                   format=date_format)
                # intersect original with suimulations
                idx_orig_sim = in_df_orig.index.intersection(
                    df_sim_vals.index)

                # new df to hold sorted simulations per idx
                df_all_basic_sims = pd.DataFrame(index=idx_orig_sim)
                df_all_depend_sims = pd.DataFrame(index=idx_orig_sim)
                break

        for i, df_file in enumerate(in_df_simulation):

            if station in df_file:
                # read simulations file
                df_sim_vals = pd.read_csv(df_file,
                                          sep=df_sep,
                                          index_col=0)

                df_sim_vals.index = pd.to_datetime(df_sim_vals.index,
                                                   format=date_format)
                # intersect original with suimulations
                idx_orig_sim = in_df_orig.index.intersection(
                    df_sim_vals.index)
                df_orig_vals = in_df_orig.loc[idx_orig_sim,
                                              station]
                # append results to df simulations
                df_basic_vals = df_sim_vals.loc[idx_orig_sim,
                                                'baseline rainfall %s'
                                                % cascade_level].values
                df_depdent_sim = df_sim_vals.loc[idx_orig_sim,
                                                 'unbounded rainfall %s'
                                                 % cascade_level].values

                df_all_basic_sims[i] = df_basic_vals
                df_all_depend_sims[i] = df_depdent_sim

        df_orig_vals.to_csv(os.path.join(out_dir_all,
                                         '%s_%s_orig_vals.csv'
                                         % (station, cascade_level)),
                            sep=df_sep, float_format='%0.2f')
        df_all_basic_sims.to_csv(os.path.join(out_dir_all,
                                              '%s_%s_basic_sim.csv'
                                              % (station, cascade_level)),
                                 sep=df_sep, float_format='%0.2f')
        df_all_depend_sims.to_csv(os.path.join(out_dir_all,
                                               '%s_%s_depend_sim.csv'
                                               % (station, cascade_level)),
                                  sep=df_sep, float_format='%0.2f')


# call function level one and level two
if buildDFsim:

    buildDFSimulations(
        in_df_30min_orig, dfs_files_sim, cascade_level_1)
    buildDFSimulations(
        in_df_15min_orig, dfs_files_sim_2, cascade_level_2)
    print('done plotting the mean of all the simulations')

# =============================================================================
# import dfs of all simulations
# =============================================================================
df_bs = os.path.join(out_dir_all, 'rr_07_Level one_basic_sim.csv')
df_de = os.path.join(out_dir_all, 'rr_07_Level one_depend_sim.csv')
df_lo = os.path.join(out_dir_all, 'rr_07_Level one_orig_vals.csv')

df_bs2 = os.path.join(out_dir_all, 'rr_07_Level two_basic_sim.csv')
df_de2 = os.path.join(out_dir_all, 'rr_07_Level two_depend_sim.csv')
df_lo2 = os.path.join(out_dir_all, 'rr_07_Level two_orig_vals.csv')

#df_bs1 = os.path.join(out_dir_all, 'rr_09_Level one_basic_sim.csv')
#df_de1 = os.path.join(out_dir_all, 'rr_09_Level one_depend_sim.csv')
#df_lo1 = os.path.join(out_dir_all, 'rr_09_Level one_orig_vals.csv')
#
#df_bs21 = os.path.join(out_dir_all, 'rr_09_Level two_basic_sim.csv')
#df_de21 = os.path.join(out_dir_all, 'rr_09_Level two_depend_sim.csv')
#df_lo21 = os.path.join(out_dir_all, 'rr_09_Level two_orig_vals.csv')
#
#df_bs31 = os.path.join(out_dir_all, 'rr_10_Level one_basic_sim.csv')
#df_de31 = os.path.join(out_dir_all, 'rr_10_Level one_depend_sim.csv')
#df_lo31 = os.path.join(out_dir_all, 'rr_10_Level one_orig_vals.csv')
#
#df_bs32 = os.path.join(out_dir_all, 'rr_10_Level two_basic_sim.csv')
#df_de32 = os.path.join(out_dir_all, 'rr_10_Level two_depend_sim.csv')
#df_lo32 = os.path.join(out_dir_all, 'rr_10_Level two_orig_vals.csv')

assert df_bs
assert df_de
assert df_lo

assert df_bs2
assert df_de2
assert df_lo2

# =============================================================================
#
# =============================================================================

# select one year to plot
'''
#start_date = '2013-07-29 06:00:00'
#end_date = '2013-07-29 18:00:00'

#start_date = '2013-07-24 00:00:00'
#end_date = '2013-07-25 00:00:00'

#start_date = '2011-08-04 00:00:00'
#end_date = '2011-08-06 00:00:00'
'''
start_date = '2014-05-21 04:00:00'
end_date = '2017-10-08 13:00:00'


def sliceDF(data_frame, start, end):

    # slice data for selecting certain events
    mask = (data_frame.index > start) &\
        (data_frame.index <= end)
    data_frame = data_frame.loc[mask]
    return data_frame


def readDFsObsSim(basic_df_vals,
                  depdt_df_vals,
                  orig_df_vals):

    _df_basic = pd.read_csv(basic_df_vals, sep=df_sep, index_col=0,
                            engine='python')
    _df_basic.index = pd.to_datetime(
        _df_basic.index, format=date_format)

    # read df dependent model all simulations
    _df_depdnt = pd.read_csv(depdt_df_vals,
                             sep=df_sep, index_col=0, engine='python')
    _df_depdnt.index = pd.to_datetime(_df_depdnt.index,
                                      format=date_format)

    # read df original values
    _df_orig = pd.read_csv(orig_df_vals, sep=df_sep,
                           header=None, index_col=0)
    _df_orig.fillna(0.1, inplace=True)
    _df_orig.index = pd.to_datetime(_df_orig.index,
                                    format=date_format)

    return _df_basic, _df_depdnt, _df_orig


def pltCdfSimulations(in_df_basic_simulations,
                      in_df_dependt_simulations,
                      in_df_orig_vals,
                      fig_name,
                      ylabel):

    global df_sorted_basic, _df_depdnt, df_cdf,\
        df_sorted_depdnt, _df_orig, x_axis_00

    # def fig and subplots
    fig, ax = plt.subplots(figsize=fig_size)

    # read df basic model all simulations
    _df_basic, _df_depdnt, _df_orig = readDFsObsSim(in_df_basic_simulations,
                                                    in_df_dependt_simulations,
                                                    in_df_orig_vals)

    # slice df to get one event
    _df_basic = sliceDF(_df_basic, start_date, end_date)
    _df_orig = sliceDF(_df_orig, start_date, end_date)
    _df_depdnt = sliceDF(_df_depdnt, start_date, end_date)

    # new df to hold cdf values per index per model as column
    df_cdf = pd.DataFrame(index=_df_basic.index)

    # extract original vals in a df
    orig_vals = _df_orig.values

    # def a list of nbrs and strs for desired percentiles for cdf
    percentages_nbr = [5, 25, 50, 75, 95]
    percentages_strg = ['5', '25', '50', '75', '95']
    # def list of colors
    colors = ['r', 'b', 'm', 'y', 'g']

    # go through idx of simulations and build cdf for each date
    for idx in _df_basic.index:

        # extract vals per date per model
        vals = _df_basic.loc[idx].values
        vals2 = _df_depdnt.loc[idx].values

        # go through the percentiles and build the cdf
        for percent_nbr, percent_str in zip(percentages_nbr,
                                            percentages_strg):

            df_cdf.loc[idx, '%s_percent_basic' % (percent_str)] =\
                np.percentile(vals, percent_nbr)
            df_cdf.loc[idx, '%s_percent_depdt' % (percent_str)] =\
                np.percentile(vals2, percent_nbr)

    # get idx for plotting make it numeric for plotting
    x_axis_0 = df_cdf.index
    t = x_axis_0.to_pydatetime()
    x_axis_00 = md.date2num(t)

    for percent_str, color in zip(percentages_strg,
                                  colors):
        # get values to plot
        basic_vals = df_cdf['%s_percent_basic' % percent_str].values
        dependt_vals = df_cdf['%s_percent_depdt' % percent_str].values

        ax.plot(x_axis_00, basic_vals, marker='+',
                color=color, alpha=0.7,
                label='%s_Percent_Basic_model' % (percent_str))

        ax.plot(x_axis_00, dependt_vals, marker='o',
                color=color, alpha=0.7,
                label='%s_Percent_Dependent_model' % (percent_str))

    # plot original values
    ax.plot(x_axis_00, orig_vals, marker='*',
            color='k', alpha=0.95,
            lw=2,
            label='Original_values')

    # customize plot
    ax.grid(True)
    xfmt = md.DateFormatter('%d-%m %H:%M')
    ax.tick_params(labelsize=font_size_title, rotation=0)
    ax.xaxis.set_major_formatter(xfmt)
    ax.set_xlabel('Time (hour-minutes)', fontsize=font_size_title,
                  rotation=0)
    ax.set_ylabel(ylabel, fontsize=font_size_title)
    plt.gcf().autofmt_xdate()

    ax.xaxis.labelpad = 20
    ax.yaxis.labelpad = 20

    plt.setp(ax.xaxis.get_majorticklabels(), rotation=0)
    plt.setp(ax.yaxis.get_majorticklabels(), rotation=0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.01, 1.1, 0.99, .0502),
               ncol=4, mode="expand", borderaxespad=0.,
               fontsize=font_size_title)
    plt.savefig(os.path.join(out_dir_all,
                             'dpt_cdf_%s.pdf'
                             % (fig_name)),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

    return


if plotCdfSim:
    #    start_date = '2013-07-29 06:00:00'
    #    end_date = '2013-07-29 18:00:00'
    start_date = '2015-08-14 01:15:00'
    end_date = '2015-08-14 20:00:00'

    pltCdfSimulations(df_bs, df_de, df_lo, 'aug_l10__',
                      'Rainfall (mm/30min)')
    pltCdfSimulations(df_bs2, df_de2, df_lo2, 'aug_l20__',
                      'Rainfall (mm/15min)')
    print('done plotting the cdf of all simulations')
    raise Exception

# =============================================================================
#
# =============================================================================


def plotBoxplot(in_df_basic_simulations,
                in_df_dependt_simulations,
                in_df_orig_vals,
                in_df_basic_simulations2,
                in_df_dependt_simulations2,
                in_df_orig_vals2,
                in_df_basic_simulations3,
                in_df_dependt_simulations3,
                in_df_orig_vals3,
                figname,
                ylabel):

    f, axarr = plt.subplots(2, figsize=(20, 10), dpi=dpi,
                            sharex='col', sharey='row')
    f.tight_layout()
    f.subplots_adjust(top=1.1)

    meanlineprops = dict(linestyle='--', linewidth=2., color='purple')
    global _df_basic, _df_depdnt, _df_orig, _df_2

    # read dfs
    _df_basic, _df_depdnt, _df_orig = readDFsObsSim(in_df_basic_simulations,
                                                    in_df_dependt_simulations,
                                                    in_df_orig_vals)

    _df_basic2, _df_depdnt2, _df_orig2 = readDFsObsSim(
        in_df_basic_simulations2,
        in_df_dependt_simulations2,
        in_df_orig_vals2)

    _df_basic3, _df_depdnt3, _df_orig3 = readDFsObsSim(
        in_df_basic_simulations3,
        in_df_dependt_simulations3,
        in_df_orig_vals3)

    # slice df
    _df_basic = sliceDF(_df_basic, start_date, end_date)
    _df_orig = sliceDF(_df_orig, start_date, end_date)
    _df_depdnt = sliceDF(_df_depdnt, start_date, end_date)

    _df_basic2 = sliceDF(_df_basic2, start_date, end_date)
    _df_orig2 = sliceDF(_df_orig2, start_date, end_date)
    _df_depdnt2 = sliceDF(_df_depdnt2, start_date, end_date)

    _df_basic3 = sliceDF(_df_basic3, start_date, end_date)
    _df_orig3 = sliceDF(_df_orig3, start_date, end_date)
    _df_depdnt3 = sliceDF(_df_depdnt3, start_date, end_date)

    idx1_h = pd.Series(_df_orig.index.format())

    data_sim1 = _df_basic.values.T
    data_sim2 = _df_depdnt.values.T
    orig_data = _df_orig.values

#    data_sim12 = _df_basic2.values.T
#    data_sim22 = _df_depdnt2.values.T
    orig_data2 = _df_orig2.values

#    data_sim13 = _df_basic3.values.T
#    data_sim23 = _df_depdnt3.values.T
    orig_data3 = _df_orig3.values

    inter_ = np.arange(1, _df_orig.size+1)
    inter_2 = np.arange(1, _df_orig2.size+1)
    inter_3 = np.arange(1, _df_orig3.size+1)

    axarr[0].boxplot(data_sim1,
                     showmeans=True,
                     meanline=True,
                     meanprops=meanlineprops)

#    axarr[0].boxplot(data_sim12,
#                     showmeans=True,
#                     meanline=True,
#                     meanprops=meanlineprops)

#    axarr[0].boxplot(data_sim13,
#                     showmeans=True,
#                     meanline=True,
#                     meanprops=meanlineprops)
    axarr[0].text(inter_[-2], 24,
                  'Basic model',
                  fontsize=font_size_title,
                  style='normal',
                  rotation=0)

    axarr[0].plot(inter_,
                  orig_data,
                  alpha=0.7,
                  marker='D',
                  color='r',

                  label='rr_07')

    axarr[0].plot(inter_2,
                  orig_data2,
                  alpha=0.7,
                  marker='H',
                  color='g',

                  label='rr_09')

    axarr[0].plot(inter_3,
                  orig_data3,
                  alpha=0.7,
                  marker='X',
                  color='b',

                  label='rr_10')

    axarr[0].yaxis.grid()
    axarr[0].set_ylabel(ylabel,
                        fontsize=font_size_title,
                        rotation=90)
    axarr[0].tick_params(labelsize=font_size_title)

    axarr[1].boxplot(data_sim2,
                     meanprops=meanlineprops,
                     showmeans=True, meanline=True)

#    axarr[1].boxplot(data_sim22,
#                     meanprops=meanlineprops,
#                     showmeans=True, meanline=True)
#    axarr[1].boxplot(data_sim23,
#                     meanprops=meanlineprops,
#                     showmeans=True, meanline=True)

    axarr[1].text(inter_[-2], 24,
                  'Dependent model',
                  style='normal',
                  fontsize=font_size_title, rotation=0)

    axarr[1].plot(inter_,
                  orig_data,
                  color='r',
                  marker='D',
                  alpha=0.7,

                  label='rr_07')

    axarr[1].plot(inter_2,
                  orig_data2,
                  color='g',
                  marker='H',
                  alpha=0.7,

                  label='rr_09')
    axarr[1].plot(inter_3,
                  orig_data3,
                  color='b',
                  marker='X',
                  alpha=0.7,

                  label='rr_10')
    axarr[1].yaxis.grid()

    axarr[1].get_xaxis().tick_bottom()
    axarr[1].get_yaxis().tick_left()

    axarr[1].set_ylabel('Rainfall (mm/15min)',
                        fontsize=font_size_title,
                        rotation=90)
    axarr[1].set_xlabel('',
                        fontsize=font_size_title,
                        rotation=0)
    axarr[1].set_xticklabels([i[-20:-3] for i in idx1_h],
                             rotation=5)

    axarr[1].tick_params(labelsize=font_size_title)
#    plt.legend(loc='best', fontsize=font_size_title)
#    f.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    axarr[1].xaxis.labelpad = 15
    axarr[0].yaxis.labelpad = 15
    axarr[1].yaxis.labelpad = 15
    plt.setp(axarr[1].xaxis.get_majorticklabels(), rotation=15)
    plt.setp(axarr[0].yaxis.get_majorticklabels(), rotation=0)
    plt.setp(axarr[1].yaxis.get_majorticklabels(), rotation=0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.01, 2.1, 0.99, .0502),
               ncol=4, mode="expand", borderaxespad=0.,
               fontsize=font_size_title)

    plt.savefig(os.path.join(out_dir_all,
                             'Boxplot_all_%s.pdf'
                             % (figname)),
                bbox_inches='tight',
                frameon=True,
                papertype='a4')

    return


if boxPlot:
    #    start_date = '2013-07-29 06:00:00'
    #    end_date = '2013-07-29 18:00:00'
    start_date = '2015-08-14 01:15:00'
    end_date = '2015-08-14 20:00:00'
#    start_date = '2016-06-24 22:15:00'
#    end_date = '2016-06-25 23:45:00'

    plotBoxplot(df_bs, df_de, df_lo,
                'Aug2015', 'Rainfall (mm/30min)')
    plotBoxplot(df_bs2, df_de2, df_lo2,
                'Aug2015', 'Rainfall (mm/15min)')

#    plotBoxplot(df_bs, df_de, df_lo,
#                df_bs1, df_de1, df_lo1,
#                df_bs31, df_de31, df_lo31,
#                'Aug2015', 'Rainfall (mm/30min)')
#    plotBoxplot(df_bs2, df_de2, df_lo2,
#                df_bs21, df_de21, df_lo21,
#                df_bs32, df_de32, df_lo32,
#                'Aug20152', 'Rainfall (mm/15min)')
    raise Exception
# =============================================================================
# calculate ranked histograms
# =============================================================================


def rankz(obs, ensemble):
    ''' Parameters
    ----------
    obs : array of observations
    ensemble : array of ensemble, with the first dimension being the
        ensemble member and the remaining dimensions being identical to obs
    Returns
    -------
    histogram data for ensemble.shape[0] + 1 bins.
    The first dimension of this array is the height of
    each histogram bar, the second dimension is the histogram bins.
         '''

    obs = obs
    ensemble = ensemble

    combined = np.vstack((obs[np.newaxis], ensemble))

    # print('computing ranks')
    ranks = np.apply_along_axis(lambda x: rankdata(x, method='min'),
                                0, combined)

    # print('computing ties')
    ties = np.sum(ranks[0] == ranks[1:], axis=0)
    ranks = ranks[0]
    tie = np.unique(ties)

    for i in range(1, len(tie)):
        index = ranks[ties == tie[i]]
        ranks[ties == tie[i]] = [np.random.randint(index[j], index[j] +
                                                   tie[i]+1, tie[i])
                                 [0] for j in range(len(index))]

    return np.histogram(ranks, bins=np.linspace(0.5, combined.shape[0]+0.5,
                                                combined.shape[0]+1))

# =============================================================================
# plot ranked Histograms
# =============================================================================


def plotRankedHistsSimulations(in_df_basic_simulations,
                               in_df_dependt_simulations,
                               in_df_orig_vals,
                               figname):

    global df_sorted_basic, _df_depdnt, df_cdf,\
        df_sorted_depdnt, _df_orig, x_axis_00

    _df_basic, _df_depdnt, _df_orig = readDFsObsSim(in_df_basic_simulations,
                                                    in_df_dependt_simulations,
                                                    in_df_orig_vals)

    # slice DF
    _df_basic = sliceDF(_df_basic, start_date, end_date)
    _df_orig = sliceDF(_df_orig, start_date, end_date)
    _df_depdnt = sliceDF(_df_depdnt, start_date, end_date)

    if _df_depdnt.shape[0] != _df_orig.shape[0]:
        _df_depdnt = _df_depdnt[1:]

    if _df_basic.shape[0] != _df_orig.shape[0]:
        _df_basic = _df_basic[1:]

    obs = _df_orig.values
    ensemble = np.array([_df_basic.values]).T
    ensemble2 = np.array([_df_depdnt.values]).T

    result = rankz(obs, ensemble)
    result2 = rankz(obs, ensemble2)

    x_line = np.arange(0, 101, 1)
    y_line = 0.*x_line + len(obs)/100.

    fig, ax = plt.subplots(figsize=(20, 10), dpi=dpi)
    ax.plot(x_line, y_line, color='r', linestyle='--', alpha=0.89,
            label='Expected uniform distribution of ranks')

    ax.bar(range(1, ensemble.shape[0]+2), result[0],
           color='darkblue', alpha=0.5, label='Basic Model',
           width=2)

    ax.bar(range(1, ensemble2.shape[0]+2), result2[0],
           color='g', alpha=0.35, label='Dependent Model',
           width=2)
#    ax.set_xlim([3, 998])
#    ax.set_ylim([1, 40])
    ax.grid(color='gray',
            linestyle='dashed',
            linewidth=line_width*0.1,
            alpha=0.2)
    ax.set_xlabel('Simulation Number',
                  fontsize=font_size_axis)
    ax.tick_params(axis='x',
                   labelsize=font_size_axis)
    ax.set_facecolor('w')
    ax.set_ylabel('Frequency',
                  fontsize=font_size_axis)
    ax.tick_params(axis='y',
                   labelsize=font_size_axis)

    plt.legend(loc=1, fontsize=font_size_title)

    plt.savefig(os.path.join(out_dir_all,
                             'rr_07_ensemble22_%s.pdf' % (figname)),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')


if rankedHist:
    plotRankedHistsSimulations(df_bs, df_de, df_lo, 'levelone')
    plotRankedHistsSimulations(df_bs2, df_de2, df_lo2, 'leveltwo')
    raise Exception
# =============================================================================
#
# =============================================================================


def get_lag_ser(in_ser_raw, lag=0):
    in_ser = in_ser_raw.copy()
    # shift time for simulated values
    if lag < 0:
        in_ser.values[:lag] = in_ser.values[-lag:]
        in_ser.values[lag:] = np.nan
    elif lag > 0:
        in_ser.values[lag:] = in_ser.values[:-lag]
        in_ser.values[:lag] = np.nan
    return in_ser


lags = [i for i in range(-3, 4)]  # 3 shifts


def plotShiftedDataCorr(in_df_basic_simulations,
                        in_df_dependt_simulations,
                        in_df_orig_vals,
                        in_df_basic_simulations2,
                        in_df_dependt_simulations2,
                        in_df_orig_vals2,
                        model_name,
                        time_shifts,
                        time1,
                        time2):
    df_corr_1 = pd.DataFrame(index=[k for k in time_shifts])
    df_corr_2 = pd.DataFrame(index=[k for k in time_shifts])

    # read dfs
    _df_basic, _df_depdnt, _df_orig =\
        readDFsObsSim(in_df_basic_simulations,
                      in_df_dependt_simulations,
                      in_df_orig_vals)
    # read dfs
    _df_basic2, _df_depdnt2, _df_orig2 =\
        readDFsObsSim(in_df_basic_simulations2,
                      in_df_dependt_simulations2,
                      in_df_orig_vals2)

    # get the mean of all simulations
    df_mean_sim = pd.DataFrame()
    df_mean_sim2 = pd.DataFrame()

    for idx in _df_basic.index:

        vals = _df_basic.loc[idx].values
        vals2 = _df_depdnt.loc[idx].values
        df_mean_sim.loc[idx, '50_percent_basic'] = np.percentile(
            vals, 50)
        df_mean_sim.loc[idx, '50_percent_depdnt'] = np.percentile(
            vals2, 50)

    for idx in _df_basic2.index:

        vals12 = _df_basic2.loc[idx].values
        vals22 = _df_depdnt2.loc[idx].values
        df_mean_sim2.loc[idx, '50_percent_basic'] = np.percentile(
            vals12, 50)
        df_mean_sim2.loc[idx, '50_percent_depdnt'] = np.percentile(
            vals22, 50)

    # max val for [Timestamp('2013-07-24 12:15:00')]

    # shift and plot the scatter plots
    for shift in time_shifts:

        print('scatter plots, simulations shifted: %d' % shift)
        shifted_sr = get_lag_ser(df_mean_sim, shift)
        shifted_sr.dropna(inplace=True)
        orig_stn = _df_orig.loc[shifted_sr.index]
        orig_stn = orig_stn[orig_stn >= 0]
        orig_stn.dropna(inplace=True)

        simstn = shifted_sr.loc[orig_stn.index, '50_percent_basic']
        simstn = simstn[simstn >= 0]
        simstn.dropna(inplace=True)

        simstn2 = shifted_sr.loc[orig_stn.index, '50_percent_depdnt']
        simstn2 = simstn2[simstn2 >= 0]
        simstn2.dropna(inplace=True)

        orig_stn = orig_stn.loc[simstn.index]
        # pearson and spearman correlation
        rho1 = spr(orig_stn.values, simstn.values)[0]
        rho2 = spr(orig_stn.values, simstn2.values)[0]
        df_corr_1.loc[shift, 'ro1'] = rho1
        df_corr_1.loc[shift, 'ro2'] = rho2
        print(shift, rho1, rho2)

        print('scatter plots, simulations shifted: %d' % shift)
        shifted_sr2 = get_lag_ser(df_mean_sim2, shift)
        shifted_sr2.dropna(inplace=True)
        orig_stn2 = _df_orig2.loc[shifted_sr2.index]
        orig_stn2 = orig_stn2[orig_stn2 >= 0]
        orig_stn2.dropna(inplace=True)

        simstn22 = shifted_sr2.loc[orig_stn2.index,
                                   '50_percent_basic']
        simstn22 = simstn22[simstn22 >= 0]
        simstn22.dropna(inplace=True)

        simstn23 = shifted_sr2.loc[orig_stn2.index,
                                   '50_percent_depdnt']
        simstn23 = simstn23[simstn23 >= 0]
        simstn23.dropna(inplace=True)

        orig_stn2 = orig_stn2.loc[simstn22.index]
        # pearson and spearman correlation
        rho12 = spr(orig_stn2.values, simstn22.values)[0]
        rho22 = spr(orig_stn2.values, simstn23.values)[0]

        print(shift, rho12, rho22)
        df_corr_2.loc[shift, 'ro1'] = rho12
        df_corr_2.loc[shift, 'ro2'] = rho22
    df_corr_1.to_csv(os.path.join(out_dir_all,
                                  'spear_corr_1.csv'), sep=';')
    df_corr_2.to_csv(os.path.join(out_dir_all,
                                  'spear_corr_2.csv'), sep=';')


if shiftedOrigSimValsCorr:

    plotShiftedDataCorr(df_bs, df_de, df_lo,
                        df_bs2, df_de2, df_lo2,
                        'depbasicEM10_1',
                        lags, '(mm/30min)',
                        '(mm/15min)')


def plotShiftedData(in_df_basic_simulations,
                    in_df_dependt_simulations,
                    in_df_orig_vals,
                    in_df_basic_simulations2,
                    in_df_dependt_simulations2,
                    in_df_orig_vals2,
                    model_name,
                    time1,
                    time2):

    global _df_basic, _df_orig

    f, axarr = plt.subplots(2, figsize=fig_size, dpi=dpi)

    # read dfs
    _df_basic, _df_depdnt, _df_orig =\
        readDFsObsSim(in_df_basic_simulations,
                      in_df_dependt_simulations,
                      in_df_orig_vals)
    # read dfs
    _df_basic2, _df_depdnt2, _df_orig2 =\
        readDFsObsSim(in_df_basic_simulations2,
                      in_df_dependt_simulations2,
                      in_df_orig_vals2)

    # get the mean of all simulations
    df_mean_sim = pd.DataFrame()
    df_mean_sim2 = pd.DataFrame()

    for idx in _df_basic.index:

        vals = _df_basic.loc[idx].values
        vals2 = _df_depdnt.loc[idx].values
        df_mean_sim.loc[idx, '50_percent_basic'] = np.percentile(
            vals, 50)
        df_mean_sim.loc[idx, '50_percent_depdnt'] = np.percentile(
            vals2, 50)

    for idx in _df_basic2.index:

        vals12 = _df_basic2.loc[idx].values
        vals22 = _df_depdnt2.loc[idx].values
        df_mean_sim2.loc[idx, '50_percent_basic'] = np.percentile(
            vals12, 50)
        df_mean_sim2.loc[idx, '50_percent_depdnt'] = np.percentile(
            vals22, 50)

    # max val for [Timestamp('2013-07-24 12:15:00')]

    # shift and plot the scatter plots
    r_thre = 40
    min_thre = 0
    print('scatter plots, simulations shifted: %d')
    orig_stn = _df_orig
    orig_stn = orig_stn[(orig_stn <= r_thre) & (orig_stn > min_thre)]
    orig_stn.dropna(inplace=True)

    simstn = df_mean_sim['50_percent_basic']
    simstn = simstn[simstn <= r_thre]
#    simstn.dropna(inplace=True)

    simstn2 = df_mean_sim['50_percent_depdnt']
    simstn2 = simstn2[simstn2 <= r_thre]
#    simstn2.dropna(inplace=True)

    idx_intersct01 = orig_stn.index.intersection(simstn.index)
    idx_intersct02 = orig_stn.index.intersection(simstn2.index)

    orig_stn0 = orig_stn.loc[idx_intersct01]
    simstn = simstn.loc[idx_intersct01]

    orig_stn1 = orig_stn.loc[idx_intersct02]
    simstn2 = simstn2.loc[idx_intersct02]
    # pearson and spearman correlation
    rho1 = spr(orig_stn0.values, simstn.values)[0]
    rho2 = spr(orig_stn1.values, simstn2.values)[0]
#
#    print(rho1, rho2)

    print('scatter plots, simulations shifted: 2')
    orig_stn2 = _df_orig2
    orig_stn2 = orig_stn2[(orig_stn2 <= r_thre)
                          & (orig_stn2 > min_thre)]
    orig_stn2.dropna(inplace=True)

    simstn22 = df_mean_sim2['50_percent_basic']
    simstn22 = simstn22[simstn22 <= r_thre]

#    simstn22.dropna(inplace=True)

    simstn23 = df_mean_sim2['50_percent_depdnt']
    simstn23 = simstn23[simstn23 <= r_thre]
#    simstn23.dropna(inplace=True)

    idx_intersct21 = orig_stn2.index.intersection(simstn22.index)
    idx_intersct22 = orig_stn2.index.intersection(simstn23.index)

    orig_stn20 = orig_stn2.loc[idx_intersct21]
    simstn22 = simstn22.loc[idx_intersct21]

    orig_stn21 = orig_stn2.loc[idx_intersct22]
    simstn23 = simstn23.loc[idx_intersct22]

    # pearson and spearman correlation
    rho12 = spr(orig_stn20.values, simstn22.values)[0]
    rho22 = spr(orig_stn21.values, simstn23.values)[0]
#
#    reliability_diagrams(simstn22, orig_stn2, [0.25, 0.5, 0.95])
#    print(rho12, rho22)
    axarr[0].scatter(orig_stn0,
                     simstn,
                     marker='D',
                     s=marker_size/3,
                     alpha=0.5,
                     facecolors='none',
                     edgecolors='r',
                     label='Basic model, Spr. Corr.=%0.4f'
                     % (rho1))
    axarr[0].scatter(orig_stn1,
                     simstn2,
                     marker='+',
                     c='b',
                     s=marker_size/6,
                     alpha=0.5,
                     label='Dependent model, Spr. Corr.=%0.4f'
                     % (rho2))

    _min = min(orig_stn0.values.min(), simstn.values.min())
    _max = max(orig_stn0.values.max(), simstn.values.max())

    axarr[0].plot([_min, _max], [_min, _max],
                  c='k', linestyle='--',
                  alpha=0.4)

    axarr[0].set_xlim(-0.01, _max)
    axarr[0].set_ylim(-0.01, _max)
    axarr[0].set_xlabel('Original Rainfall Values %s' % time1,
                        fontsize=font_size_title)
    axarr[0].set_ylabel('Mean of all Simulated Rainfall Values %s'
                        % time1,
                        fontsize=font_size_title)
    axarr[0].grid(color='lightgrey',
                  linestyle='--',
                  linewidth=0.01)
    ####################
    axarr[1].scatter(orig_stn20,
                     simstn22,
                     alpha=0.5,
                     marker='D',
                     s=marker_size/3,
                     facecolors='none',
                     edgecolors='r',
                     label='Basic model, Spr. Corr.=%0.4f'
                     % (rho12))
    axarr[1].scatter(orig_stn21,
                     simstn23,
                     marker='+',
                     s=marker_size/6,
                     c='b',
                     alpha=0.5,
                     label='Dependent model, Spr. Corr.=%0.4f'
                     % (rho22))

    _min2 = min(orig_stn20.values.min(), simstn22.values.min())
    _max2 = max(orig_stn20.values.max(), simstn22.values.max())
#    axarr[1].axis('equal')
#    axarr[0].axis('equal')
    axarr[1].plot([_min2, _max2], [_min2, _max2],
                  c='k', linestyle='--',
                  alpha=0.5)

    axarr[1].set_xlim(-0.01, _max2)
    axarr[1].set_ylim(-0.01, _max2)
    axarr[1].tick_params(axis='x', labelsize=font_size_title)
    axarr[1].tick_params(axis='y', labelsize=font_size_title)
    axarr[0].tick_params(axis='x', labelsize=font_size_title)
    axarr[0].tick_params(axis='y', labelsize=font_size_title)

    axarr[1].set_xlabel('Original Rainfall Values %s' % time2,
                        fontsize=font_size_title)
    axarr[1].set_ylabel('Mean of all Simulated Rainfall Values %s'
                        % time2,
                        fontsize=font_size_title)
    axarr[1].grid(color='lightgrey',
                  linestyle='--',
                  linewidth=0.01)
    axarr[0].legend(loc=4, fontsize=font_size_legend)
    axarr[1].legend(loc=4, fontsize=font_size_legend)

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.savefig(os.path.join(out_dir_all,
                             (r'orig_vs_sim_shift_%s_2_0n.pdf')
                             % (model_name)),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')
    plt.close('all')


if plotShiftedOrigSimVals:

    plotShiftedData(df_bs,
                    df_de,
                    df_lo,
                    df_bs2,
                    df_de2,
                    df_lo2,
                    'depbasicrr07_2',
                    '(mm/30min)',
                    '(mm/15min)')
    raise Exception


def boundsSimulated(_df_basic,
                    _df_depdnt,
                    _df_orig,
                    _df_basic2,
                    _df_depdnt2,
                    _df_orig2,
                    cascade_level):
    '''
    idea: plot all simulations bounds, min and max simulated vs orig values
    '''
    f, axarr = plt.subplots(2, figsize=fig_size, dpi=dpi,
                            sharex='col', sharey='row')
    r_min = 5.0
    idx_intersct = _df_orig.index.intersection(_df_depdnt.index)
    _df_orig = _df_orig.loc[idx_intersct]
    _df_orig = _df_orig[_df_orig >= r_min]
    _df_basic = _df_basic[_df_basic >= r_min]
    _df_depdnt = _df_depdnt[_df_depdnt >= r_min]

    _df_orig2 = _df_orig2[_df_orig2 >= r_min]
    _df_basic2 = _df_basic2[_df_basic2 >= r_min]
    _df_depdnt2 = _df_depdnt2[_df_depdnt2 >= r_min]

    idx_intersct2 = _df_orig2.index.intersection(_df_depdnt2.index)
    _df_orig2 = _df_orig2.loc[idx_intersct2]

    x_axis2 = _df_orig.index
    t = x_axis2.to_pydatetime()
    x_axis2 = md.date2num(t)
    axarr[0].scatter(x_axis2,
                     _df_orig.values,
                     color='b',
                     marker='+',
                     s=marker_size,
                     alpha=0.8,
                     label='Original values')

    x_axis0 = _df_basic.index
    t = x_axis0.to_pydatetime()
    x_axis0 = md.date2num(t)

    axarr[0].scatter(x_axis0,
                     _df_basic.values,
                     facecolors='none',
                     edgecolors='r',
                     marker='o',
                     s=marker_size*0.6,
                     alpha=0.5,
                     label='Basic model')

    x_axis1 = _df_depdnt.index
    t = x_axis1.to_pydatetime()
    x_axis1 = md.date2num(t)

    axarr[0].scatter(x_axis1,
                     _df_depdnt.values,
                     facecolors='none',
                     edgecolors='g',
                     marker='D',
                     s=marker_size*0.2,
                     alpha=0.4,
                     label='Dependent model')

    axarr[0].grid(color='grey', axis='both',
                  linestyle='dashdot',
                  linewidth=0.05, alpha=0.5)

    axarr[0].set_ylabel('Rainfall (mm / 30min)',
                        fontsize=font_size_title,
                        rotation=90)
    axarr[0].tick_params(labelsize=font_size_title)
    axarr[0].yaxis.labelpad = 15

# =============================================================================
#
# =============================================================================
    x_axis22 = _df_orig2.index
    t = x_axis22.to_pydatetime()
    x_axis22 = md.date2num(t)
    axarr[1].scatter(x_axis22,
                     _df_orig2.values,
                     color='b',
                     marker='+',
                     s=marker_size,
                     alpha=0.8,
                     label='Original values')

    x_axis02 = _df_basic2.index
    t2 = x_axis02.to_pydatetime()
    x_axis02 = md.date2num(t2)

    axarr[1].scatter(x_axis02,
                     _df_basic2.values,
                     marker='o',
                     s=marker_size*0.6,
                     facecolors='none',
                     edgecolors='r',
                     alpha=0.5,
                     label='Basic model')

    x_axis12 = _df_depdnt2.index
    t2 = x_axis12.to_pydatetime()
    x_axis12 = md.date2num(t2)

    axarr[1].scatter(x_axis12,
                     _df_depdnt2.values,
                     facecolors='none',
                     edgecolors='g',
                     marker='D',
                     s=marker_size*0.2,
                     alpha=0.4,
                     label='Dependent model')

    axarr[1].grid(color='grey', axis='both',
                  linestyle='dashdot',
                  linewidth=0.05, alpha=0.5)

    axarr[1].tick_params(labelsize=font_size_title)

    axarr[1].set_ylabel('Rainfall (mm / 15min)',
                        fontsize=font_size_title,
                        rotation=90)

#    axarr[1].set_xlabel('Time', fontsize=font_size_title,
#                        rotation=-90)

    xfmt = md.DateFormatter('%d-%m')

    axarr[1].xaxis.set_major_formatter(xfmt)
    axarr[1].set_xticks(np.arange(x_axis1[0],
                                  x_axis1[-1]+1, 15))

    axarr[1].xaxis.labelpad = 15
    axarr[1].yaxis.labelpad = 15

    plt.gcf().autofmt_xdate()
#    plt.xticks(rotation=-90)
#    plt.yticks(rotation=-90)
    plt.setp(axarr[1].xaxis.get_majorticklabels(), rotation=0)
    plt.setp(axarr[0].yaxis.get_majorticklabels(), rotation=0)
    plt.setp(axarr[1].yaxis.get_majorticklabels(), rotation=0)
    plt.legend(loc='upper center', bbox_to_anchor=(0.01, 2.27, 0.99, .0502),
               ncol=4, mode="expand", borderaxespad=0.,
               fontsize=font_size_title)

    plt.savefig(os.path.join(out_dir_all,
                             r'%ssimrr092005%s' % (cascade_level,
                                                   save_format)),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')
    plt.close('all')

    return


start_date = '2015-01-01 00:00:00'
end_date = '2015-12-31 00:00:00'


def readOneDf(df_file_):
    _df_ = pd.read_csv(df_file_, sep=df_sep, index_col=0)

    try:
        _df_.index = pd.to_datetime(_df_.index, format=date_format)
    except Exception as msg:
        _df_.index = pd.to_datetime(
            _df_.index, format='%d.%m.%Y %H:%M')
    return _df_


if plotAllSim:
    station = 'rr_07'

    dbs1 = dfs_files_sim_01[0]
    dbs2 = dfs_files_sim_02[0]
    assert station in dbs1, 'error locating station'
    assert station in dbs2, 'error locating station'

    df_orig_vals_L1 = readOneDf(in_df_30min_orig)[station]
    df_orig_vals_L2 = readOneDf(in_df_15min_orig)[station]

    df_level_one = readOneDf(dbs1)
    df_level_two = readOneDf(dbs2)

    df_level_one_sliced = sliceDF(df_level_one, start_date, end_date)
    df_level_two_sliced = sliceDF(df_level_two, start_date, end_date)
    df_orig_vals_L1_silced = sliceDF(
        df_orig_vals_L1, start_date, end_date)
    df_orig_vals_L2_silced = sliceDF(
        df_orig_vals_L2, start_date, end_date)

    bs_vals = df_level_one_sliced['baseline rainfall Level one']
    db_vals = df_level_one_sliced['unbounded rainfall Level one']
    o1 = df_orig_vals_L1_silced

    bs_vals2 = df_level_two_sliced['baseline rainfall Level two']
    db_vals2 = df_level_two_sliced['unbounded rainfall Level two']
    o2 = df_orig_vals_L2_silced

    boundsSimulated(bs_vals, db_vals, o1,
                    bs_vals2, db_vals2, o2,
                    cascade_level_1)

    print('done plotting the bounds of the simulations')
    raise Exception
# ==============================================================================
# LORENZ Curves
# ==============================================================================

dfs_lorenz_L1_sim = getSimFiles(in_lorenz_df_L1_sim)
dfs_lorenz_L2_sim = getSimFiles(in_lorenz_df_L2_sim)

# to calculate gini coeffecient, call fct when plotting


def gini(list_of_values):
    height, area = 0, 0
    for value in list_of_values:
        height += value
        area += height - value / 2.
    fair_area = height * len(list_of_values) / 2.
    return (fair_area - area) / fair_area


def plotLorenzCurves(in_lorenz_vals_dir_sim,
                     cascade_level):
    '''
    cumulative frequency of rainy days (X)is
    plotted against associated precipitation amount (Y).
    '''
    wanted_stns_data = {k: {'fct': [], 'data': []}
                        for k in wanted_stns_list}

    original_files = []
    baseline_files = []
    unbounded_files = []

    for in_sim_file in in_lorenz_vals_dir_sim:

        if 'Orig' in in_sim_file:
            original_files.append(in_sim_file)

        if 'baseline' in in_sim_file and 'Orig' not in in_sim_file:
            baseline_files.append(in_sim_file)

        if 'unbounded' in in_sim_file:
            unbounded_files.append(in_sim_file)

    for station in wanted_stns_list:

        for orig_file, base_file, unbound_file in\
            zip(original_files,
                baseline_files, unbounded_files):

            if station in orig_file and\
                station in base_file and\
                    station in unbound_file:

                # start new fig

                in_lorenz_vlas_df_orig = pd.read_csv(orig_file,
                                                     sep=df_sep,
                                                     index_col=0)

                x_vals = in_lorenz_vlas_df_orig['X O']
                y_vals = in_lorenz_vlas_df_orig['Y O']

                in_lorenz_df_sim_base = pd.read_csv(base_file,
                                                    sep=df_sep,
                                                    index_col=0)

                x_vals_sim = in_lorenz_df_sim_base['X']
                y_vals_sim = in_lorenz_df_sim_base['Y']

                in_lorenz_df_sim_unbound = pd.read_csv(unbound_file,
                                                       sep=df_sep,
                                                       index_col=0)

                x_vals_sim_ = in_lorenz_df_sim_unbound['X']
                y_vals_sim_ = in_lorenz_df_sim_unbound['Y']

                wanted_stns_data[station]['fct'].append(
                    [(x_vals, y_vals),
                     (x_vals_sim, y_vals_sim),
                     (x_vals_sim_, y_vals_sim_)])

    return wanted_stns_data


# call fct level one and two
if lorenz:
    L1_orig_sim = plotLorenzCurves(dfs_lorenz_L1_sim,
                                   cascade_level_1)

    L2_orig_sim = plotLorenzCurves(dfs_lorenz_L2_sim,
                                   cascade_level_2)

    plotdictData(L1_orig_sim, L2_orig_sim,
                 cascade_level_1, cascade_level_2,
                 'Accumulated occurences', 'Rainfall contribution',
                 'lorenz')
    print('done plotting the Lorenz curves')


def distMaximums(in_df_orig_file, in_df_simu_files, cascade_level):
    '''
    Idea: read observed and simulated values
        select highest 20_30 values
        see how they are distributed
        compare observed to simulated

    '''
    wanted_stns_data = {k: {'fct': [], 'data': []}
                        for k in wanted_stns_list}

    # read df orig values
    in_df_orig = pd.read_csv(in_df_orig_file, sep=df_sep, index_col=0)
    in_df_orig.index = pd.to_datetime(
        in_df_orig.index, format=date_format)

    for station in wanted_stns_list:

        for i, df_file in enumerate(in_df_simu_files):

            if station in df_file:

                # new df to hold all simulated values, baseline and unbounded
                df_baseline = pd.DataFrame()
                df_unbounded = pd.DataFrame()

                df_max_baseline = pd.DataFrame()
                df_max_unbounded = pd.DataFrame()
                df_max_orig = pd.DataFrame()

                df_sim_vals = pd.read_csv(df_file,
                                          sep=df_sep,
                                          index_col=0)

                df_sim_vals.index = pd.to_datetime(df_sim_vals.index,
                                                   format=date_format)
                idx_intersct = df_sim_vals.index.intersection(
                    in_df_orig[station].index)

                df_sim_vals['orig vals'] = in_df_orig[station].loc[
                    idx_intersct]

                for idx in df_sim_vals.index:
                    try:

                        # for each idx,extract orig values
                        df_sim_vals.loc[idx, 'orig vals'] =\
                            in_df_orig[station].loc[idx]

                        # get values from each simulation
                        df_baseline.loc[idx, i] =\
                            df_sim_vals.loc[idx,
                                            'baseline rainfall %s'
                                            % cascade_level]

                        df_unbounded.loc[idx, i] =\
                            df_sim_vals.loc[idx,
                                            'unbounded rainfall %s'
                                            % cascade_level]

                    except KeyError as msg:
                        print(msg)
                        continue

                # sort values, to extract extremes
                df_max_baseline = df_baseline[i].\
                    sort_values(ascending=False,
                                kind='mergesort')
                df_max_unbounded = df_unbounded[i].\
                    sort_values(ascending=False,
                                kind='mergesort')
                df_max_orig = df_sim_vals['orig vals'].\
                    sort_values(ascending=False,
                                kind='mergesort')

                # extract extremes, what interest us : 20 vals
                y1, y2, y3 = df_max_orig[:20],\
                    df_max_baseline[:20],\
                    df_max_unbounded[:20]
#                y1, y2, y3 = df_max_orig,\
#                    df_max_baseline,\
#                    df_max_unbounded
#
                # Cumulative counts:
                x0 = np.concatenate([y1.values[::-1], y1.values[[0]]])
                y0 = (np.arange(y1.values.size+1)/len(y1.values))

                x1 = np.concatenate([y2.values[::-1], y2.values[[0]]])
                y1 = (np.arange(y2.values.size+1)/len(y2.values))

                x2 = np.concatenate([y3.values[::-1], y3.values[[0]]])
                y2 = (np.arange(y3.values.size+1)/len(y3.values))

                wanted_stns_data[station]['fct'].append(
                    [(x0, y0), (x1, y1), (x2, y2)])

    return wanted_stns_data


if cdf_max:
    max_l1 = distMaximums(
        in_df_30min_orig, dfs_files_sim_01, cascade_level_1)
    max_l2 = distMaximums(
        in_df_15min_orig, dfs_files_sim_02, cascade_level_2)
    df_kls_test_1 = pd.DataFrame(
        index=['orig_bas', 'orig_dpt', 'bas_dpt'])
    df_kls_test_2 = pd.DataFrame(
        index=['orig_bas', 'orig_dpt', 'bas_dpt'])

    plotdictData(max_l1, max_l2,
                 cascade_level_1, cascade_level_2,
                 'Rainfall Volume (mm)',
                 'Cumulative Distribution function',
                 'cdfmaxall40')

df_observations_df = sliceDF(readOneDf(df_lo),
                             '2014-05-23 03:00:00',
                             '2017-07-11 00:00:00')
df_simulations_df = sliceDF(readOneDf(df_de),
                            '2014-05-23 03:00:00',
                            '2017-07-11 00:00:00')

df_observations = df_observations_df.values
df_simulations = df_simulations_df.values
# Initialization of Measurement Error, only on diagonals
MeasurementError = 0.9  # Standard Deviation
Measurement_Error = np.zeros((df_observations.shape[0],
                              df_observations.shape[0]))

for i in range(df_observations.shape[0]):
    Measurement_Error[i, i] = MeasurementError**2
# Deviation of Model from Observations
deviations = np.zeros((df_simulations.shape[1], df_observations.shape[0]))

for i in range(df_observations.shape[0]):
    deviations[:, i] = df_observations[i] - df_simulations[i, :]

# Computation of Gaussian Likelihood: Weight according to the Deviation
weights_dev = np.zeros((1, df_simulations.shape[1]))

inv_me = np.linalg.inv(Measurement_Error)
# cs = (1 / (np.sqrt(2 * np.pi * np.exp(1))))** df_observations.shape[1]
cs = 1
for i in range(df_simulations.shape[1]):
    print('done with simulation nbr:', i)
    weights_dev[0, i] = (cs * np.exp(-0.5 *
                         np.matmul(np.matmul(deviations[i, :], inv_me),
                                   deviations[i, :])))
idx_max = weights_dev.argmax()
likelihood_max_df = df_simulations_df.loc[:, str(idx_max)]
plt.plot(df_observations_df, color='blue')
plt.plot(likelihood_max_df,
         color='red', alpha=0.5)
plt.savefig(os.path.join(out_dir_all, 'max_likelihood_bayes.png'))
STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))
