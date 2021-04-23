# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: EL Hachem Abbas, IWS
"""

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
from scipy.stats import beta
from matplotlib.dates import YearLocator,DateFormatter

import os
import timeit
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import fnmatch

plt.ioff()

print('\a\a\a\a Started on %s \a\a\a\a\n' % time.asctime())
START = timeit.default_timer() # to get the runtime of the program

main_dir = (r'X:\hiwi\El Hachem\CascadeModel_Daily_Hourly')
#main_dir = r'/home/abbas/Desktop/peru_cascade'

os.chdir(main_dir)

#def data dir
in_data_dir = os.path.join(main_dir,
                           r'Weights')
#==============================================================================
# Level ONE
#==============================================================================
cascade_level_1 = 'Level_one'

#in_df of P0 and P1 per month
in_data_prob_df_file_L1 = os.path.join(in_data_dir,
                                       'P1 P0 per month %s.csv'\
                                       %cascade_level_1)
#in_df of fitted beta dist params
in_data_beta_df_file_L1 = os.path.join(in_data_dir,
                                       r'bounded maximum likelihood %s.csv'\
                                       %cascade_level_1)
#in_df of P01 per stn
in_data_prob_stn_df_file_L1 = os.path.join(in_data_dir,
                                           r'Prob W P01 %s.csv'\
                                           %cascade_level_1)
#in_df of w_vals, R_wals and logRegression cols
unbounded_model_dir_L1 = os.path.join(in_data_dir,
                                      r'%s P01 volume dependancy'\
                                      %cascade_level_1)
#in_df of logistic regression params
params_file_L1 = os.path.join(unbounded_model_dir_L1,
                              r'%s log_regress params' %cascade_level_1,
                              r'loglikehood params.csv')

#in_df results of simulation for model evaluation
in_dfs_simulation = os.path.join(in_data_dir,
                                 r'%s model evaluation' %cascade_level_1)
#read original values, to compare to model
in_df_720min_orig = os.path.join(main_dir, r'resampled 720min.csv')

#read dfs holding the results of Lorenz curves of observed values
in_lorenz_df_L1 = os.path.join(in_data_dir,
                           r'%s Lorenz curves original' %cascade_level_1)

#read dfs holding the results of Lorenz curves of simulated values
in_lorenz_df_L1_sim = os.path.join(in_data_dir,
                           r'%s Lorenz curves simulations' %cascade_level_1)

#==============================================================================
# Level TWO ONE
#==============================================================================

cascade_level_2 = 'Level_two'

#in_df of P0 and P1 per month
in_data_prob_df_file_L2 = os.path.join(in_data_dir,
                                       'P1 P0 per month %s.csv'\
                                       %cascade_level_2)
#in_df of fitted beta dist params
in_data_beta_df_file_L2 = os.path.join(in_data_dir,
                                       r'bounded maximum likelihood %s.csv'\
                                       %cascade_level_2)
#in_df of P01 per stn
in_data_prob_stn_df_file_L2 = os.path.join(in_data_dir,
                                           r'Prob W P01 %s.csv'\
                                           %cascade_level_2)

#in_df of w_vals, R_wals and logRegression cols
unbounded_model_dir_L2 = os.path.join(in_data_dir,
                                      r'%s P01 volume dependancy'\
                                      %cascade_level_2)
#in_df of logistic regression params
params_file_L2 = os.path.join(unbounded_model_dir_L2,
                              r'%s log_regress params' %cascade_level_2,
                              r'loglikehood params.csv')
in_dfs_simulation_2 = os.path.join(in_data_dir,
                                 r'%s model evaluation' %cascade_level_2)

in_df_360min_orig = os.path.join(main_dir, r'resampled 360min.csv')

in_lorenz_df_L2 = os.path.join(in_data_dir,
                               r'%s Lorenz curves original' %cascade_level_2)

in_lorenz_df_L2_sim = os.path.join(in_data_dir,
                               r'%s Lorenz curves simulations' %cascade_level_2)

#==============================================================================
# level_three
#==============================================================================
cascade_level_3 = 'Level_three'

#in_df of P0 and P1 per month
in_data_prob_df_file_L3 = os.path.join(in_data_dir,
                                       'P1 P0 per month %s.csv'\
                                       %cascade_level_3)
#in_df of fitted beta dist params
in_data_beta_df_file_L3 = os.path.join(in_data_dir,
                                       r'bounded maximum likelihood %s.csv'\
                                       %cascade_level_3)
#in_df of P01 per stn
in_data_prob_stn_df_file_L3 = os.path.join(in_data_dir,
                                           r'Prob W P01 %s.csv'\
                                           %cascade_level_3)

#in_df of w_vals, R_wals and logRegression cols
unbounded_model_dir_L3 = os.path.join(in_data_dir,
                                      r'%s P01 volume dependancy'\
                                      %cascade_level_3)
#in_df of logistic regression params
params_file_L3 = os.path.join(unbounded_model_dir_L3,
                              r'%s log_regress params' %cascade_level_3,
                              r'loglikehood params.csv')

in_dfs_simulation_3 = os.path.join(in_data_dir,
                                 r'%s model evaluation' %cascade_level_3)

in_df_180min_orig = os.path.join(main_dir, r'resampled 180min.csv')

in_lorenz_df_L3 = os.path.join(in_data_dir,
                               r'%s Lorenz curves original' %cascade_level_3)

in_lorenz_df_L3_sim = os.path.join(in_data_dir,
                               r'%s Lorenz curves simulations' %cascade_level_3)


#==============================================================================
# create OUT dir and make sure all IN dir are correct
#==============================================================================
#def out_dir to hold plots
out_dir = os.path.join(main_dir,
                       r'Histograms_Weights')
if not os.path.exists(out_dir): os.mkdir(out_dir)

#make sure all defined directories exist
assert os.path.exists(in_data_dir), 'wrong data DF location'

#LEVEL ONE
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
assert os.path.exists(in_df_720min_orig),\
        'wrong orig DF location L1'
assert os.path.exists(in_lorenz_df_L1),\
         'wrong Lorenz Curves original Df L1'
assert os.path.exists(in_lorenz_df_L1_sim),\
         'wrong Lorenz Curves simu Df L1'

#LEVEL TWO
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
assert os.path.exists(in_df_360min_orig),\
        'wrong orig DF location L2'
assert os.path.exists(in_lorenz_df_L2),\
         'wrong Lorenz Curves original Df L2'
assert os.path.exists(in_lorenz_df_L2_sim),\
         'wrong Lorenz Curves Df L2'

#LEVEL three
assert os.path.exists(in_data_prob_df_file_L3),\
        'wrong data prob location L3'
assert os.path.exists(in_data_prob_stn_df_file_L3),\
         'wrong data stn prob location L3'
assert os.path.exists(in_data_beta_df_file_L3),\
        'wrong data beta location L3'
assert os.path.exists(unbounded_model_dir_L3),\
        'wrong unbounded model DF location L3'
assert os.path.exists(params_file_L3),\
        'wrong params DF location L3'
assert os.path.exists(in_dfs_simulation_3),\
        'wrong simulation DF location L3'
assert os.path.exists(in_df_180min_orig),\
        'wrong orig DF location L3'
assert os.path.exists(in_lorenz_df_L3),\
         'wrong Lorenz Curves original Df L3'
assert os.path.exists(in_lorenz_df_L3_sim),\
         'wrong Lorenz Curves Df L3'
#used for plotting and labelling
stn_list = ['EM02', 'EM03', 'EM05','EM06','EM07','EM08',
            'EM09', 'EM10',  'EM12', 'EM13',  'EM15', 'EM16']

#==============================================================================
# class to break inner loop and execute outer loop when needed
#==============================================================================
class ContinueI(Exception):
  pass
#==============================================================================
# PLOT Histogram WEIGHTS and fitted BETA distribution
#==============================================================================

#def fig size, will be read in imported module to plot Latex like plots
fig_size = (12, 7)
dpi = 600
save_format = '.png'

font_size_title =  16
font_size_axis =  15
font_size_legend = 11

marker_size = 44

year_ticks = YearLocator(1)
ticks_format = DateFormatter('%Y')

##get dfs from directory Level one

def getWeights(cascade_level):
   dfs_files = []
   for r, dir_, f in os.walk(os.path.join(in_data_dir,
                                           r'%s' % cascade_level)):
        for fs in f:
            if fs.endswith('.csv'):
                dfs_files.append(os.path.join(r, fs))
   return dfs_files
dfs_files_L1 = getWeights(cascade_level_1)
dfs_files_L2 = getWeights(cascade_level_2)
dfs_files_L3 = getWeights(cascade_level_3)

#define df_seperator
df_sep = ';'
date_format = '%Y-%m-%d %H:%M:%S'

#define transparency
transparency = 0.55

#define ratio of bars, and width decrease factor
ratio_bars = 0.075
ratio_decrease_factor = 0.99

#define if normed or not
norme_it = True

#define line width
line_width = 0.01

#define colors
colors = ['b', 'r']

#plot
def pltHistWeights(in_dfs_data_list, in_df_beta_param, cascade_level):

    '''
    input: weights_dfs , beta_params_df, cascade_level
    output: plots of weights histogram and fitted beta pdf
    '''

    #define out_dir for these plots
    out_figs_dir0 = os.path.join(out_dir, 'W_hist and Beta_dist %s'\
                                %cascade_level)
    if not os.path.exists(out_figs_dir0): os.mkdir(out_figs_dir0)

    global bins, ratio_bars

    #read beta pdf params results
    df_beta = pd.read_csv(in_df_beta_param, sep=df_sep, index_col=0)

    #go through file and stations
    for df_file in in_dfs_data_list:

        for station in df_beta.index:

            #make sure it is same station
            if station in df_file:

                    fig = plt.figure(figsize=fig_size, dpi=dpi)
                    ax = fig.add_subplot(111)

                    #read each df-file
                    d = pd.read_csv(df_file, sep=df_sep, index_col=0)

                    #select stn beta distribution params
                    a = df_beta.loc[station, 'alfa']
                    b = df_beta.loc[station, 'beta']

                    for i, col in enumerate(d.columns):

                        #plot weights sub interval left W1
                        if i == 0:

                            # select weights between 0 and 1 for plot
                            for val in d[col].values:
                                if val != 0 and val <= 1e-8: #in case low vals
                                    d[col].replace(val, value=0., inplace=True)

                            d = d.loc[(d[col] != 0.0) & (d[col] != 1.0)]

                            #define bins nbr for histogram of weights

                            bins = np.linspace(0., 1.0, 11)
                            center = (bins[:-1] + bins[1:]) / 2

                            #plot beta dist for weights 0 < W < 1
                            ax.scatter(d[col].values,
                                        beta.pdf(d[col].values, a, b),
                                        color='r', marker='o',alpha=0.75)

                            #plot hist weights 0 < W < 1
                            hist,bins = np.histogram(d[col].values,
                                                     bins=bins,
                                                     normed=norme_it)

                            ax.bar(center,
                                    hist,
                                    align='center',
                                    width=ratio_bars,
                                    alpha=transparency,
                                    linewidth=line_width,
                                    color=colors[i],
                                    label=col)

                            ratio_bars = ratio_bars * ratio_decrease_factor

                            ax.set_xlabel('Weights W',
                                          fontsize=font_size_axis)
                            ax.set_ylabel('Normed Weights Count',
                                          fontsize=font_size_axis)

                            red_patch = mpatches.\
                                        Patch(color='r',
                                              label='Fitted Beta Distribution')

                            blue_patch = mpatches.\
                                        Patch(color='b',
                                              label='Weights Distribution')

                            plt.title((r'%s cascade %s'
                                       '\nnumber of $0<W<1$: %d')\
                                        %(station, cascade_level,
                                          len(d[col].index)),
                                          fontsize=font_size_title)

                            plt.text(0.86, 1.01, (r'$\beta$=%0.2f' %a),
                                     withdash=True, fontsize=font_size_title)

                            legend = ax.legend(handles=[red_patch, blue_patch],
                                      loc=2, prop={'size':font_size_legend})

                            ax.grid(True, which='both',
                                    linestyle='-', linewidth=0.1)

                            plt.setp(ax.get_xticklabels(),
                                     rotation='horizontal',
                                     fontsize=font_size_axis)

                            plt.setp(ax.get_yticklabels(),
                                     rotation='horizontal',
                                     fontsize=font_size_axis)

                            frame = legend.get_frame()
                            frame.set_facecolor('0.9')
                            frame.set_edgecolor('0.9')

                            plt.savefig(os.path.join(out_figs_dir0,
                                station+'_'+cascade_level+save_format),
                                frameon=True,
                                papertype='a4',
                                bbox_inches='tight')


                            plt.close('all')
#                break
    return

#call fct for level one and level two
#pltHistWeights(dfs_files_L1, in_data_beta_df_file_L1, cascade_level_1)
#pltHistWeights(dfs_files_L2, in_data_beta_df_file_L2, cascade_level_2)
#pltHistWeights(dfs_files_L3, in_data_beta_df_file_L3, cascade_level_3)
print('done plotting hist of weights and fitted beta distribution')

def plotProbMonth(prob_df_file, cascade_level):

    out_figs_dir1 = os.path.join(out_dir, 'Seasonal effects on P01 %s'\
                                 %cascade_level)
    if not os.path.exists(out_figs_dir1): os.mkdir(out_figs_dir1)

    fig2 = plt.figure(figsize=fig_size)
    ax2 = fig2.add_subplot(111)

    ax2.grid(color='k', linestyle='-', linewidth=0.1)

    in_prob_df = pd.read_csv(prob_df_file, sep=df_sep, index_col=0)
    in_prob_df = in_prob_df[in_prob_df >= 0.]
    x = np.array([(in_prob_df.index)])

    ax2.set_xticks(np.linspace(1, 12, 12))

    y_1 = np.array([(in_prob_df['P1 per Month'].values)])
    y_1 = y_1.reshape((x.shape))

    ax2.scatter(x, y_1, c='r', marker='x',
                s=marker_size,
                label=r'P($W_{1}$) = 1')

    y_0 = np.array([(in_prob_df['P0 per Month'].values)])
    y_0 = y_0.reshape((x.shape))

    ax2.scatter(x, y_0, c='b', marker='o',
                s=marker_size,
                label='P($W_{1}$) = 0')

    y_3 = np.array([(in_prob_df['P01 per month'].values)])

    y_3 = y_3.reshape((x.shape))

    ax2.scatter(x, y_3, c='k', marker='+',
                s=marker_size,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1')

    ax2.yaxis.set_ticks(np.arange(0, max(y_3[0]), 0.05))

    plt.title(r'Seasonal Effects on $P_{01}$ for Cascade %s'\
              % cascade_level, fontsize=font_size_title)

    ax2.set_xlabel(r'Month', fontsize=font_size_axis)
    ax2.set_ylabel('P($W_{1}$)', fontsize=font_size_axis)

    legend = ax2.legend(loc=2, prop={'size':font_size_legend})
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')

    plt.savefig(os.path.join(out_figs_dir1,
                             r'P01 per month_%s'%save_format),
                    frameon=True,
                    papertype='a4',
                    bbox_inches='tight')

    plt.close('all')
    return

#call fct level 1 and level 2
#plotProbMonth(in_data_prob_df_file_L1, cascade_level_1)
#plotProbMonth(in_data_prob_df_file_L2, cascade_level_2)
#plotProbMonth(in_data_prob_df_file_L3, cascade_level_3)

print('done plotting seasonal effect on P01')

#==============================================================================
# Plot Prob that P(W=0) or P(W=1) per Station
#==============================================================================
def probStation(prob_stn_df_file, cascade_level):

    out_figs_dir2 = os.path.join(out_dir, 'Prob P01 per stn %s'\
                                %cascade_level)
    if not os.path.exists(out_figs_dir2): os.mkdir(out_figs_dir2)

    fig3 = plt.figure(figsize=fig_size)
    ax3 = fig3.add_subplot(111)

    global in_df

    #read prob df file and select >= 0 values
    in_df = pd.read_csv(prob_stn_df_file, sep=df_sep, index_col=0)
    in_df = in_df[in_df >= 0.]

    #for labeling x axis by station names
    u, x = np.unique([(in_df.index)], return_inverse=True)
    alpha = 0.85

    #plot P1 values
    y_1 = np.array([(in_df['P1'].values)])
    y_1 = y_1.reshape(x.shape)
    ax3.scatter(x, y_1, c='r', marker='x',
                s=marker_size,
                label=r'P($W_{1}$) = 1', alpha=alpha)

    #plot P0 values
    y_2 = np.array([(in_df['P0'].values)])
    y_2 = y_2.reshape(x.shape)
    ax3.scatter(x, y_2, c='b', marker='o',
                s=marker_size,
                label=r'P($W_{1}$) = 0', alpha=alpha)

    #plot P01 values
    y_3 = np.array([(in_df['P01'].values)])
    y_3 = y_3.reshape(x.shape)
    ax3.scatter(x, y_3, c='k', marker='*',
                s=45,
                label=r'P($W_{1}$)=0 or P($W_{1}$)=1', alpha=alpha)

    ax3.set(xticks=range(len(u)), xticklabels=u)

    ax3.yaxis.set_ticks(np.arange(0, max(y_3), 0.05))
    ax3.grid(color='k', linestyle='-', linewidth=0.1)

    plt.setp(ax3.get_xticklabels(), rotation='horizontal',
             fontsize=font_size_axis)

    plt.setp(ax3.get_yticklabels(), rotation='horizontal',
             fontsize=font_size_axis)

    plt.title(('$P_{01}$ per Station'
              ' Cascade %s') %cascade_level, fontsize=font_size_title)

    ax3.set_ylabel('P(W1)', fontsize=font_size_axis)
    ax3.set_xlabel('Station ID', fontsize=font_size_axis)

    legend = plt.legend(loc=2, prop={'size':font_size_legend})
    frame = legend.get_frame()
    frame.set_facecolor('0.9')
    frame.set_edgecolor('0.9')

    plt.savefig(os.path.join(out_figs_dir2,
                         r'P01perStation%s' %save_format),
                frameon=True,
                papertype='a4',
                bbox_inches='tight')

    plt.close('all')
    return

#call fct Level 1 and Level 2
#probStation(in_data_prob_stn_df_file_L1, cascade_level_1)
#probStation(in_data_prob_stn_df_file_L2, cascade_level_2)
#probStation(in_data_prob_stn_df_file_L3, cascade_level_3)

print('done plotting P0, P1, P01 for every station')
#raise Exception
#==============================================================================
# PLOT Unbounded Model Volume Dependency
#==============================================================================

##read df values of R_vals, W_vals and logLikelihood vals L1

def getUnboudParams(unbouded_model_dir):
   dfs_files = []
   for r, dir_, f in os.walk(os.path.join(in_data_dir,
                                           r'%s' % unbouded_model_dir)):
        for fs in f:
            if fs.endswith('.csv'):
                dfs_files.append(os.path.join(r, fs))
   return dfs_files
dfs_files_P01_L1 = getUnboudParams(unbounded_model_dir_L1)
dfs_files_P01_L2 = getUnboudParams(unbounded_model_dir_L2)
dfs_files_P01_L3 = getUnboudParams(unbounded_model_dir_L3)

def volumeDependacyP01(in_df_files, in_param_file, cascade_level):

    percentile = 0.05 #to divide R values into classes and fill classes with W
    min_w_nbrs = 10 #when calculating P01, min nbr of W to consider

    global d_plot, klassen_w_vals, w_01, klassen_R_vals, nbr_classes, klasses

    #out_dir
    out_figs_dir3 = os.path.join(out_dir, 'P01 unbounded model %s'\
                                %cascade_level)
    if not os.path.exists(out_figs_dir3): os.mkdir(out_figs_dir3)

    for df_file in in_df_files:

        for stn in stn_list:

            if fnmatch.fnmatch(df_file,'*.csv') and stn in df_file:

                #new fig per stn

                fig = plt.figure(figsize=fig_size)
                ax = fig.add_subplot(111)

                #read df_file with coulms: R_vals, W_vals, W_vals(0_1), L(teta)
                d = pd.read_csv(df_file, sep=df_sep, index_col=0)

                #calculate P01 as freq from observed R, W values
                '''
                superimposed are the 'observed' values of P01 estimated:
                    by fitting to the observed values of W with in each second
                    percentile of R, plotted against the mean values of R
                    in these ranges.
                '''
                #new df to plot P01 vs log_10R
                d_plot = pd.DataFrame(index=\
                                      np.arange(0, len(d.index), 1))

                #new cols for  R vals and W1 vals
                d_plot['R vals'] = d['R vals']
                d_plot['W_01'] = d['W 01']

                #define classes min and max R values
                r_min = min(d_plot['R vals'].values)
                r_max = max(d_plot['R vals'].values)

                #define classes width increase
                k_inc = percentile * r_max

                #find needed nbr of classes
                nbr_classes = int(round((r_max-r_min) / (r_min*k_inc)))

                #new dicts to hold klasses intervals
                klasses={}

                #new dicts to hold klasses W values
                klassen_w_vals={}

                #new dicts to hold klasses R values
                klassen_r_vals = {}

                #new dicts to hold klasses W01 values for P01 observed
                w_01 = {}

                #create new classes and lists to hold values
                for i in range(nbr_classes+1):
                    klasses[i] = [r_min+i*k_inc*r_min, r_min+(1+i)*k_inc*r_min]
                    klassen_w_vals[i] = []
                    klassen_r_vals[i] = []
                    w_01[i] = []

                #go through values
                for val, w_val in zip(d_plot['R vals'].values,
                                       d_plot['W_01'].values):
                    #find Rvals and Wvals per class
                    for klass in klasses.keys():
                        # if R val is in class, append w_val and r_val to class
                        if min(klasses[klass]) <= val <= max(klasses[klass]):

                            klassen_w_vals[klass].append(w_val)
                            klassen_r_vals[klass].append(val)

                #find P01 as frequency per class
                for klass in klassen_w_vals.keys():

                    #if enough values per class
                    if len(klassen_w_vals[klass]) >= min_w_nbrs:

                        #if w_val = 0 then w=0 or w=1 elif w_val=1 then 0<w<1
                        #this is why use P01 = 1-sum(W01)/len(W01)
                        w_01[klass].append(1-(sum(klassen_w_vals[klass])\
                                            / len(klassen_w_vals[klass])))

                        #calculate mean of rainfall values of the class
                        w_01[klass].append(np.mean\
                            (np.log10(klassen_r_vals[klass])))

                #convert dict Class: [P01, Log(Rmean)] to df, Classes as idx
                ds = pd.DataFrame.from_dict(w_01, orient='index')
#                print(ds)

                #count 0<w<1 for plotting it in title
                ct = 0
                for val in d_plot['W_01'].values:
                    if val == 0.: ct += 1

                if len(d['R vals'].values) >= min_w_nbrs:

                    #plot observed P01, x=mean(log10(R_values)), y=(P01)

                    ax.scatter(ds[1], ds[0], c='b', marker='x', alpha=0.85,
                               label='nbr of observed W=0 or W=1: %0.2f' % ct)

                    #read df for logRegression parameters
                    df_param = pd.read_csv(in_param_file,
                                           sep=df_sep,
                                           index_col=0)

                    #implement logRegression fct
                    def logRegression(r_vals, a, b):
                        return  np.array([1 - 1 / (1 + np.exp(\
                            - (np.array([a + b * np.log10(r_vals)]))))])

                    #x values = log10 R values
                    x_vals = np.log10(d['R vals'].values)

                    #extract logRegression params from df
                    a_ = df_param.loc[stn, 'a']
                    b_ = df_param.loc[stn, 'b']

                    #calculate logRegression fct
                    y_vals = logRegression(d['R vals'].values, a_, b_)

                    #plot x, y values of logRegression
                    ax.scatter(x_vals, y_vals, c='r', marker='o',alpha=0.85,
                               label='fitted logRegression a=%.2f b =%.2f'\
                               %(a_, b_))

                    ax.set_xlabel('log$_{10}$ R', fontsize = font_size_axis)
                    ax.set_ylabel('P$_{01}$', fontsize = font_size_axis)
                    ax.grid(color='k', linestyle='-', linewidth=0.1)

                    plt.title('%s Volume dependence of P$_{01}$ for Cascade %s'\
                              %(stn,cascade_level), fontsize = font_size_title )

                    legend = ax.legend(loc=1, prop={'size':font_size_legend})
                    frame = legend.get_frame()
                    frame.set_facecolor('0.9')
                    frame.set_edgecolor('0.9')

                    plt.savefig(os.path.join(out_figs_dir3,
                                'Volume dependence of P(01) for %s_%s'%(stn,
                                                       save_format)),
                                frameon=True,
                                papertype='a4',
                                bbox_inches='tight')
                    plt.close('all')
#        break
    return d_plot, klasses

#call fct Level one and two
#volumeDependacyP01(dfs_files_P01_L1, params_file_L1, cascade_level_1)
#volumeDependacyP01(dfs_files_P01_L2, params_file_L2, cascade_level_2)
#volumeDependacyP01(dfs_files_P01_L3, params_file_L3, cascade_level_3)

print('done plotting the volume dependency of P01')


#==============================================================================
# Model EVALUATION
#==============================================================================

#get simulated files level one
dfs_files_sim = []
for r, dir_, f in os.walk(in_dfs_simulation):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_files_sim.append(os.path.join(r, fs))

#get simulated files level two
dfs_files_sim_2 = []
for r, dir_, f in os.walk(in_dfs_simulation_2):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_files_sim_2.append(os.path.join(r, fs))

#get simulated files level three
dfs_files_sim_3 = []
for r, dir_, f in os.walk(in_dfs_simulation_3):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_files_sim_3.append(os.path.join(r, fs))

#color map for subplots
cmap = plt.cm.get_cmap('Blues')

cm_vals = 2.5

continue_i = ContinueI()

#raise Exception
def compareHistRain (in_df_orig_vals_file,
                     in_df_simulation_files,
                     cascade_level):
    '''
        input: original precipitaion values
                simuated precipitation values, baseline and unbounded
                cascade level
        output: subplot of baseline and unbounded model vs orig vals
                np.log10(values) is plotted
    '''

    #out_dir
    out_figs_dir = os.path.join(out_dir, 'Model Evaluation %s'\
                        %cascade_level)
    if not os.path.exists(out_figs_dir): os.mkdir(out_figs_dir)

    #define shape of sub plot
    sb_plt_shp = (3, 6)

    #define bins and centering of bars
    bins2 = np.arange(-0.6,1.5,0.05)
    center2 = (bins2[:-1] + bins2[1:]) / 2

    #read df orig values
    in_df_orig_vals = pd.read_csv(in_df_orig_vals_file,
                                  sep=df_sep, index_col=0)

    for station in (in_df_orig_vals.columns):
#        print(station)

        for i, df_file in enumerate(in_df_simulation_files):
#            print(df_file)

            if station in df_file:

#                    print(station, df_file)

                    #read file as dataframe
                    df_sim_vals= pd.read_csv(df_file,
                                             sep=df_sep,
                                             index_col=0)


                    #start going through index of station data
                    for idx in df_sim_vals.index:

                        try:
                            #locate idx in simulated files, extract orig vals
                            df_sim_vals.loc[idx, 'orig vals'] =\
                            in_df_orig_vals[station].loc[idx]

                        except KeyError as msg:
                            print(msg)
                            continue
#==============================================================================
# start plotting original values
#==============================================================================

                    #define patch for legend
                    red_patch = mpatches.\
                                Patch(color='b',
                                      label='original_vals')

                    #plot hist for original vals
                    hist0, bins = np.histogram(np.log10\
                                           (df_sim_vals['orig vals'].values),
                                           bins=bins2, range=(-0.6,1.5),
                                           normed=norme_it)

                    long_top_axes = plt.subplot2grid(sb_plt_shp, (0, 0),
                                                     rowspan=1, colspan=6)

                    long_top_axes.bar(center2,
                                      hist0,
                                      width=0.05,
                                      alpha=0.9,
                                      linewidth=0.09,
                                      color='b',
                                      label='orig vals')

                    #adjust plot labels and grid
                    long_top_axes.set_ylabel('Frequency')

                    long_top_axes.grid(True, which='both',
                                        linestyle='-', linewidth=0.01)

                    long_top_axes.legend(handles=[red_patch],
                                            loc='best')

                    long_top_axes.set_xticklabels([])

#==============================================================================
#start plotting baseline model values
#==============================================================================

                    #define patch for legend
                    blue_patch = mpatches.\
                                Patch(color='r',
                                      label='baseline model')

                    #extract baseline values from simulated file
                    hist1, bins1 = np.histogram(np.log10\
                    (df_sim_vals['baseline rainfall %s' %cascade_level].values),
                     bins=bins2,
                     range=(-0.6,1.5),
                     normed=norme_it)

                    shrt_left_axes = plt.subplot2grid(sb_plt_shp, (1, 0),
                                                      rowspan=1, colspan=6,
                                                      sharex=long_top_axes)

                    shrt_left_axes.bar(center2,
                                       hist1,
                                       align='center',
                                       width=0.05,
                                       alpha=0.9,
                                       linewidth=0.09,
                                       color='r',
                                       label='baseline mdoel')

                    shrt_left_axes.set_ylabel('Frequency')

                    shrt_left_axes.grid(True, which='both',
                                        linestyle='-', linewidth=0.01)

                    shrt_left_axes.legend(handles=[blue_patch],
                                            loc='best')
                    shrt_left_axes.set_xticklabels([])
#==============================================================================
#start plotting unbounded model values
#==============================================================================

                    #define patch for legend
                    cyan_patch = mpatches.\
                                Patch(color='m',
                                      label='unbounded model')

                    #extract unbounded values from simulated file
                    hist2, bins2 = np.histogram\
                    (np.log10\
                     (df_sim_vals['unbounded rainfall %s' %cascade_level]\
                                                                  .values),
                     bins=bins2,
                     range=(-0.6,1.5),
                     normed=norme_it)

                    second_left_axes = plt.subplot2grid(sb_plt_shp, (2, 0),
                                                        rowspan=1, colspan=6)

                    second_left_axes.bar(center2,
                                         hist2,
                                         align='center',
                                         width=0.05,
                                         alpha=0.9,
                                         linewidth=0.09,
                                         color='m',
                                         label='unbounded model')

                    second_left_axes.set_xlabel('log10 (R)')

                    second_left_axes.set_ylabel('Frequency')


                    second_left_axes.legend(handles=[cyan_patch],
                                            loc='best')

                    second_left_axes.grid(True, which='both',
                                          linestyle='-', linewidth=0.01)

                    # adjust shape and title of subplot
                    plt.subplots_adjust(wspace=10, hspace=0.3, top=0.93)
                    plt.grid()

                    plt.suptitle('%s Observed vs Simulated Rainfall at %s'\
                                 %(station, cascade_level))

                    plt.savefig(os.path.join(out_figs_dir,
                                 'comparision %s nbr %d_%s'%(station,
                                                             i,save_format)),
                                bbox_inches='tight')


                    plt.close('all')

#            else: continue

#                    raise continue_i

        #break inner for loop
#        except ContinueI :
#            continue
#                break
#        continue

    return

#call fct
#compareHistRain(in_df_720min_orig, dfs_files_sim, cascade_level_1)
#compareHistRain(in_df_360min_orig, dfs_files_sim_2, cascade_level_2)
#compareHistRain(in_df_180min_orig, dfs_files_sim_3, cascade_level_3)
#
print('done plotting the results of the simulations')

#raise Exception
#==============================================================================
#
#==============================================================================
def compareSimVals(orig_vals_df, in_df_simulation, cascade_level):

    '''
    idea: compare the mean of all simulations and standard deviations
    '''
    #out_dir
    out_figs_dir = os.path.join(out_dir, 'Comparing models %s'\
                        %cascade_level)
    if not os.path.exists(out_figs_dir): os.mkdir(out_figs_dir)
    global df_mean_stn, df_mean_all, df_extremes

    #new df to hold all means
    df_all = pd.DataFrame()

    sb_plt_shp = (3, 6)

    #define bins and centering of bars
    bins2 = np.arange(-0.6,1.5,0.05)
    center2 = (bins2[:-1] + bins2[1:]) / 2

    #read df orig values
    in_df_orig = pd.read_csv(orig_vals_df, sep=df_sep, index_col=0)
    for i, df_file in enumerate(in_df_simulation):

        for station in (in_df_orig.columns):
            if station in df_file:

                #new df to hold all simulated values, baseline and unbounded
                df_mean_stn = pd.DataFrame()
                df_mean_all = pd.DataFrame()
#        try:
                df_sim_vals= pd.read_csv(df_file,
                                         sep=df_sep,
                                         index_col=0)

#                    in_df_orig[station].dropna(axis=0, inplace=True)

                for idx in df_sim_vals.index:
                    try:

                        #for each idx,extract all simulated values
                        df_sim_vals.loc[idx, 'orig vals'] =\
                        in_df_orig[station].loc[idx]

                        df_mean_stn.loc[idx, i] =\
                        df_sim_vals.loc\
                        [idx,'baseline rainfall %s' %cascade_level]

                        df_mean_all.loc[idx, i] =\
                        df_sim_vals.loc\
                        [idx,'unbounded rainfall %s' %cascade_level]

                    except KeyError as msg:
                        print(msg)
                        pass

                # calculate the mean of all simulations and plot it vs obs
                df_all['mean baseline'] = df_mean_stn.mean(axis=1)
                df_all['mean unbounded'] = df_mean_all.mean(axis=1)

#==============================================================================
# start plotting simulations
#==============================================================================
            cyan_patch = mpatches.\
                        Patch(color=cmap(cm_vals),
                              label='original_vals')

            hist0, bins = np.histogram(np.log10\
                                       (df_sim_vals['orig vals'].values),
                                       bins=bins2, range=(-0.6,1.5),
                                       normed=norme_it)

            long_top_axes = plt.subplot2grid(sb_plt_shp, (0, 0),
                                             rowspan=1, colspan=6)

            long_top_axes.bar(center2,
                              hist0,
                              width=0.05,
                              alpha=0.9,
                              linewidth=0.09,
                              color=cmap(cm_vals),
                              label='orig vals')

            long_top_axes.set_ylabel('Frequency')

            long_top_axes.grid()

            long_top_axes.legend(handles=[cyan_patch],
                                    loc='best')

            long_top_axes.set_xticklabels([])

#==============================================================================
#
#==============================================================================
            red_patch = mpatches.\
                        Patch(color=cmap(cm_vals),
                              label='baseline model mean')

            hist1, bins1 = np.histogram\
            (np.log10(df_all['mean baseline'].values),
             bins=bins2,
             range=(-0.6,1.5),
             normed=norme_it)

            shrt_left_axes = plt.subplot2grid(sb_plt_shp, (1, 0),
                                              rowspan=1, colspan=6,
                                              sharex=long_top_axes)
            shrt_left_axes.bar(center2,
                               hist1,
                               align='center',
                               width=0.05,
                               alpha=0.9,
                               linewidth=0.09,
                               color=cmap(cm_vals),
                               label='baseline model')

            shrt_left_axes.set_ylabel('Frequency')

            shrt_left_axes.grid()

            shrt_left_axes.legend(handles=[red_patch],
                                    loc='best')

            shrt_left_axes.set_xticklabels([])
#==============================================================================
#
#==============================================================================

            blue_patch = mpatches.\
                        Patch(color=cmap(cm_vals),
                              label='unbounded model mean')

            hist2, bins1 = np.histogram\
            (np.log10(df_all['mean unbounded'].values),
             bins=bins2,
             range=(-0.6,1.5),
             normed=norme_it)

            left_axes = plt.subplot2grid(sb_plt_shp, (2, 0),
                                              rowspan=1, colspan=6)
            left_axes.bar(center2,
                               hist2,
                               align='center',
                               width=0.05,
                               alpha=0.9,
                               linewidth=0.09,
                               color=cmap(cm_vals),
                               label='unbounded mdoel')

            left_axes.set_ylabel('Frequency')
            left_axes.set_xlabel('log10 (R)')
            left_axes.grid()

            left_axes.legend(handles=[blue_patch],
                                    loc='best')
        plt.suptitle('%s average of 100 Simulations vs Observed' %station)

        plt.savefig(os.path.join(out_figs_dir,
                            '%s compare %s_%s' %(station,
                                                  cascade_level,
                                                  save_format)))

#            raise continue_i

#        except ContinueI:
#            continue
#        df_mean_all.to_csv(os.path.join(out_figs_dir,
#                           '%s baseline compare %s.csv' %(station,
#                                                          cascade_level)),
#                        sep=df_sep)

#        df_mean_stn.to_csv(os.path.join(out_figs_dir,
#                           '%s unbounded compare %s.csv' %(station,
#                                                          cascade_level)),
#                        sep=df_sep)

    return

#call function level one and level two
#compareSimVals(in_df_30min_orig, dfs_files_sim, cascade_level_1)
#compareSimVals(in_df_15min_orig, dfs_files_sim_2, cascade_level_2)

print('done plotting the mean of all the simulations')

#==============================================================================
#
#==============================================================================

def compareExtremes(orig_vals_df, in_df_simulation, cascade_level):

    '''
    idea: compare frequency and magnitudes of extremes

    '''
    #out_dir
    out_figs_dir = os.path.join(out_dir, 'Comparing Extremes of models %s'\
                        %cascade_level)
    if not os.path.exists(out_figs_dir): os.mkdir(out_figs_dir)

    global df_count_vls

#    quartiles = [0.25, 0.5, 0.75, 0.99]
    quartiles = [0.25]

    #read df orig values
    in_df_orig = pd.read_csv(orig_vals_df, sep=df_sep, index_col=0)

    for i, df_file in enumerate(in_df_simulation):

        for station in (in_df_orig.columns):
                if station in df_file and station != 'EM07':

#                    print(station)
                    plt.figure(figsize=fig_size, dpi=dpi)

                #new df to hold all simulated values, baseline and unbounded
                    df_extremes_orig = pd.DataFrame()
                    df_extremes_baseline = pd.DataFrame()
                    df_extremes_unbound = pd.DataFrame()

                    df_count_vls = pd.DataFrame()
        #        try:
                    df_sim_vals= pd.read_csv(df_file,
                                             sep=df_sep,
                                             index_col=0)

                    for idx in df_sim_vals.index:
                        try:

                            #for each idx,extract orig vals
                            df_sim_vals.loc[idx, 'orig vals'] =\
                            in_df_orig[station].loc[idx]

                        except KeyError as msg:
                            print(msg)
                            pass

                    #first get maximum of original values
                    try:
                        max_orig = max(df_sim_vals['orig vals'].values)

                        max_baseline = max(df_sim_vals\
                                   ['baseline rainfall %s' %cascade_level].values)
                        max_unbound = max(df_sim_vals\
                                   ['unbounded rainfall %s' %cascade_level].values)

                        print(max_orig, max_baseline, max_unbound)

                        #extract all values with in 75% of the maximum
                        for quartile in quartiles:
                            df_extremes_orig['orig extremes'] =\
                                        df_sim_vals['orig vals']\
                                        [df_sim_vals\
                                         ['orig vals'] >= quartile*max_orig]

                            df_extremes_baseline['baseline extremes'] =\
                                        df_sim_vals\
                                        ['baseline rainfall %s' %cascade_level]\
                                        [df_sim_vals\
                                         ['baseline rainfall %s' %cascade_level]\
                                         >= quartile*max_baseline]

                            df_extremes_unbound['unbounded extremes'] =\
                                        df_sim_vals\
                                        ['unbounded rainfall %s' %cascade_level]\
                                        [df_sim_vals\
                                         ['unbounded rainfall %s' %cascade_level]\
                                         >= quartile*max_unbound]

                            # plot  extremes

                            x_axis_0 = df_extremes_orig.index.astype(np.datetime64)
    #                        print(df_extremes_orig['orig extremes'].values)
    #                        print(x_axis_0)
                            print(df_extremes_orig['orig extremes'].values)
                            plt.scatter(x_axis_0,
                                        df_extremes_orig['orig extremes'].values,
                                        marker = 'o', c = 'b',
                                        alpha=0.85,
                                        label='orig extremes %s'%len\
                                        (df_extremes_orig['orig extremes'].values))

                            x_axis_1 = df_extremes_baseline.index.\
                                        astype(np.datetime64)

                            plt.scatter(x_axis_1,
                                        df_extremes_baseline\
                                        ['baseline extremes'].values,
                                        marker='+', c = 'r',
                                        alpha=0.85,
                                        label='baseline extremes %s'%len\
                                        (df_extremes_baseline\
                                         ['baseline extremes'].values))

                            x_axis_2 = df_extremes_unbound.index.\
                                        astype(np.datetime64)

                            plt.scatter(x_axis_2,
                                df_extremes_unbound['unbounded extremes'].values,
                                marker='*', c = 'm',
                                alpha=0.85,
                                label='unbounded extremes %s' %len\
                                (df_extremes_unbound['unbounded extremes'].values))

        #                            plt.tick_params(axis=)
        #                            plt.xticks.__format__(ticks_format)

                            plt.grid()

                            plt.legend(loc='best')
                            plt.ylabel('R(mm)')
                            plt.xlabel('Date')
                            plt.suptitle(('%s Simulations vs Observed'
                                          'Extremes from %0.2f Percent' %(station,
                                                                  100*quartile)))

                            plt.savefig(os.path.join(out_figs_dir,
                            '%s compare extremes %0.2f_%d %s_%s' %(station,
                                                                 quartile,i,
                                                                 cascade_level,
                                                                 save_format)))
                            plt.close('all')
                    except Exception as msg:
                            print(msg)
                            continue
    return
#                        raise continue_i
#            #break outer loop
#        except ContinueI:
#            continue
compareExtremes(in_df_720min_orig, dfs_files_sim, cascade_level_1)
compareExtremes(in_df_360min_orig, dfs_files_sim_2, cascade_level_2)
compareExtremes(in_df_180min_orig, dfs_files_sim_3, cascade_level_3)

#==============================================================================
#
#==============================================================================

def boundsSimulated(orig_vals_df, in_df_simulation, cascade_level):

    '''
    idea: plot all simulations bounds, min and max simulated vs orig values
    '''
    #out_dir
    out_figs_dir = os.path.join(out_dir, 'Simulation bounds %s'\
                        %cascade_level)
    if not os.path.exists(out_figs_dir): os.mkdir(out_figs_dir)
    global df_baseline, df_unbounded

    #read df orig values
    in_df_orig = pd.read_csv(orig_vals_df, sep=df_sep, index_col=0)

    #    in_df_orig = in_df_orig[in_df_orig > threshold]
    for i, df_file in enumerate(in_df_simulation):

        for station in (in_df_orig.columns):
            if station in df_file:
                print(station)
                #new df to hold all simulated values, baseline and unbounded
                df_baseline = pd.DataFrame()
                df_unbounded = pd.DataFrame()
                df_bounds = pd.DataFrame()
                df_bounds_unbounded = pd.DataFrame()
                #        try:

                plt.figure(figsize=(16,7), dpi=300.)

                df_sim_vals= pd.read_csv(df_file,
                                         sep=df_sep,
                                         index_col=0)

                for idx in df_sim_vals.index:
                    try:

                        #for each idx,extract all simulated values
                        df_sim_vals.loc[idx, 'orig vals'] =\
                        in_df_orig[station].loc[idx]

                        df_baseline.loc[idx, i] =\
                        df_sim_vals.loc\
                        [idx,'baseline rainfall %s' %cascade_level]

                        df_unbounded.loc[idx, i] =\
                        df_sim_vals.loc\
                        [idx,'unbounded rainfall %s' %cascade_level]

                    except KeyError as msg:
                        print(msg)
                        continue

                df_bounds['min baseline'] =\
                            df_baseline.min(axis=1)

                df_bounds['max baseline'] =\
                            df_baseline.max(axis=1)
                #adjust index so it is time
                df_bounds.reindex(df_baseline.index)

                df_bounds_unbounded['min unbounded'] =\
                            df_unbounded.min(axis=1)
                df_bounds_unbounded['max unbounded'] =\
                            df_unbounded.max(axis=1)
                #adjust index so it is time
                df_bounds_unbounded.reindex(df_unbounded.index)

                x_axis0 = df_bounds.index.astype(np.datetime64)

                plt.scatter(x_axis0,
                            df_bounds['max baseline'].values,
                            c='r',marker='+', alpha=0.75,
                            label='max baseline')

                x_axis1 = df_bounds_unbounded.index.astype(np.datetime64)

                plt.scatter(x_axis1,
                            df_bounds_unbounded['max unbounded'].values,
                            c='m', marker='*', alpha=0.75,
                            label='max unbounded')

                x_axis2 = df_sim_vals.index.astype(np.datetime64)

                plt.scatter(x_axis2,
                            df_sim_vals['orig vals'].values,
                            c='b', marker='o', alpha=0.75,
                            label='orig data')

                plt.grid()
                plt.ylabel('R(mm)')
                plt.xlabel('Date')
                plt.legend(loc='best')

                plt.title('bounds of Simulations %s %s' %(station,cascade_level))

                plt.savefig(os.path.join(out_figs_dir,
                                '%s Simulation Bounds max %d %s_%s' %(station,
                                                                      i,
                                                     cascade_level, save_format)))
                plt.close('all')

    return

boundsSimulated(in_df_720min_orig, dfs_files_sim, cascade_level_1)
boundsSimulated(in_df_360min_orig, dfs_files_sim_2, cascade_level_2)
boundsSimulated(in_df_180min_orig, dfs_files_sim_3, cascade_level_3)

print('done plotting the bounds of the simulations')
#==============================================================================
#LORENZ Curves
#==============================================================================

#get lorenz files level one
dfs_lorenz_L1 = []
for r, dir_, f in os.walk(in_lorenz_df_L1):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_lorenz_L1.append(os.path.join(r, fs))

dfs_lorenz_L1_sim = []
for r, dir_, f in os.walk(in_lorenz_df_L1_sim):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_lorenz_L1_sim.append(os.path.join(r, fs))

#get lorenz files level two
dfs_lorenz_L2 = []
for r, dir_, f in os.walk(in_lorenz_df_L2):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_lorenz_L2.append(os.path.join(r, fs))

dfs_lorenz_L2_sim = []
for r, dir_, f in os.walk(in_lorenz_df_L2_sim):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_lorenz_L2_sim.append(os.path.join(r, fs))

#get lorenz files level three
dfs_lorenz_L3 = []
for r, dir_, f in os.walk(in_lorenz_df_L3):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_lorenz_L3.append(os.path.join(r, fs))

dfs_lorenz_L3_sim = []
for r, dir_, f in os.walk(in_lorenz_df_L3_sim):
    for fs in f:
        if fs.endswith('.csv'):
            dfs_lorenz_L3_sim.append(os.path.join(r, fs))

def plotLorenzCurves(in_lorenz_vals_dir_orig,
                     in_lorenz_vals_dir_sim,
                     cascade_level):
    '''
    cumulative frequency of rainy days (X)is
    plotted against associated precipitation amount (Y).
    '''
    out_dir_ = os.path.join(out_dir,
                            '%s Lorenz curves' %cascade_level)
    if not os.path.exists(out_dir_): os.mkdir(out_dir_)

    baseline_files = []
    unbounded_files = []

    for in_sim_file in in_lorenz_vals_dir_sim:

        if 'baseline' in in_sim_file:
            baseline_files.append(in_sim_file)

        if 'unbounded' in in_sim_file:
            unbounded_files.append(in_sim_file)

    for  station in stn_list:

            for orig_file, base_file, unbound_file in\
                zip(in_lorenz_vals_dir_orig,
                    baseline_files, unbounded_files):

    #            print(orig_file, base_file, unbound_file )

                if station in orig_file and\
                    station in base_file and\
                    station in unbound_file:

#                    print(station, orig_file, base_file, unbound_file )

                    #start new fig
                    plt.figure(figsize=fig_size, dpi=dpi)

                    in_lorenz_vlas_df_orig = pd.read_csv(orig_file,
                                                        sep=df_sep,
                                                        index_col=0)

                    x_vals = in_lorenz_vlas_df_orig['X']
                    y_vals = in_lorenz_vlas_df_orig['Y']

                    plt.scatter(x_vals, y_vals,
                                color='r',
                                marker='o',
                                label='Observed')

                    in_lorenz_df_sim_base = pd.read_csv(base_file,
                                                    sep=df_sep,
                                                    index_col=0)

                    x_vals_sim = in_lorenz_df_sim_base['X']
                    y_vals_sim = in_lorenz_df_sim_base['Y']

                    plt.scatter(x_vals_sim, y_vals_sim, color='g',
                                marker='*', label='baseline', alpha=0.75)

                    in_lorenz_df_sim_unbound = pd.read_csv(unbound_file,
                                                    sep=df_sep,
                                                    index_col=0)

                    x_vals_sim = in_lorenz_df_sim_unbound['X']
                    y_vals_sim = in_lorenz_df_sim_unbound['Y']

                    plt.scatter(x_vals_sim, y_vals_sim, color='b',
                                marker='+', label='unbounded', alpha=0.75)

                    plt.title('Lorenz Curve %s %s' %(station,
                                                 cascade_level))
                    plt.xlabel('accumulated occurences')
                    plt.ylabel('rainfall contribution')

                    plt.legend(loc='best')

                    plt.savefig(os.path.join(out_dir_,
                         'lorenz_curves_th_%s_%s_%s' %(station,cascade_level,
                                                    save_format)))

                    plt.close('all')
    return

#call fct level one and two
L1_orig_sim = plotLorenzCurves(dfs_lorenz_L1, dfs_lorenz_L1_sim , cascade_level_1)
L2_orig_sim = plotLorenzCurves(dfs_lorenz_L2, dfs_lorenz_L2_sim, cascade_level_2)
L3_orig_sim = plotLorenzCurves(dfs_lorenz_L3, dfs_lorenz_L3_sim, cascade_level_3)

#==============================================================================
# find distribution of the maximums
#==============================================================================

def distMaximums(in_df_orig_file, in_df_simu_files, cascade_level):
    '''
    Idea: read observed and simulated values
        select highest 20_30 values
        see how they are distributed
        compare observed to simulated

    '''
    out_dir_dist = os.path.join(out_dir,
                                r'distribution of Maximums %s' %cascade_level)

    if not os.path.exists(out_dir_dist): os.makedirs(out_dir_dist)

    #read df orig values
    in_df_orig = pd.read_csv(in_df_orig_file, sep=df_sep, index_col=0)

    for station in (in_df_orig.columns):

        for i, df_file in enumerate(in_df_simu_files):

            if station in df_file:


                #new df to hold all simulated values, baseline and unbounded
                df_baseline = pd.DataFrame()
                df_unbounded = pd.DataFrame()

                df_max_baseline = pd.DataFrame()
                df_max_unbounded = pd.DataFrame()
                df_max_orig = pd.DataFrame()
                #        try:
                plt.figure(figsize=(14,6), dpi=300.)

                df_sim_vals= pd.read_csv(df_file,
                                         sep=df_sep,
                                         index_col=0)

                for idx in df_sim_vals.index:
                    try:

                        #for each idx,extract orig values
                        df_sim_vals.loc[idx, 'orig vals'] =\
                        in_df_orig[station].loc[idx]

                        # get values from each simulation
                        df_baseline.loc[idx, i] =\
                        df_sim_vals.loc\
                        [idx,'baseline rainfall %s' %cascade_level]

                        df_unbounded.loc[idx, i] =\
                        df_sim_vals.loc\
                        [idx,'unbounded rainfall %s' %cascade_level]

                    except KeyError as msg:
                        print(msg)
                        continue

                #sort values, to extract extremes
                df_max_baseline = df_baseline[i].\
                                                sort_values(ascending=False,
                                                            kind='mergesort')
                df_max_unbounded = df_unbounded[i].\
                                                sort_values(ascending=False,
                                                            kind='mergesort')
                df_max_orig = df_sim_vals['orig vals'].\
                                                sort_values(ascending=False,
                                                            kind='mergesort')

                #extract extremes, what interest us : 20 vals
                y1, y2, y3 = df_max_orig[:20],\
                             df_max_baseline[:20],\
                             df_max_unbounded[:20]

                    # Cumulative counts:

                plt.step(np.concatenate([y1.values[::-1], y1.values[[0]]]),
                         (np.arange(y1.values.size+1)/len(y1.values)), c='b',
                         label='Observed',
                         alpha=0.85)

                plt.step(np.concatenate([y2.values[::-1], y2.values[[0]]]),
                         (np.arange(y2.values.size+1)/len(y2.values)),
                         c='r',
                         label='Baseline Model',
                         alpha=0.85)


                plt.step(np.concatenate([y3.values[::-1], y3.values[[0]]]),
                         (np.arange(y3.values.size+1)/len(y3.values)),
                         c='m',
                         label='Unbounded Model',
                         alpha=0.85)

                plt.legend(loc='best')
                plt.xlabel('Rainfall Volume (mm)')
                plt.ylabel('F')

                plt.title(('Cumulative Distribution function'
                           '\n Observed vs Simulated Rainfall'
                           '\n Station %s Cascade %s' %(station,cascade_level)))

                plt.savefig(os.path.join(out_dir_dist, 'cdf_%s_%s_%d_%s'\
                                         %(station,cascade_level,
                                           i,save_format)))
                plt.close('all')
    return
max_l1 = distMaximums(in_df_720min_orig, dfs_files_sim, cascade_level_1)
max_l2 = distMaximums(in_df_360min_orig, dfs_files_sim_2, cascade_level_2)
max_l3 = distMaximums(in_df_180min_orig, dfs_files_sim_3, cascade_level_3)


STOP = timeit.default_timer()  # Ending time
print(('\n\a\a\a Done with everything on %s. Total run time was'
       ' about %0.4f seconds \a\a\a' % (time.asctime(), STOP-START)))


