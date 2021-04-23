#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 14 21:46:38 2017

autor: EL Hachem Abbas, IWS

"""
import numpy as np
import matplotlib.pyplot as plt

# mass to be distributed along the support.
masse = 1000

# number of levels of the cascade that we want to develop.
nbrlevels = 12

# length of the support.
# The cascade has support in the interval [0,T].
lengthT = 365

# nbr of branches we want to develop
branchnbr = 2**nbrlevels

# generate matrix with complementary weights (w).
weightsmatrix = np.zeros([nbrlevels, branchnbr], dtype=np.float)

for i in np.arange(1, nbrlevels+1, 1):  # rows i €[1,8]
    for j in np.arange(1, 2**i, 2):  # columns j € [1,3,...,255]
        # sample weights
        weightsmatrix[i-1, j-1] = float(np.random.beta(2, 2))
        # complementary weights
        weightsmatrix[i-1, j] = 1-weightsmatrix[i-1, j-1]

# ordered weights matrix
weights_ordered_matx = np.ones([nbrlevels+1, branchnbr], dtype=float)

for i in range(2, nbrlevels+2):

    repeat = 2**(nbrlevels-i+1)
    bound = int(branchnbr/repeat)

    for k in range(0, bound):
        for j in range(k*repeat+1, (k+1)*repeat+1):
            if j <= (2**k)*repeat:
                temp = k+1
            else:
                temp = k+2
            weights_ordered_matx[i-1, j-1] = weightsmatrix[i-2, temp-1]

# calculate the values for every branch:
# Every row of the matrix 'cascade' corresponds to a level of the cascade.

cascade = np.cumprod([weights_ordered_matx], axis=1)*masse
cascade.shape = weights_ordered_matx.shape

# Transpose the matrix for adjusting the shapes
graf = cascade.transpose()

t = np.arange(float(lengthT)/branchnbr,
              lengthT+0.00001,
              float(lengthT)/branchnbr)

# =============================================================================
# Plot the values per cascade level , ax14, ax15
# =============================================================================
f, (ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9, ax10, ax11,
    ax12, ax13) =\
    plt.subplots(13, figsize=(14, 18),
                 sharex=True,
                 sharey=False)
# def font size legend
ftl = 10


def plotCascadeLevels(subplot, cascade_support, cascade_values,
                      plot_color, level_name):
    ''' function for plotting subplots '''
    subplot.plot(cascade_support, cascade_values, c=plot_color,
                 linewidth=0.8, label=level_name)
    subplot.legend(loc='center left',  bbox_to_anchor=(1, 0.75), fontsize=ftl)
#    subplot.grid()
    return subplot

# go through values and get values per level
for i in np.arange(0, nbrlevels+1, 1):

    if i == 0:
        plotCascadeLevels(ax1, t, graf[:, i], 'b',
                          'Original Mass \n (12 month)')
    if i == 1:
        plotCascadeLevels(ax2, t, graf[:, i], 'g',
                          'Level one \n (6 month)')
    if i == 2:
        plotCascadeLevels(ax3, t, graf[:, i], 'fuchsia',
                          'Level two \n (3 month)')
    if i == 3:
        plotCascadeLevels(ax4, t, graf[:, i], 'coral',
                          'Level three \n (90 days)')
    if i == 4:
        plotCascadeLevels(ax5, t, graf[:, i], 'r',
                          'Level four \n (45 days)')
    if i == 5:
        plotCascadeLevels(ax6, t, graf[:, i], 'c',
                          'Level five \n (22.5 days)')
    if i == 6:
        plotCascadeLevels(ax7, t, graf[:, i], 'lime',
                          'Level six \n (11.25 days)')
    if i == 7:
        plotCascadeLevels(ax8, t, graf[:, i], 'indigo',
                          'Level seven \n (5.625 days)')
    if i == 8:
        plotCascadeLevels(ax9, t, graf[:, i], 'steelblue',
                          'Level eight \n (2.8125 days)')
    if i == 9:
        plotCascadeLevels(ax10, t, graf[:, i], 'darkred',
                          'Level nine \n (33.75 hours)')
    if i == 10:
        plotCascadeLevels(ax11, t, graf[:, i], 'darkgreen',
                          'Level ten \n (16.875 hours)')
    if i == 11:
        plotCascadeLevels(ax12, t, graf[:, i], 'orange',
                          'Level eleven \n (8.437 hours)')
    if i == 12:
        plotCascadeLevels(ax13, t, graf[:, i], 'crimson',
                          'Level twelve \n (4.218 hours)')
#    if i == 13:
#        plotCascadeLevels(ax14, t, graf[:, i], 'crimson',
#                          'Level thirteen \n (2.109 hours)')
#    if i == 14:
#        plotCascadeLevels(ax15, t, graf[:, i], 'teal',
#                          'Level fourteen \n (1.054 hours)')


# adjust limits and axis labels and save fig
plt.xlim([1/branchnbr, lengthT])
plt.xticks(np.arange(1, lengthT+1, 30))
plt.xlabel('Days', fontsize=14)

f.text(-0.0005, 0.5, 'Rainfall (mm)',
       va='center', rotation='vertical',
       fontsize=14)
f.tight_layout()
plt.savefig('binarycascade.pdf',
            frameon=True,
            papertype='a4',
            bbox_inches='tight')
