#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:52:21 2018

@author: kremling
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt

# use LaTeX font style
from matplotlib import rc
rc('font',**{'family':'Helvetica','size':18})
rc('text', usetex=True)


def plot_errors(fname_errors, figname, order_sdc, order_mlsdc):
    # load data
    fin = open(fname_errors, "rb")
    error = pickle.load(fin)
    fin.close()

    # marker colors and styles
    color = ['red', 'magenta', 'blue', 'green', 'black']
    marker = ['x', 'd', 'o', '^', 's']

    # create figure
    f, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(15, 5))

    # determine ylims
    ymin1 = min([err for err in error[0].values() if isinstance(err, np.float64) and err>0])
    ymin2 = min([err for err in error[1].values() if isinstance(err, np.float64) and err>0])
    ymin = min(ymin1, ymin2)
    ymax1 = max([err for err in error[0].values() if isinstance(err, np.float64)])
    ymax2 = max([err for err in error[1].values() if isinstance(err, np.float64)])
    ymax = max(ymax1, ymax2)

    # create subplot for SDC and MLSDC
    for i, err in enumerate(error):
        dt_arr = 1./np.array(err['nsteps_arr'])

        axes[i].set_title("{}".format(err['type']))

        # plot points and lines for different number of iterations k
        for j, niter in enumerate(err['niter_arr']):

            # obtained errors
            axes[i].plot(dt_arr, [err[(niter, n)] for n in err['nsteps_arr']], color=color[j],
                         marker=marker[j], markersize=9, linestyle='None', label='k={}'.format(niter))

            # expected lines
            if err['type'] == 'SDC':
                axes[i].plot(dt_arr, [(err[(niter, err['nsteps_arr'][0])] / ((nstep/err['nsteps_arr'][0])
                    ** (order_sdc(niter)))) for nstep in err['nsteps_arr']], color=color[j],
                    linestyle='--', linewidth=1)
            elif err['type'] == 'MLSDC':
                axes[i].plot(dt_arr, [(err[(niter, err['nsteps_arr'][0])]/((nstep/err['nsteps_arr'][0])
                    ** (order_mlsdc(niter)))) for nstep in err['nsteps_arr']], color=color[j],
                    linestyle='--', linewidth=1)

        # specify axis options
        axes[i].set_xscale('log', basex=2)
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'$\Delta$t')
        axes[i].set_xlim(dt_arr[0]+dt_arr[0]/8, dt_arr[-1]-dt_arr[-1]/8)
        axes[i].set_yticks(np.power(10., np.arange(-20, 3, 2)))
        axes[i].set_ylim([ymin/10, ymax*10])

    axes[0].set_ylabel('error')

    # create legend declaring number of iterations
    # reference: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # at the bottom
#    handles, labels = axes[0].get_legend_handles_labels()
#    f.legend(handles, labels, loc=8, numpoints=1, ncol=5)
#    plt.subplots_adjust(bottom=0.25)
    # at the right
    axes[1].legend(numpoints=1, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize='small')

    plt.show()

    # save figure
    if figname:
        f.savefig(figname, bbox_inches='tight', format="pdf")
        print('figure saved in: {}'.format(figname))


if __name__ == "__main__":
    plot_errors(fname_errors="data/errors_heat1d.pickle", figname="figures/errors_heat1d.png",
                order_sdc=lambda k: min(k, 2*5), order_mlsdc=lambda k: min(2*k, 2*5))
