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

def plot_errors_mlsdc_vars(fname_errors, titles, figname, order_mlsdc, order_labels):
    # load data
    error = []
    for ferr in fname_errors:
        fin = open(ferr, "rb")
        error.append(pickle.load(fin))
        fin.close()

    # marker colors and styles
    color = ['red', 'magenta', 'blue', 'green', 'black']
    marker = ['x', 'd', 'o', '^', 's']

    # create figure
    f, axes = plt.subplots(3, 2, sharex=False, sharey=True, figsize=(15, 18))
    
    if len(error) == 5:
        axes[-1, -1].axis('off')
    
    # determine ylims
    ymin = min([min([e for e in err.values() if isinstance(e, np.float64) and e>0]) for err in error])
    ymax = max([max([e for e in err.values() if isinstance(e, np.float64) and e>0]) for err in error])

    # create subplot for each variant of SDC and MLSDC
    for i, err in enumerate(error):
        row, col = int(i/2), i%2    # row and column of subplot
        
        dt_arr = 1./np.array(err['nsteps_arr'])

        axes[row, col].set_title(titles[i], fontdict = {'family':'sans-serif'})

        # plot points and lines for different number of iterations k
        for j, niter in enumerate(err['niter_arr']):
            # obtained errors
            axes[row, col].plot(dt_arr, [err[(niter, n)] for n in err['nsteps_arr']], color=color[j],
                                marker=marker[j], markersize=8, linestyle='None', label='k={}'.format(niter),
                                markeredgecolor=color[j])

            # expected lines
            axes[row, col].plot(dt_arr, [(err[(niter, err['nsteps_arr'][0])]/((nstep/err['nsteps_arr'][0])
                                ** (order_mlsdc(niter)[i]))) for nstep in err['nsteps_arr']], color=color[j], 
                                linestyle='--', linewidth=1, label=r"order {}".format(order_labels[i]))

        # specify axis options
        axes[row, col].set_xscale('log', basex=2)
        axes[row, col].set_yscale('log')
        axes[row, col].set_xlim(dt_arr[0]+dt_arr[0]/8, dt_arr[-1]-dt_arr[-1]/8)
        axes[row, col].set_xlabel(r'$\Delta$t')
        axes[row, col].set_yticks(np.power(10., np.arange(-20, 3, 2)))
        axes[row, col].set_ylim([ymin/10, ymax*10000])
        
        # create an individual legend declaring the order of the drawn lines
        handles, labels = axes[row,col].get_legend_handles_labels()
        axes[row, col].legend([handles[1]], [labels[1]], loc=1)

    axes[0,0].set_ylabel('error')
    axes[1,0].set_ylabel('error')
    
    # remove unnecessary subplots
    if len(fname_errors) == 3:
        axes[1,1].remove()
        
    
    # specify space between subplots
    plt.subplots_adjust(hspace=0.3, wspace=0.15)

    # create legend declaring number of iterations
    # reference: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    
    # at the bottom
    f.legend(handles[::2], labels[::2], loc=8, numpoints=1, ncol=5, fontsize=18)
    plt.subplots_adjust(bottom=0.08)
    
    # at the right
#    axes[1].legend(numpoints=1, bbox_to_anchor=(1.04, 0.5), loc="center left", borderaxespad=0, fontsize='small')

    plt.show()

    # save figure
    if figname:
        f.savefig(figname, bbox_inches='tight', format="pdf")
        print('figure saved in: {}'.format(figname))


if __name__ == "__main__":
    ivp = "heat1d"
    # ivp = "allencahn_2d_space"
    # ivp = "auzinger"
    
    # ivp = "allencahn_2d_time"
    
    
    if ivp == "heat1d":
        plot_errors_mlsdc_vars( fname_errors = ["data/errors_heat1d_optimal-sdc.pickle",
                                                "data/errors_heat1d_random-sdc.pickle",
                                                "data/errors_heat1d_optimal-mlsdc.pickle",
                                                "data/errors_heat1d_dxbig-mlsdc.pickle",
                                                "data/errors_heat1d_psmall-mlsdc.pickle",
                                                "data/errors_heat1d_random-mlsdc.pickle"], 
                                titles = [r"\textbf{(a)} SDC spread initial guess",
                                          r"\textbf{(b)} SDC random initial guess",
                                          r"\textbf{(c)} MLSDC optimal parameters",
                                          r"\textbf{(d)} MLSDC coarser grid spacing $\Delta$x",
                                          r"\textbf{(e)} MLSDC lower interpolation order p",
                                          r"\textbf{(f)} MLSDC random initial guess"],
                                figname = "figures/errors_heat1d.pdf", 
                                order_mlsdc = lambda k: np.array([k, k, 2*k, k, k, k])-1,
                                order_labels = ["k-1", "k-1", "2k-1", "k-1", "k-1", "k-1"])
        
        
    
    elif ivp == "allencahn_2d_space":
        plot_errors_mlsdc_vars( fname_errors = ["data/errors_allencahn_2d_optimal-sdc.pickle",
                                                "data/errors_allencahn_2d_random-sdc.pickle",
                                                "data/errors_allencahn_2d_optimal-mlsdc.pickle",
                                                "data/errors_allencahn_2d_dxbig-mlsdc.pickle",
                                                "data/errors_allencahn_2d_psmall-mlsdc.pickle",
                                                "data/errors_allencahn_2d_random-mlsdc.pickle"], 
                                titles = [r"\textbf{(a)} SDC spread initial guess",
                                          r"\textbf{(b)} SDC random initial guess",
                                          r"\textbf{(c)} MLSDC optimal parameters",
                                          r"\textbf{(d)} MLSDC coarser grid spacing $\Delta$x",
                                          r"\textbf{(e)} MLSDC lower interpolation order p",
                                          r"\textbf{(f)} MLSDC random initial guess"],
                                figname = "figures/errors_allencahn_2d.pdf",
                                order_mlsdc = lambda k: np.array([k, k, 2*k, k, k, k])-1,
                                order_labels = ["k-1", "k-1", "2k-1", "k-1", "k-1", "k-1"])
        
        
        
    elif ivp == "auzinger":
        plot_errors_mlsdc_vars( fname_errors = ["data/errors_auzinger_optimal-sdc.pickle",
                                                "data/errors_auzinger_random-sdc.pickle",
                                                "data/errors_auzinger_optimal-mlsdc.pickle",
                                                "data/errors_auzinger_dtbig-mlsdc.pickle",
                                                "data/errors_auzinger_Msmall-mlsdc.pickle",
                                                "data/errors_auzinger_random-mlsdc.pickle"], 
                                titles = [r"\textbf{(a)} SDC spread initial guess",
                                          r"\textbf{(b)} SDC random initial guess",
                                          r"\textbf{(c)} MLSDC optimal parameters",
                                          r"\textbf{(d)} MLSDC larger time step size $\Delta$t",
                                          r"\textbf{(e)} MLSDC lower interpolation order $p$",
                                          r"\textbf{(f)} MLSDC random initial guess"],
                                figname = "figures/errors_auzinger.pdf",
                                order_mlsdc = lambda k: np.array([k, k-1, 2*k, k, k+1, k-1]),
                                order_labels = ["k", "k-1", "2k", "k", "k+1", "k-1"])
    
    
    
    elif ivp == "allencahn_2d_time":
        plot_errors_mlsdc_vars( fname_errors = ["data/errors_allencahn_2d_time_Mhigh.pickle",
                                                "data/errors_allencahn_2d_time_Mlow.pickle",
                                                "data/errors_allencahn_2d_time_Mlow_dtsmaller.pickle"], 
                                titles = [r"\textbf{(a)} optimal parameters",
                                          r"\textbf{(b)} lower interpolation order $p = M_H$",
                                          r"\textbf{(c)} lower interpolation order p and smaller $\Delta t$"],
                                figname = "figures/errors_allencahn_2d_time.pdf",
                                order_mlsdc = lambda k: [2*k, 2*k, 2*k],
                                order_labels = ["2k", "2k", "2k"])
     