#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:52:21 2018

@author: kremling
"""

import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 18})

def plot_errors(fname_errors=["errors.pickle"], figname=None, order_sdc=lambda n: n, order_mlsdc= lambda n: 2*n):
    # Daten einlesen
    fin = open(fname_errors, "rb")
    error = pickle.load(fin)
    fin.close()
    
    # Plotgroesse einstellen
#    plt.rcParams["figure.figsize"] = [7.5,5.]
    
    # Farben und Symbole fuer Linien und Punkte einstellen
    color = ['red', 'magenta', 'blue', 'teal', 'green']
    marker = ['x','d','o','^','s']
    
    # Grenzen der y-Achse (letzter Punkt der untersten Linie bei MLSDC, erster Fehler bei SDC)
#    ymin = error[1][(error[1]['niter_arr'][-1],error[1]['nsteps_arr'][0])]/((error[1]['nsteps_arr'][-1]/error[1]['nsteps_arr'][0])**(error[1]['niter_arr'][-1]+1))
#    ymax = error[0][(error[0]['niter_arr'][0], error[0]['nsteps_arr'][0])]
        
    f, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(15,5))
    
#    ymin = np.min(np.min(error[0]))
    
    # Plot erstellen
    for i, err in enumerate(error):            
        dt_arr = 1./np.array(err['nsteps_arr'])
        
        axes[i].set_title("{}".format(err['type']))
        
        for j, niter in enumerate(err['niter_arr']):

            # erreichte Punkte
            axes[i].plot(dt_arr, [err[(niter,n)] for n in err['nsteps_arr']], color=color[j], marker=marker[j], markersize=10, linestyle='None', label='k={}'.format(niter))
            
            # erwartete Linie: err
            if err['type'] == 'SDC':
                axes[i].plot(dt_arr, [(err[(niter,err['nsteps_arr'][0])]/((nstep/err['nsteps_arr'][0])**(order_sdc(niter)))) for nstep in err['nsteps_arr']], color=color[j])
            elif err['type'] == 'MLSDC':
                axes[i].plot(dt_arr, [(err[(niter,err['nsteps_arr'][0])]/((nstep/err['nsteps_arr'][0])**(order_mlsdc(niter)))) for nstep in err['nsteps_arr']], color=color[j])
            
        axes[i].set_xscale('log', basex=2)
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'$\Delta$t')
        axes[i].set_xlim(dt_arr[0]+dt_arr[0]/8, dt_arr[-1]-dt_arr[-1]/8)
#        start, end = axes[i].get_ylim()
#        axes[i].yaxis.set_ticks(np.linspace(start, end, 6))
        axes[i].set_yticks(axes[i].get_yticks()[::2])
#        axes[i].set_ylim([0.01,1])
    
    axes[0].set_ylabel('error')
    
    ## LEGEND
    # reference: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # at the bottom
#    handles, labels = axes[0].get_legend_handles_labels()
#    f.legend(handles, labels, loc=8, numpoints=1, ncol=5)
#    plt.subplots_adjust(bottom=0.25)
    # at the right    
    axes[1].legend(numpoints=1, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fontsize='small')
    
    plt.show()  
    
    if figname:
        f.savefig(figname, bbox_inches='tight', format="pdf")
#        plt.savefig('/home10/kremling/Documents/Studium/Gitte/Master/Seminararbeit/seminararbeit/daten/graphics/{}_{}_errors.png'.format(err['type'], dgl))
        print('figure saved in: {}'.format(figname)) 
    
if __name__ == "__main__":
    plot_errors(fname_errors="data/errors_heat1d.pickle", figname="figures/errors_heat1d.png", order_sdc=lambda n: n, order_mlsdc=lambda n: n)