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

matplotlib.rcParams.update({'font.size': 14})

def plot_errors(fname_errors=["errors.pickle"], figname=None, order_sdc=lambda n: n, order_mlsdc= lambda n: 2*n):
    # Daten einlesen
    fin = open(fname_errors, "rb")
    error = pickle.load(fin)
    fin.close()
    
    # Plotgroesse einstellen
#    plt.rcParams["figure.figsize"] = [7.5,5.]
    
    # Farben und Symbole fuer Linien und Punkte einstellen
    color = ['orange', 'red', 'magenta', 'blue', 'green']
    marker = ['x','d','o','^','s']
    
    # Grenzen der y-Achse (letzter Punkt der untersten Linie bei MLSDC, erster Fehler bei SDC)
#    ymin = error[1][(error[1]['niter_arr'][-1],error[1]['nsteps_arr'][0])]/((error[1]['nsteps_arr'][-1]/error[1]['nsteps_arr'][0])**(error[1]['niter_arr'][-1]+1))
#    ymax = error[0][(error[0]['niter_arr'][0], error[0]['nsteps_arr'][0])]
        
    f, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(15,5))
    
    # Plot erstellen
    for i, err in enumerate(error):            
        dt_arr = 1./np.array(err['nsteps_arr'])
        
        axes[i].set_title("{}".format(err['type']))
        
        for j, niter in enumerate(err['niter_arr']):

            # erreichte Punkte
            axes[i].plot(dt_arr, [err[(niter,n)] for n in err['nsteps_arr']], color=color[j], marker=marker[j], linestyle='None', label='k={}'.format(niter))
            
            # erwartete Linie: err
            if err['type'] == 'SDC':
                axes[i].plot(dt_arr, [(err[(niter,err['nsteps_arr'][0])]/((nstep/err['nsteps_arr'][0])**(order_sdc(niter)))) for nstep in err['nsteps_arr']], color=color[j])
            elif err['type'] == 'MLSDC':
                axes[i].plot(dt_arr, [(err[(niter,err['nsteps_arr'][0])]/((nstep/err['nsteps_arr'][0])**(order_mlsdc(niter)))) for nstep in err['nsteps_arr']], color=color[j])
            
        axes[i].set_xscale('log', basex=2)
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'$\Delta$t')
        axes[i].set_ylabel('error')
        axes[i].set_xlim(dt_arr[0]+dt_arr[0]/8, dt_arr[-1]-dt_arr[-1]/8)
#        plt.ylim(ymin*1E-1, ymax*1E2)
#        if err['type'] == 'MLSDC':
#            plt.legend(title='lines = k')
#        else:
        axes[i].legend(numpoints=1, loc="lower left")
    
    plt.show()  
    
    if figname:
        f.savefig(figname, bbox_inches='tight')
#        plt.savefig('/home10/kremling/Documents/Studium/Gitte/Master/Seminararbeit/seminararbeit/daten/graphics/{}_{}_errors.png'.format(err['type'], dgl))
        print('figure saved in: {}'.format(figname)) 
    
if __name__ == "__main__":
    plot_errors(fname_errors="data/errors_heat1d.pickle", figname="figures/errors_heat1d.png", order_sdc=lambda n: n, order_mlsdc=lambda n: n)