import numpy as np
import pickle

import pySDC.helpers.plot_helper as plt_helper
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from AllenCahn_1D_FD_homogeneous import allencahn_wave_fullyimplicit
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.explicit import explicit
from pySDC.playgrounds.Allen_Cahn.AllenCahn_monitor_Bayreuth import monitor
from solve_allencahn import setup_parameters


def run_reference(nsteps_arr, nnodes, nvars, radius=0.25, eps=0.04):
    """
    Routine to run particular SDC variant

    Args:
        Tend (float): end time for dumping
    """

    # load default parameters
    description, controller_params = setup_parameters(restol=1E-10, maxiter=50, initial_guess='zero', m=nnodes, n=nvars, radius=radius, eps=eps)

    # setup parameters "in time"
    t0 = 0.
    
    lsg = {}
    
    #verschiedene dt
    for i,nsteps in enumerate(nsteps_arr):
        print("nsteps:{}".format(nsteps))
        
        # set time step (delta t)
        dt = 1./nsteps
        description['level_params']['dt'] = dt
        
        Tend = t0+dt
    
        # instantiate the controller
        controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
    
        # get initial values
        P = controller.MS[0].levels[0].prob
        uinit = P.u_exact(t0)
    
        # call main function to get things done...
        uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
        
        # lsg speichern
        L = controller.MS[0].levels[0]
        nodes = [0]
        nodes.extend(L.sweep.coll.nodes)
        
        lsg[nsteps] = [[],[]]
        for j, node in enumerate(nodes):
            lsg[nsteps][0].append(L.time+node*L.dt)
            lsg[nsteps][1].append(L.u[j].values)
#            print('t={}\tu={}'.format(L.time+node*L.dt, L.u[j].values))
            
            
        # filter statistics by first time intervall and type (residual)
        filtered_stats = filter_stats(stats, time=t0, type='residual_post_iteration')
    
        # sort and convert stats to list, sorted by iteration numbers
        residuals = sort_stats(filtered_stats, sortby='iter')
    
        for item in residuals:
            out = 'Residual in iteration %2i: %8.4e' % item
            print(out)
    
        # filter statistics by type (number of iterations)
        filtered_stats = filter_stats(stats, type='niter')
    
        # convert filtered statistics to list of iterations count, sorted by time
        iter_counts = sort_stats(filtered_stats, sortby='time')
    
        for item in iter_counts:
            out = 'Number of iterations at time %4.2f: %2i' % item
            print(out)
            
    fout = open("data/lsg_allencahn.pickle", "wb")
    pickle.dump(lsg, fout)
    fout.close()


def main(cwd=''):
    """
    Main driver

    Args:
        cwd (str): current working directory (need this for testing)
    """
    nsteps_arr = [2**i for i in range(10,20)]
    nnodes = 5
    nvars = 127
    run_reference(nsteps_arr, nnodes, nvars)



if __name__ == "__main__":
    main()