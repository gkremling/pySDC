from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.Van_der_Pol_implicit import vanderpol
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

from pySDC.helpers.stats_helper import filter_stats, sort_stats, get_list_of_types

import pickle
from math import log
import numpy as np

def main():
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1e-12

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right # Q
    sweeper_params['num_nodes'] = 7 # M
    sweeper_params['QI'] = 'IE' # Q_delta

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1E-12
    problem_params['newton_maxiter'] = 50
    problem_params['mu'] = 5
    problem_params['u0'] = (1.0, 0)
    problem_params['nvars'] = 2

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 100

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30

    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = vanderpol
    description['problem_params'] = problem_params
    description['sweeper_class'] = generic_implicit
    description['sweeper_params'] = sweeper_params
    description['step_params'] = step_params

    # set time parameters
    t0 = 0.
    Tend = 1.0
        
    ### UM NUR UEND (BEI TEND=1) ZU VERGLEICHEN    
#    level_params['dt'] = 1e-3
#    description['level_params'] = level_params
#    
#    # instantiate the controller
#    controller = allinclusive_multigrid_nonMPI(num_procs=1, controller_params=controller_params, description=description)
#    
#    # get initial values
#    P = controller.MS[0].levels[0].prob
#    uinit = P.u_exact(t0)
#
#    # call main function to get things done...
#    uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)
#    
#    lsg = uend
    
    ### UM GANZEN VEKTOR ZU VERGLEICHEN
    lsg = {}
    
    nsteps_arr = [2**i for i in range(3,8)]
    
    #verschiedene dt
    for i,nsteps in enumerate(nsteps_arr):
        print("nsteps:{}".format(nsteps))
        
        # set time step (delta t)
        dt = 1./nsteps
        level_params['dt'] = dt
        description['level_params'] = level_params
        
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
        
#        print(lsg.values)
            
    fout = open("data/lsg_vanderpol.pickle", "wb")
    pickle.dump(lsg, fout)
    fout.close()


if __name__ == "__main__":
    main()