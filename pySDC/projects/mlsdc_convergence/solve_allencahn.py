# import required classes from pySDC ...
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from AllenCahn_1D_FD_homogeneous import allencahn_wave_fullyimplicit
from sweeper_random_initial_guess import sweeper_random_initial_guess
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.helpers.stats_helper import filter_stats, sort_stats
from plot_errors import plot_errors

# ... and other packages
import pickle
from math import log
import numpy as np
from copy import deepcopy

# code partly from pySDC/playgrounds/Allen_Cahn/AllenCahn_reference.py and AllenCahn_contracting_circle_SDC.py


def setup_parameters(restol, maxiter, initial_guess, m, n, radius, eps):
    """
    Helper routine to fill in all relevant parameters

    Returns:
        description (dict)
        controller_params (dict)
    """

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = restol
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['QI'] = ['IE']
    sweeper_params['initial_guess'] = initial_guess
    sweeper_params['num_nodes'] = m

    # This comes as read-in for the problem class
    problem_params = dict()
    problem_params['dw'] = -0.04
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1E-10
    problem_params['lin_tol'] = 1E-10
    problem_params['lin_maxiter'] = 100
    problem_params['interval'] = (-10.,10.) #-0.5,0.5
    problem_params['nvars'] = n
    problem_params['radius'] = radius
    problem_params['eps'] = eps

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
#    controller_params['hook_class'] = monitor

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = allencahn_wave_fullyimplicit
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = sweeper_random_initial_guess
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


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


def solve_allencahn(m, n, iorder, radius, eps, initial_guess, niter_arr, nsteps_arr, fname_errors):
    """
    Run SDC and MLSDC for 1D heat equation with given parameters
    and compare errors for different numbers of iterations and time steps
    """    
    # get description dict for SDC
    description_sdc, controller_params = setup_parameters(restol=0, maxiter=1, initial_guess=initial_guess, \
                                            m=m[0], n=n[0], radius=radius, eps=eps)
    
    # changes to get description dict for MLSDC
    description_mlsdc = deepcopy(description_sdc)
    description_mlsdc['sweeper_params']['num_nodes'] = m
    description_mlsdc['problem_params']['nvars'] = n
    # initialize space transfer parameters
    space_transfer_params_mlsdc = dict()
    space_transfer_params_mlsdc['rorder'] = 0
    space_transfer_params_mlsdc['iorder'] = iorder
    description_mlsdc['space_transfer_class'] = mesh_to_mesh
    description_mlsdc['space_transfer_params'] = space_transfer_params_mlsdc
    
    # set time parameters
    t0 = 0.
    
    # define error dicts for SDC and MLSDC and save parameters there (needed to plot results afterwards)
    # error_ode:  error of quadrature nodes of the last time step compared to ODE solution
    # error_coll: error of quadrature nodes of the last time step compared to solution of the collocation problem
    # error_uend: error of last quadrature node compared to ODE solution
    error_sdc = {'type' : 'SDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    error_uend_sdc = {'type' : 'SDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    
    error_mlsdc = {'type' : 'MLSDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    error_uend_mlsdc = {'type' : 'MLSDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    
    ## read in the "exact" solution (computed by SDC with res_tol 1E-14 and maxiter 100)
    fin = open("data/lsg_allencahn.pickle", "rb")
    lsg = pickle.load(fin)
    fin.close()
    
    # vary number of iterations
    for niter in niter_arr:
        # set number of iterations
        description_sdc['step_params']['maxiter'] = niter
        description_mlsdc['step_params']['maxiter'] = niter
        
        # vary length of a time step
        for i,nsteps in enumerate(nsteps_arr):
            # set time step
            dt = 1./nsteps
            description_sdc['level_params']['dt'] = dt
            description_mlsdc['level_params']['dt'] = dt
            
            # set end of time interval (only one time step is made)
            Tend=t0+dt
            
            # print current parameters
            print('niter: %d\tnsteps: %e' % (niter, 1./nsteps))
        
            # instantiate the controller for SDC and MLSDC
            controller_sdc = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description_sdc)
            controller_mlsdc = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description_mlsdc)
        
            # get initial values
            P = controller_sdc.MS[0].levels[0].prob
            uinit = P.u_exact(t0)
        
            # call main function to get things done...
            uend_sdc, stats_sdc = controller_sdc.run(u0=uinit, t0=t0, Tend=Tend)
            uend_mlsdc, stats_mlsdc = controller_mlsdc.run(u0=uinit, t0=t0, Tend=Tend)
            
            # save numerical solution as matrix
            L_sdc = controller_sdc.MS[0].levels[0]
            u_num_sdc = np.array([u.values for u in L_sdc.u])
            
            L_mlsdc = controller_mlsdc.MS[0].levels[0]
            u_num_mlsdc = np.array([u.values for u in L_mlsdc.u])
            
            # compute, save and print ode error and resulting order in dt
            uex = np.array(lsg[nsteps][1])
#            err_sdc = np.linalg.norm(uex - u_num_sdc, ord=np.inf)
            err_sdc = np.max(uex - u_num_sdc)
            error_sdc[(niter, nsteps)] = err_sdc
            order_sdc = 0 if i == 0 or error_sdc[(niter, nsteps_arr[i-1])] == 0 else \
                        log(error_sdc[(niter, nsteps_arr[i-1])]/err_sdc)/log(nsteps/nsteps_arr[i-1])
            print('SDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_sdc, order_sdc))
            
#            err_mlsdc = np.linalg.norm(uex - u_num_mlsdc, ord=np.inf)
            err_mlsdc = np.max(uex - u_num_mlsdc)
            error_mlsdc[(niter, nsteps)] = err_mlsdc
            order_mlsdc = log(error_mlsdc[(niter, nsteps_arr[i-1])]/err_mlsdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('MLSDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_mlsdc, order_mlsdc))
        
            # compute, save and print ode error at the last quadrature node
#            err_uend_sdc = np.linalg.norm(uex[-1] - uend_sdc.values)
#            err_uend_sdc = np.max(uex[-1] - uend_sdc.values)
#            error_uend_sdc[(niter, nsteps)] = err_uend_sdc
#            order_uend_sdc = log(error_uend_sdc[(niter, nsteps_arr[i-1])]/err_uend_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
##            print('SDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_sdc, order_uend_sdc))
#            
##            err_uend_mlsdc = np.linalg.norm(uex[-1] - uend_mlsdc.values)
#            err_uend_mlsdc = np.max(uex[-1] - uend_mlsdc.values)
#            error_uend_mlsdc[(niter, nsteps)] = err_uend_mlsdc
#            order_uend_mlsdc = log(error_uend_mlsdc[(niter, nsteps_arr[i-1])]/err_uend_mlsdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
##            print('MLSDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_mlsdc, order_uend_mlsdc))
    
    # compute, save and print order of the ratio between U-U^(k) and U-U^(k-1)
#    error_k_sdc = {}
#    error_k_mlsdc = {}
#    # iterate over k
#    for j, niter in enumerate(niter_arr[:-1]):
#        print("relation between U-U^%d and U-U^%d" % (niter, niter_arr[j+1]))
#        # iterate over dt
#        for i, nsteps in enumerate(nsteps_arr):
#            error_k_sdc[nsteps] = error_sdc[(niter_arr[j+1], nsteps)] / error_sdc[(niter, nsteps)]
#            order = log(error_k_sdc[nsteps_arr[i-1]]/error_k_sdc[nsteps])/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print("SDC:\tdt: %.10f\terror_k: %8.6e\torder:%4.2f" % (1./nsteps, error_k_sdc[nsteps], order))
#            
#            error_k_mlsdc[nsteps] = error_mlsdc[(niter_arr[j+1], nsteps)] / error_mlsdc[(niter, nsteps)]
#            order = log(error_k_mlsdc[nsteps_arr[i-1]]/error_k_mlsdc[nsteps])/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print("MLSDC:\tdt: %.10f\terror_k: %8.6e\torder:%4.2f" % (1./nsteps, error_k_mlsdc[nsteps], order))
    
    # save results in pickle files (needed to plot results)
    fout = open(fname_errors, "wb")
    pickle.dump([error_sdc, error_mlsdc], fout)
    fout.close()
    
    print("results saved in: {}".format(fname_errors))

    
def main():
    # set problem params
    radius = 4 #0.25
    eps = 1.2 #0.07
    n = [511, 255]
    
    # set method params
    m = [3,3]
    initial_guess = 'spread'
    # ATTENTION: change Sweeper.py to set initial value to 0 if spread is False (instead of random)
    iorder = 10
    # set number of iterations and time steps which shall be analysed
    niter_arr = range(1,6)
    nsteps_arr = [2**i for i in range(10,15)]
    
    fname_errors = "data/errors_allencahn.pickle"
    figname = "figures/errors_allencahn.png"
    
    run_reference(nsteps_arr, m[0], n[0], radius, eps)
    solve_allencahn(m, n, iorder, radius, eps, initial_guess, niter_arr, nsteps_arr, fname_errors)
    plot_errors(fname_errors, figname, order_sdc=lambda n: min(n, m[0]+1), order_mlsdc=lambda n: min(n, m[0]+1))
#    if random_init:
#        plot_errors(fname_errors, figname, order_sdc=lambda n: n, order_mlsdc=lambda n: n)
#    else:
#        plot_errors(fname_errors, figname, order_sdc=lambda n: n+1, order_mlsdc=lambda n: n if n>1 else 2*n+1)
    
    
if __name__ == "__main__":
    main()