# import required classes from pySDC ...
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.AllenCahn_2D_FD import allencahn_fullyimplicit
from pySDC.implementations.problem_classes.AllenCahn_2D_FFT import allencahn2d_imex
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

# ... and other packages
import pickle
from math import log
import numpy as np
from plot_errors import plot_errors

# code partly from pySDC/playgrounds/Allen_Cahn/AllenCahn_reference.py and AllenCahn_contracting_circle_SDC.py

def solve_allencahn(m, n, iorder, nu, freq, random_init, niter_arr, nsteps_arr, fname_errors):
    """
    Run SDC and MLSDC for 1D heat equation with given parameters
    and compare errors for different numbers of iterations and time steps
    """    
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 0
    level_params['nsweeps'] = [1]

    # initialize sweeper parameters
    sweeper_params_sdc = dict()
    sweeper_params_sdc['collocation_class'] = CollGaussRadau_Right
    sweeper_params_sdc['num_nodes'] = m[0]
    sweeper_params_sdc['QI'] = ['LU'] #IE
    sweeper_params_sdc['spread'] = not random_init
    
    sweeper_params_mlsdc = sweeper_params_sdc.copy()
    sweeper_params_mlsdc['num_nodes'] = m

    # This comes as read-in for the problem class
    problem_params_sdc = dict()
    problem_params_sdc['nu'] = 2
    problem_params_sdc['nvars'] = [(128, 128)]
    problem_params_sdc['eps'] = [0.04]
    problem_params_sdc['newton_maxiter'] = 100
    problem_params_sdc['newton_tol'] = 1E-12
    problem_params_sdc['lin_tol'] = 1E-12
    problem_params_sdc['lin_maxiter'] = 100
    problem_params_sdc['radius'] = 0.25
    
    problem_params_mlsdc = problem_params_sdc.copy()
    problem_params_mlsdc['nvars'] = [(128, 128)]

    # initialize step parameters
    step_params = dict()    
    
    # initialize space transfer parameters
    space_transfer_params_mlsdc = dict()
    space_transfer_params_mlsdc['rorder'] = 0
    space_transfer_params_mlsdc['iorder'] = iorder
    
    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
#    controller_params['predict'] = False
    
    # Fill description dictionary for easy hierarchy creation
    description_sdc = dict()
    description_sdc['problem_class'] = allencahn2d_imex #allencahn_fullyimplicit
    description_sdc['problem_params'] = problem_params_sdc
#    description_sdc['dtype_u'] = mesh
#    description_sdc['dtype_f'] = mesh
    description_sdc['sweeper_class'] = imex_1st_order #generic_implicit
    description_sdc['sweeper_params'] = sweeper_params_sdc
    
    description_mlsdc = description_sdc.copy()
    description_mlsdc['problem_params'] = problem_params_mlsdc
    description_mlsdc['sweeper_params'] = sweeper_params_mlsdc
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
        step_params['maxiter'] = niter
        description_sdc['step_params'] = step_params
        description_mlsdc['step_params'] = step_params
        
        # vary length of a time step
        for i,nsteps in enumerate(nsteps_arr):
            # set time step
            dt = 1./nsteps
            level_params['dt'] = dt
            description_sdc['level_params'] = level_params
            description_mlsdc['level_params'] = level_params
            
            # set end of time interval (only one time step is made)
            Tend=t0+dt
            
            # print current parameters
            print('niter: %d\tnsteps: %f' % (niter, 1./nsteps))
        
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
            order_sdc = log(error_sdc[(niter, nsteps_arr[i-1])]/err_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('SDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_sdc, order_sdc))
            
#            err_mlsdc = np.linalg.norm(uex - u_num_mlsdc, ord=np.inf)
            err_mlsdc = np.max(uex - u_num_mlsdc)
            error_mlsdc[(niter, nsteps)] = err_mlsdc
            order_mlsdc = log(error_mlsdc[(niter, nsteps_arr[i-1])]/err_mlsdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('MLSDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_mlsdc, order_mlsdc))
        
            # compute, save and print ode error at the last quadrature node
#            err_uend_sdc = np.linalg.norm(uex[-1] - uend_sdc.values)
            err_uend_sdc = np.max(uex[-1] - uend_sdc.values)
            error_uend_sdc[(niter, nsteps)] = err_uend_sdc
            order_uend_sdc = log(error_uend_sdc[(niter, nsteps_arr[i-1])]/err_uend_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('SDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_sdc, order_uend_sdc))
            
#            err_uend_mlsdc = np.linalg.norm(uex[-1] - uend_mlsdc.values)
            err_uend_mlsdc = np.max(uex[-1] - uend_mlsdc.values)
            error_uend_mlsdc[(niter, nsteps)] = err_uend_mlsdc
            order_uend_mlsdc = log(error_uend_mlsdc[(niter, nsteps_arr[i-1])]/err_uend_mlsdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('MLSDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_mlsdc, order_uend_mlsdc))
    
    # compute, save and print order of the ratio between U-U^(k) and U-U^(k-1)
    error_k_sdc = {}
    error_k_mlsdc = {}
    # iterate over k
    for j, niter in enumerate(niter_arr[:-1]):
        print("relation between U-U^%d and U-U^%d" % (niter, niter_arr[j+1]))
        # iterate over dt
        for i, nsteps in enumerate(nsteps_arr):
            error_k_sdc[nsteps] = error_sdc[(niter_arr[j+1], nsteps)] / error_sdc[(niter, nsteps)]
            order = log(error_k_sdc[nsteps_arr[i-1]]/error_k_sdc[nsteps])/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print("SDC:\tdt: %.10f\terror_k: %8.6e\torder:%4.2f" % (1./nsteps, error_k_sdc[nsteps], order))
            
            error_k_mlsdc[nsteps] = error_mlsdc[(niter_arr[j+1], nsteps)] / error_mlsdc[(niter, nsteps)]
            order = log(error_k_mlsdc[nsteps_arr[i-1]]/error_k_mlsdc[nsteps])/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print("MLSDC:\tdt: %.10f\terror_k: %8.6e\torder:%4.2f" % (1./nsteps, error_k_mlsdc[nsteps], order))
    
    # save results in pickle files (needed to plot results)
    fout = open(fname_errors, "wb")
    pickle.dump([error_sdc, error_mlsdc], fout)
    fout.close()
    
    print("results saved in: {}".format(fname_errors))


if __name__ == "__main__":
    # set problem params
    nu = 0.1
    freq = 24
    n = [255, 127]
    
    # set method params
    m = [5, 5]
    random_init = True
    # ATTENTION: change Sweeper.py to set initial value to 0 if spread is False (instead of random)
    iorder = 10
    # set number of iterations and time steps which shall be analysed
    niter_arr = range(1,6)
    nsteps_arr = [2**i for i in range(12,17)]
    
    fname_errors = "data/errors_allencahn.pickle"
    figname = "figures/errors_allencahn.png"
    
    solve_allencahn(m, n, iorder, nu, freq, random_init, niter_arr, nsteps_arr, fname_errors)
    if random_init:
        plot_errors(fname_errors, figname, order_sdc=lambda n: n, order_mlsdc=lambda n: n)
    else:
        plot_errors(fname_errors, figname, order_sdc=lambda n: n+1, order_mlsdc=lambda n: n if n>1 else 2*n+1)