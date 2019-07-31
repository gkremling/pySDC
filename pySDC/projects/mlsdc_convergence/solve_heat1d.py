# import required classes from pySDC ...
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from sweeper_random_initial_guess import sweeper_random_initial_guess
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.helpers.stats_helper import get_list_of_types

# ... and other packages
import pickle
from math import log
import numpy as np
from plot_errors import plot_errors

def solve_heat1d(m, n, iorder, nu, freq, init_val, niter_arr, nsteps_arr, only_uend, fname_errors):
    """
    Run SDC and MLSDC for 1D heat equation with given parameters
    and compare errors for different numbers of iterations and time steps
    """    
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 0
    
    # initialize sweeper parameters
    sweeper_params_sdc = dict()
    sweeper_params_sdc['collocation_class'] = CollGaussRadau_Right
    sweeper_params_sdc['num_nodes'] = m[0]
    sweeper_params_sdc['QI'] = 'IE'
    sweeper_params_sdc['initial_guess'] = init_val
    print(init_val)
    
    sweeper_params_mlsdc = sweeper_params_sdc.copy()
    sweeper_params_mlsdc['num_nodes'] = m
    
    # initialize problem parameters
    problem_params_sdc = dict()
    problem_params_sdc['nu'] = nu # diffusion coefficient
    problem_params_sdc['freq'] = freq # frequency for the test value
    problem_params_sdc['nvars'] = n[0] # number of degrees of freedom
    
    problem_params_mlsdc = problem_params_sdc.copy()
    problem_params_mlsdc['nvars'] = n
    
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
    description_sdc['problem_class'] = heat1d
    description_sdc['problem_params'] = problem_params_sdc
#    description_sdc['dtype_u'] = mesh
#    description_sdc['dtype_f'] = mesh
    description_sdc['sweeper_class'] = sweeper_random_initial_guess
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
    error_ode_sdc = {'type' : 'SDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    error_coll_sdc = {'type' : 'SDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    error_uend_sdc = {'type' : 'SDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    
    error_ode_mlsdc = {'type' : 'MLSDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    error_coll_mlsdc = {'type' : 'MLSDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    error_uend_mlsdc = {'type' : 'MLSDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    
    u_coll = {}    
    
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
            Tend=t0+3*dt
            
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
            
#            print(np.linalg.norm(controller_mlsdc.MS[0].levels[0].sweep.QI, ord=np.inf))
#            print(np.linalg.norm(controller_mlsdc.MS[0].levels[1].sweep.QI, ord=np.inf))
            
            # compute ode solution (by calling the exact function of the problem at all quadrature nodes)
            nodes = [0]
            nodes.extend(L_sdc.sweep.coll.nodes)
            u_ode = np.array([P.u_exact(L_sdc.time + c*L_sdc.dt).values for c in nodes])
            
            # compute, save and print ode error and resulting order in dt
            err_ode_sdc = np.linalg.norm(u_ode - u_num_sdc, ord=np.inf)
            error_ode_sdc[(niter, nsteps)] = err_ode_sdc
            order_ode_sdc = log(error_ode_sdc[(niter, nsteps_arr[i-1])]/err_ode_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('SDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_ode_sdc, order_ode_sdc))
            
            err_ode_mlsdc = np.linalg.norm(u_ode - u_num_mlsdc, ord=np.inf)
            error_ode_mlsdc[(niter, nsteps)] = err_ode_mlsdc
            order_ode_mlsdc = log(error_ode_mlsdc[(niter, nsteps_arr[i-1])]/err_ode_mlsdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('MLSDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_ode_mlsdc, order_ode_mlsdc))
        
            # compute solution of the collocation problem
            if not nsteps in u_coll:
                I_dtQxA = np.eye((m[0]+1)*n[0]) - L_sdc.dt*np.kron(L_sdc.sweep.coll.Qmat, P.A.toarray())
                u0 = np.kron(np.ones(m[0]+1), uinit.values)
                u_coll[nsteps] = np.linalg.solve(I_dtQxA, u0)
            
            # compute, save and print collocation error and resulting order in dt
            err_coll_sdc = np.linalg.norm(u_coll[nsteps] - u_num_sdc.flatten(), ord=np.inf)
            error_coll_sdc[(niter, nsteps)] = err_coll_sdc
            order_coll_sdc = log(error_coll_sdc[(niter, nsteps_arr[i-1])]/err_coll_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('SDC:\tu_coll:\terror: %8.6e\torder:%4.2f' % (err_coll_sdc, order_coll_sdc))
            
#            err_coll_mlsdc = np.linalg.norm(u_coll[nsteps] - u_num_mlsdc.flatten(), ord=np.inf)
#            err_coll_mlsdc = np.linalg.norm(u_coll[nsteps][-n[0]:] - u_ode[-1], ord=np.inf)
            err_coll_mlsdc = np.linalg.norm(u_coll[nsteps] - u_ode.flatten(), ord=np.inf)
            error_coll_mlsdc[(niter, nsteps)] = err_coll_mlsdc
            order_coll_mlsdc = log(error_coll_mlsdc[(niter, nsteps_arr[i-1])]/err_coll_mlsdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('MLSDC:\tu_coll:\terror: %8.6e\torder:%4.2f' % (err_coll_mlsdc, order_coll_mlsdc))
            print('COLL:\tu_end\terror: %8.6e\torder:%4.2f' % (err_coll_mlsdc, order_coll_mlsdc))
            
            # compute, save and print ode error at the last quadrature node
            err_uend_sdc = np.linalg.norm(u_ode[-1] - uend_sdc.values)
            error_uend_sdc[(niter, nsteps)] = err_uend_sdc
            order_uend_sdc = log(error_uend_sdc[(niter, nsteps_arr[i-1])]/err_uend_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('SDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_sdc, order_uend_sdc))
            
            err_uend_mlsdc = np.linalg.norm(u_ode[-1] - uend_mlsdc.values)
            error_uend_mlsdc[(niter, nsteps)] = err_uend_mlsdc
            order_uend_mlsdc = log(error_uend_mlsdc[(niter, nsteps_arr[i-1])]/err_uend_mlsdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('MLSDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_mlsdc, order_uend_mlsdc))
    
    # compute, save and print order of the ratio between U-U^(k) and U-U^(k-1)
#    error_k_sdc = {}
#    error_k_mlsdc = {}
#    # iterate over k
#    for j, niter in enumerate(niter_arr[:-1]):
#        print("relation between U-U^%d and U-U^%d" % (niter, niter_arr[j+1]))
#        # iterate over dt
#        for i, nsteps in enumerate(nsteps_arr):
#            error_k_sdc[nsteps] = error_ode_sdc[(niter_arr[j+1], nsteps)] / error_ode_sdc[(niter, nsteps)]
#            order = log(error_k_sdc[nsteps_arr[i-1]]/error_k_sdc[nsteps])/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print("SDC:\tdt: %.10f\terror_k: %8.6e\torder:%4.2f" % (1./nsteps, error_k_sdc[nsteps], order))
#            
#            error_k_mlsdc[nsteps] = error_ode_mlsdc[(niter_arr[j+1], nsteps)] / error_ode_mlsdc[(niter, nsteps)]
#            order = log(error_k_mlsdc[nsteps_arr[i-1]]/error_k_mlsdc[nsteps])/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print("MLSDC:\tdt: %.10f\terror_k: %8.6e\torder:%4.2f" % (1./nsteps, error_k_mlsdc[nsteps], order))
    
    # save results in pickle files (needed to plot results)
    fout = open(fname_errors, "wb")
    if only_uend:
        pickle.dump([error_uend_sdc, error_uend_mlsdc], fout)
    else:
        pickle.dump([error_ode_sdc, error_ode_mlsdc], fout)
    fout.close()
    
#    print([t0+i*dt for i in range(0,4)])
#    print(len(controller_sdc.MS))
#    print([l.time for l in controller_sdc.MS])
#    print(uend_sdc.values)
    
    print("results saved in: {}".format(fname_errors))


def main():
    global fig
    
    # set problem params
    nu = 0.1
    freq = 4 #24
    n = [255, 127]
    
    # set method params
    m = [5,5]
    init_val = "spread" #"spread"
    iorder = 8
    # set number of iterations and time steps which shall be analysed
    niter_arr = range(1,6)
    nsteps_arr = [2**i for i in range(6,10)] #15,19
    
    only_uend = False
    
    if only_uend:
        fname_errors = "data/errors_heat1d_uend.pickle"
        figname = "figures/errors_heat1d_uend.png"
    else:
        fname_errors = "data/errors_heat1d.pickle"
        figname = "figures/errors_heat1d.png"
    
    path = "/home/kremling/Documents/Masterarbeit/presentation-scicade/daten/graphics/errors_heat1d_"
    figdict = ["space_spread", "space_spread_psmall", "space_spread_dxbig", "random", "spread_freqhigh"]
    
    if 1 <= fig and fig <= 5:
        figname = path + figdict[fig-1]
        if fig == 1:
            order_sdc=lambda k: min(k, m[0]+1)
            order_mlsdc=lambda k: min(2*k, m[0]+1)
        elif fig == 2:
            iorder = 4
            nsteps_arr = [2**i for i in range(14,18)]
            order_sdc=lambda k: min(k, m[0]+1)
            order_mlsdc=lambda k: min(k, m[0]+1)
        elif fig == 3:
            n = [15, 7]
            order_sdc=lambda k: min(k, m[0]+1)
            order_mlsdc=lambda k: min(k, m[0]+1)
        elif fig == 4:
            init_val = "random"
            nsteps_arr = [2**i for i in range(16,20)]
            order_sdc=lambda k: min(k, m[0]+1)
            order_mlsdc=lambda k: min(k, m[0]+1)
        elif fig == 5:
            freq = 24
            nsteps_arr = [2**i for i in range(14,18)]
            order_sdc = lambda k: min(k+1, m[0]+1)
            order_mlsdc = lambda k: min(k, m[0]+1)
    else:
        #whatsoever
        only_uend = False
        init_val = "random"
        nu = 0.01
        freq = 2
        m = [2,1]
        n = [127,127]
        niter_arr = range(1,6)
        nsteps_arr = [2**i for i in range(16,22)]
        order_sdc=lambda k: min(k, m[0]+1)
        order_mlsdc=lambda k: min(k, m[0]+1)
    
    solve_heat1d(m, n, iorder, nu, freq, init_val, niter_arr, nsteps_arr, only_uend, fname_errors)
    plot_errors(fname_errors, figname=None, order_sdc=order_sdc, order_mlsdc=order_mlsdc)
    

if __name__ == "__main__":
#    for fig in range(1,6):
    fig = 0
    main()
    
    