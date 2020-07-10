# import required classes from pySDC ...
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.collocation_classes.gauss_legendre import CollGaussLegendre
from pySDC.implementations.problem_classes.Auzinger_implicit import auzinger
from sweeper_random_initial_guess import sweeper_random_initial_guess
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

# ... and other packages
import pickle
from math import log
import numpy as np
from plot_errors import plot_errors


def solve_auzinger(m, random_init, niter_arr, nsteps_arr, only_uend, fname_errors):
    """
    Run SDC and MLSDC for auzinger ODE with given parameters
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
    sweeper_params_sdc['initial_guess'] = 'random' if random_init else 'zero'

    sweeper_params_mlsdc = sweeper_params_sdc.copy()
    sweeper_params_mlsdc['num_nodes'] = m

    # initialize problem parameters
    problem_params = dict()
    problem_params['newton_tol'] = 1E-14
    problem_params['newton_maxiter'] = 100
    problem_params['nvars'] = 2
    problem_params['stiffness'] = 0.75

    # initialize step parameters
    step_params = dict()

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
#    controller_params['predict'] = False

    # Fill description dictionary for easy hierarchy creation
    description_sdc = dict()
    description_sdc['problem_class'] = auzinger
    description_sdc['problem_params'] = problem_params
    description_sdc['sweeper_class'] = sweeper_random_initial_guess
    description_sdc['sweeper_params'] = sweeper_params_sdc

    description_mlsdc = description_sdc.copy()
    description_mlsdc['problem_params'] = problem_params
    description_mlsdc['sweeper_params'] = sweeper_params_mlsdc
    description_mlsdc['space_transfer_class'] = mesh_to_mesh

    # set time parameters
    t0 = 0.

    # define error dicts for SDC and MLSDC and save parameters there (needed to plot results afterwards)
    # error:  error of quadrature nodes of the last time step compared to ODE solution
    # error_uend: error of last quadrature node compared to ODE solution
    error_sdc = {'type': 'SDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}
    error_uend_sdc = {'type': 'SDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}

    error_mlsdc = {'type': 'MLSDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}
    error_uend_mlsdc = {'type': 'MLSDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}

    # vary number of iterations (k)
    for niter in niter_arr:
        # set number of iterations
        step_params['maxiter'] = niter
        description_sdc['step_params'] = step_params
        description_mlsdc['step_params'] = step_params

        # vary length of a time step (dt)
        for i, nsteps in enumerate(nsteps_arr):
            # set time step
            dt = 1./nsteps
            level_params['dt'] = dt
            description_sdc['level_params'] = level_params
            description_mlsdc['level_params'] = level_params

            # set end of time interval
            # Tend = t0 + dt                  # only one time step is made (LTE / consistency)
            Tend = t0 + 50./nsteps_arr[-1]  # several time steps are made (convergence)

            # print current parameters
            print('niter: %d\tnsteps: %f' % (niter, 1./nsteps))

            # instantiate the controller for SDC and MLSDC
            controller_sdc = controller_nonMPI(
                num_procs=1, controller_params=controller_params, description=description_sdc)
            controller_mlsdc = controller_nonMPI(
                num_procs=1, controller_params=controller_params, description=description_mlsdc)

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

            # compute ode solution (by calling the exact function of the problem at all quadrature nodes)
            nodes = [0]
            nodes.extend(L_sdc.sweep.coll.nodes)
            uex = np.array([P.u_exact(L_sdc.time + c*L_sdc.dt).values for c in nodes])
            uex_end = uex[-1]

            # compute, save and print ode error and resulting order in dt
            err_sdc = np.linalg.norm(uex - u_num_sdc, ord=np.inf)
            error_sdc[(niter, nsteps)] = err_sdc
            order_sdc = log(error_sdc[(niter, nsteps_arr[i-1])]/err_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('SDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_sdc, order_sdc))

            err_mlsdc = np.linalg.norm(uex - u_num_mlsdc, ord=np.inf)
            error_mlsdc[(niter, nsteps)] = err_mlsdc
            order_mlsdc = log(error_mlsdc[(niter, nsteps_arr[i-1])]/err_mlsdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('MLSDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_mlsdc, order_mlsdc))

            # compute, save and print ode error at the last quadrature node
            err_uend_sdc = np.linalg.norm(uex_end - uend_sdc.values)
            error_uend_sdc[(niter, nsteps)] = err_uend_sdc
            order_uend_sdc = log(error_uend_sdc[(niter, nsteps_arr[i-1])] / err_uend_sdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            # print('SDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_sdc, order_uend_sdc))

            err_uend_mlsdc = np.linalg.norm(uex_end - uend_mlsdc.values)
            error_uend_mlsdc[(niter, nsteps)] = err_uend_mlsdc
            order_uend_mlsdc = log(error_uend_mlsdc[(niter, nsteps_arr[i-1])] / err_uend_mlsdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            # print('MLSDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_mlsdc, order_uend_mlsdc))

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
    fout = open(fname_errors[0], "wb")
    if only_uend:
        pickle.dump(error_uend_sdc, fout)
    else:
        pickle.dump(error_sdc, fout)
    fout.close()
    
    fout = open(fname_errors[1], "wb")
    if only_uend:
        pickle.dump(error_uend_mlsdc, fout)
    else:
        pickle.dump(error_mlsdc, fout)
    fout.close()

    print("results saved in: {}".format(fname_errors))


def main():
    global fig
    
    # set method params
    m = [8,6]  # 12,10
    random_init = False
    
    # set number of iterations and time steps which shall be analysed
    niter_arr = range(1, 6)
    nsteps_arr = [2**i for i in range(3,7)]

    only_uend = False
    
    respath = "data/errors_auzinger_"
    figpath = "figures/errors_auzinger_"
    
    figoption = ["optimal", "dtbig", "Msmall", "random"]

    if fig in range(1,5):
        figname = figpath + figoption[fig-1] + ".pdf"
        prefix = respath  + figoption[fig-1] 
        fname_errors = [prefix + "-sdc.pickle", prefix + "-mlsdc.pickle"]
        
        if fig == 1:
            # optimal params: dt small, p high, init guess smooth
            def order_sdc(n): return min(n, 2*m[0])
            def order_mlsdc(n): return min(2*n, 2*m[0])
        elif fig == 2:
            # dt big
            nsteps_arr = [2**i for i in range(1,5)] 
            def order_sdc(n): return min(n, 2*m[0])
            def order_mlsdc(n): return min(2*n, 2*m[0])
        elif fig == 3:
            # p low
            m = [8,2]        
            def order_sdc(n): return min(n, 2*m[0])
            def order_mlsdc(n): return min(n+1, 2*m[0])
        elif fig == 4:
            # random initial guess
            random_init = True
            def order_sdc(n): return min(n, 2*m[0])-1
            def order_mlsdc(n): return min(n, 2*m[0])-1
    else:
        # whatsoever
        random_init = False
        m = [6,4]
        nsteps_arr = [2**i for i in range(2,6)] 
        niter_arr = range(1,6)
        only_uend = False
        
        def order_sdc(n): return min(n, 2*m[0])
        def order_mlsdc(n): return min(2*n, 2*m[0])
        
        fname_errors = "errors_auzinger.pickle"
        figname = None

    solve_auzinger(m, random_init, niter_arr, nsteps_arr, only_uend, fname_errors)
    plot_errors(fname_errors, figname=figname, order_sdc=order_sdc, order_mlsdc=order_mlsdc)


if __name__ == "__main__":    
    for fig in range(1,5):
        main()
