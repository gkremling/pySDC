# import required classes from pySDC ...
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.collocation_classes.gauss_legendre import CollGaussLegendre
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

    sweeper_params_mlsdc = sweeper_params_sdc.copy()
    sweeper_params_mlsdc['num_nodes'] = m

    # initialize problem parameters
    problem_params_sdc = dict()
    problem_params_sdc['nu'] = nu       # diffusion coefficient
    problem_params_sdc['freq'] = freq   # frequency for the test value
    problem_params_sdc['nvars'] = n[0]  # number of degrees of freedom

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
    error_ode_sdc = {'type': 'SDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}
    error_coll_sdc = {'type': 'SDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}
    error_uend_sdc = {'type': 'SDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}

    error_ode_mlsdc = {'type': 'MLSDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}
    error_coll_mlsdc = {'type': 'MLSDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}
    error_uend_mlsdc = {'type': 'MLSDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}

    u_coll = {}
    error_coll = {}

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
#            Tend = t0 + dt                  # only one time step is made (LTE / consistency)
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

            # compute ode solution (by calling the function "u_exact()" of the problem at all quadrature nodes)
            nodes = [0]
            nodes.extend(L_sdc.sweep.coll.nodes)
            u_ode = np.array([P.u_exact(L_sdc.time + c*L_sdc.dt).values for c in nodes])
            uend_ode = u_ode[-1]

            # compute, save and print ode error and resulting order in dt
            err_ode_sdc = np.linalg.norm(u_ode - u_num_sdc, ord=np.inf)
            error_ode_sdc[(niter, nsteps)] = err_ode_sdc
            order_ode_sdc = log(error_ode_sdc[(niter, nsteps_arr[i-1])]/err_ode_sdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('SDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_ode_sdc, order_ode_sdc))

            err_ode_mlsdc = np.linalg.norm(u_ode - u_num_mlsdc, ord=np.inf)
            error_ode_mlsdc[(niter, nsteps)] = err_ode_mlsdc
            order_ode_mlsdc = log(error_ode_mlsdc[(niter, nsteps_arr[i-1])] / err_ode_mlsdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('MLSDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_ode_mlsdc, order_ode_mlsdc))

            # compute solution and error of the collocation problem
            if not nsteps in u_coll:
                I_dtQxA = np.eye((m[0]+1)*n[0]) - L_sdc.dt*np.kron(L_sdc.sweep.coll.Qmat, P.A.toarray())
                u0 = np.kron(np.ones(m[0]+1), uinit.values)
                u_coll[nsteps] = np.linalg.solve(I_dtQxA, u0)
                err_coll = np.linalg.norm(u_coll[nsteps] - u_ode.flatten(), ord=np.inf)
                error_coll[nsteps] = err_coll

            # compute, save and print collocation error and resulting order in dt
            err_coll_sdc = np.linalg.norm(u_coll[nsteps] - u_num_sdc.flatten(), ord=np.inf)
            error_coll_sdc[(niter, nsteps)] = err_coll_sdc
            order_coll_sdc = log(error_coll_sdc[(niter, nsteps_arr[i-1])] / err_coll_sdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('SDC:\tu_coll:\terror: %8.6e\torder:%4.2f' % (err_coll_sdc, order_coll_sdc))

            err_coll_mlsdc = np.linalg.norm(u_coll[nsteps] - u_num_mlsdc.flatten(), ord=np.inf)
            error_coll_mlsdc[(niter, nsteps)] = err_coll_mlsdc
            order_coll_mlsdc = log(error_coll_mlsdc[(niter, nsteps_arr[i-1])] / err_coll_mlsdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('MLSDC:\tu_coll:\terror: %8.6e\torder:%4.2f' % (err_coll_mlsdc, order_coll_mlsdc))

            # compute, save and print ode error at the last quadrature node
            err_uend_sdc = np.linalg.norm(uend_ode - uend_sdc.values, ord=np.inf)
            error_uend_sdc[(niter, nsteps)] = err_uend_sdc
            order_uend_sdc = log(error_uend_sdc[(niter, nsteps_arr[i-1])] / err_uend_sdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('SDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_sdc, order_uend_sdc))

            err_uend_mlsdc = np.linalg.norm(uend_ode - uend_mlsdc.values, ord=np.inf)
            error_uend_mlsdc[(niter, nsteps)] = err_uend_mlsdc
            order_uend_mlsdc = log(error_uend_mlsdc[(niter, nsteps_arr[i-1])] /
                                   err_uend_mlsdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('MLSDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_mlsdc, order_uend_mlsdc))


    # compute and print order of the collocation problem
    for i, nsteps in enumerate(nsteps_arr):
        order_coll = log(error_coll[nsteps_arr[i-1]] / error_coll[nsteps]) / \
            log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
        print('COLL:\terror: %8.6e\torder:%4.2f' % (error_coll[nsteps], order_coll))

    # compute, save and print order of the ratio between U-U^(k) and U-U^(k-1)
#    error_k_sdc = {}
#    error_k_mlsdc = {}
#    # iterate over k
#    for j, niter in enumerate(niter_arr[:-1]):
#        print("relation between U-U^%d and U-U^%d" % (niter, niter_arr[j+1]))
#        # iterate over dt
#        for i, nsteps in enumerate(nsteps_arr):
#            error_k_sdc[nsteps] = error_coll_sdc[(niter_arr[j+1], nsteps)] / error_coll_sdc[(niter, nsteps)]
#            order = log(error_k_sdc[nsteps_arr[i-1]]/error_k_sdc[nsteps])/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print("SDC:\tdt: %.10f\terror_k: %8.6e\torder:%4.2f" % (1./nsteps, error_k_sdc[nsteps], order))
#
#            error_k_mlsdc[nsteps] = error_coll_mlsdc[(niter_arr[j+1], nsteps)] / error_coll_mlsdc[(niter, nsteps)]
#            order = log(error_k_mlsdc[nsteps_arr[i-1]]/error_k_mlsdc[nsteps])/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print("MLSDC:\tdt: %.10f\terror_k: %8.6e\torder:%4.2f" % (1./nsteps, error_k_mlsdc[nsteps], order))

    # save results in pickle files (needed to plot results)
    fout = open(fname_errors, "wb")
    if only_uend:
        pickle.dump([error_uend_sdc, error_uend_mlsdc], fout)
    else:
        pickle.dump([error_ode_sdc, error_ode_mlsdc], fout)
    fout.close()

    print("results saved in: {}".format(fname_errors))


def main():
    global fig

    # set problem params
    nu = 0.1
    freq = 4  # 24
    n = [255, 127]

    # set method params
    m = [5, 5]
    init_val = "zero"  # "spread"
    iorder = 8
    
    # set number of iterations and time steps which shall be analysed
    niter_arr = range(1,6)
    nsteps_arr = [2**i for i in range(6,10)]  # 15,19

    only_uend = True
    
    respath = "data/errors_heat1d_"
    figpath = "figures/errors_heat1d_"
    figoption = ["spread", "spread_dxbig", "spread_psmall", "random", "spread_freqhigh", "temp_discr", "temp_discr_uend"]

    if 1 <= fig and fig <= 7:
        figname = figpath + figoption[fig-1] + ".pdf"
        fname_errors = respath + figoption[fig-1] + ".pickle"
        
        if fig == 1:
            # optimal params: dx small, p high, init guess smooth
            def order_sdc(k): return min(k, 2*m[0])-1
            def order_mlsdc(k): return min(2*k, 2*m[0])-1
        elif fig == 2:
            # dx big
            n = [15, 7]
            nsteps_arr = [2**i for i in range(7, 11)]
            def order_sdc(k): return min(k, 2*m[0])-1
            def order_mlsdc(k): return min(k, 2*m[0])-1
        elif fig == 3:
            # p low
            iorder = 4
            nsteps_arr = [2**i for i in range(15, 19)]
            def order_sdc(k): return min(k, 2*m[0])-1
            def order_mlsdc(k): return min(k, 2*m[0])-1
        elif fig == 4:
            # random initial guess
            init_val = "random"
            nsteps_arr = [2**i for i in range(16, 20)]
            def order_sdc(k): return min(k, 2*m[0])-1
            def order_mlsdc(k): return min(k, 2*m[0])-1
        elif fig == 5:
            # initial guess not smooth
            freq = 24
            nsteps_arr = [2**i for i in range(14, 18)]
            def order_sdc(k): return min(k, 2*m[0])-1
            def order_mlsdc(k): return min(k, 2*m[0])-1
        elif fig in [6,7]:
            freq = 16
            n = [31, 31]
            m = [3, 1]
            init_val = "random"
            nsteps_arr = [2**i for i in range(10,14)]
            if fig == 6:
                def order_sdc(k): return min(k, m[0]+1)
                def order_mlsdc(k): return min(k, m[0]+1)
            elif fig == 7:
                niter_arr = range(3,8)
                only_uend = True
                def order_sdc(k): return min(k+1, 2*m[0]+1)
                def order_mlsdc(k): return min(k+1, 2*m[0]+1)
    else:
        # whatsoever
        figname = None
        m = [8,6]
        n = [256,256]
#        nsteps_arr = [2**i for i in range(10,14)]
        
        def order_sdc(n): return min(n, 2*m[0])-1
        def order_mlsdc(n): return min(2*n, 2*m[0])-1
        
        fname_errors = "errors_auzinger.pickle"
        figname = None


    solve_heat1d(m, n, iorder, nu, freq, init_val, niter_arr, nsteps_arr, only_uend, fname_errors)
    plot_errors(fname_errors, figname=figname, order_sdc=order_sdc, order_mlsdc=order_mlsdc)


if __name__ == "__main__":
    for fig in range(1,5):
        main()
