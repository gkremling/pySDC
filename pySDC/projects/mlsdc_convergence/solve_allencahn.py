# import required classes from pySDC ...
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from AllenCahn_1D_FD_sin import allencahn_periodic_sin_fullyimplicit
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
import matplotlib.pyplot as plt
import operator

# code partly from pySDC/playgrounds/Allen_Cahn/AllenCahn_reference.py and AllenCahn_contracting_circle_SDC.py


def setup_parameters(restol, maxiter, initial_guess, m, n, freq, eps):
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
    problem_params['dw'] = 0.0
    problem_params['newton_maxiter'] = 100
    problem_params['newton_tol'] = 1E-13
    problem_params['lin_tol'] = 1E-13
    problem_params['lin_maxiter'] = 100
    problem_params['interval'] = (0.,1.)  # -0.5,0.5
    problem_params['nvars'] = n
    problem_params['freq'] = freq
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
    description['problem_class'] = allencahn_periodic_sin_fullyimplicit
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = sweeper_random_initial_guess
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters

    return description, controller_params


def run_reference(nsteps_arr, nnodes, nvars, freq=0.25, eps=0.04):
    """
    Routine to run particular SDC variant

    Args:
        Tend (float): end time for dumping
    """

    # load default parameters
    description, controller_params = setup_parameters(
        restol=1E-12, maxiter=50, initial_guess='spread', m=nnodes, n=nvars, freq=freq, eps=eps)

    # setup parameters "in time"
    t0 = 0.

    lsg = {}

    # verschiedene dt
    for i, nsteps in enumerate(nsteps_arr):
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

        lsg[nsteps] = [[], []]
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


def solve_allencahn(m, n, iorder, freq, eps, initial_guess, niter_arr, nsteps_arr, fname_errors):
    """
    Run SDC and MLSDC for 1D heat equation with given parameters
    and compare errors for different numbers of iterations and time steps
    """
    # get description dict for SDC
    description_sdc, controller_params = setup_parameters(restol=0, maxiter=1, initial_guess=initial_guess,
                                                          m=m[0], n=n[0], freq=freq, eps=eps)

    # changes to get description dict for MLSDC
    description_mlsdc = deepcopy(description_sdc)
    description_mlsdc['sweeper_params']['num_nodes'] = m
    description_mlsdc['problem_params']['nvars'] = n
    # initialize space transfer parameters
    space_transfer_params_mlsdc = dict()
    space_transfer_params_mlsdc['rorder'] = 0
    space_transfer_params_mlsdc['iorder'] = iorder
    space_transfer_params_mlsdc['periodic'] = True
    description_mlsdc['space_transfer_class'] = mesh_to_mesh
    description_mlsdc['space_transfer_params'] = space_transfer_params_mlsdc

    # set time parameters
    t0 = 0.

    # define error dicts for SDC and MLSDC and save parameters there (needed to plot results afterwards)
    # error_ode:  error of quadrature nodes of the last time step compared to ODE solution
    # error_coll: error of quadrature nodes of the last time step compared to solution of the collocation problem
    # error_uend: error of last quadrature node compared to ODE solution
    error_sdc = {'type': 'SDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}
    error_uend_sdc = {'type': 'SDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}

    error_mlsdc = {'type': 'MLSDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}
    error_uend_mlsdc = {'type': 'MLSDC', 'niter_arr': niter_arr, 'nsteps_arr': nsteps_arr}

    # read in the "exact" solution (computed by SDC with res_tol 1E-14 and maxiter 100)
    fin = open("data/lsg_allencahn.pickle", "rb")
    lsg = pickle.load(fin)
    fin.close()

    # vary number of iterations
    for niter in niter_arr:
        # set number of iterations
        description_sdc['step_params']['maxiter'] = niter
        description_mlsdc['step_params']['maxiter'] = niter

        # vary length of a time step
        for i, nsteps in enumerate(nsteps_arr):
            # set time step
            dt = 1./nsteps
            description_sdc['level_params']['dt'] = dt
            description_mlsdc['level_params']['dt'] = dt

            # set end of time interval (only one time step is made)
            Tend = t0 + dt

            # print current parameters
            print('niter: %d\tnsteps: %e' % (niter, 1./nsteps))

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
            order_mlsdc = log(error_mlsdc[(niter, nsteps_arr[i-1])]/err_mlsdc) / \
                log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('MLSDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_mlsdc, order_mlsdc))

            # smoothness of the error
#            e_vec = uex - u_num_sdc
#            x = np.linspace(description_sdc['problem_params']['interval'][0], description_sdc['problem_params']['interval'][1], n[0])
#            for m, e in enumerate(e_vec):
#                if m > 1:
#                    plt.plot(x, e, label=r"$\tau_{}$".format(m))
##                    plt.plot(x, uex[m], label=r"$\tau_{}$".format(m))
#                    plt.legend()
#                    plt.show()
#                    print("sum(abs(c_l)) =", np.sum(np.abs(np.fft.ifft(e))))
#                    epsm = np.cumsum(np.abs(np.fft.ifft(e)))
#                    Em = epsm[-1]*np.ones(len(epsm)) - epsm
#                    poss = [2 * np.power(l, iorder) * Em[l] / epsm[l] + (Em[l]*Em[l]) / (epsm[l]*epsm[l]) for l in range(30)]
#                    min_index, min_value = min(enumerate(poss), key=operator.itemgetter(1))
#                    print("!!Integer Overflow!!" if np.any(poss < np.zeros(len(poss))) else "")
#                    print("N_0 =", min_index)
#                    print("C(E) =", np.sqrt(min_value))
#                    print("eps_m =", epsm[min_index])

            # compute, save and print ode error at the last quadrature node
            err_uend_sdc = np.linalg.norm(uex[-1] - uend_sdc.values)
            err_uend_sdc = np.max(uex[-1] - uend_sdc.values)
            error_uend_sdc[(niter, nsteps)] = err_uend_sdc
            order_uend_sdc = log(error_uend_sdc[(niter, nsteps_arr[i-1])]/err_uend_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
#            print('SDC:\tu_end:\terror: %8.6e\torder:%4.2f' % (err_uend_sdc, order_uend_sdc))

#            err_uend_mlsdc = np.linalg.norm(uex[-1] - uend_mlsdc.values)
            err_uend_mlsdc = np.max(uex[-1] - uend_mlsdc.values)
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
    global fig
    
    # set problem params
    freq = 2
    eps = 0.2
    n = [512, 256]

    # set method params
    m = [3,3]
    initial_guess = 'spread'
    # ATTENTION: change Sweeper.py to set initial value to 0 if spread is False (instead of random)
    iorder = 8
    # set number of iterations and time steps which shall be analysed
    niter_arr = range(1,6) # 3,4
    nsteps_arr = [2**i for i in range(11,15)] # 11,12

    fname_errors = "data/errors_allencahn.pickle"
    path = "figures/errors_allencahn_"
    figdict = ["spread", "spread_dxbig", "spread_psmall", "random", "spread_freqhigh", "linear"]

    if fig in range(1,7):
        figname = path + figdict[fig-1] + ".pdf"
        if fig == 1:
            # optimal: dx small, p high, init guess smooth
            def order_sdc(k): return k
            def order_mlsdc(k): return 2*k
        elif fig == 2:
            # dx big
            n = [32,16]
            nsteps_arr = [2**i for i in range(11,16)]
            def order_sdc(k): return k
            def order_mlsdc(k): return k
        elif fig == 3:
            # p low
            iorder = 2
#            nsteps_arr = [2**i for i in range(15, 19)]
            def order_sdc(k): return k
            def order_mlsdc(k): return k
        elif fig == 4:
            # init guess not smooth
            initial_guess = "random"
            nsteps_arr = [2**i for i in range(22,27)]
            def order_sdc(k): return k
            def order_mlsdc(k): return k
        elif fig == 5:
            # init guess not smooth
            freq = 24
            nsteps_arr = [2**i for i in range(14, 18)]
            def order_sdc(k): return k
            def order_mlsdc(k): return k
        elif fig == 6:
            # linear (equals heat equ)
            eps = 0
            nsteps_arr = [2**i for i in range(7,11)]
            # more non-linear
#            eps = 0.04
#            nsteps_arr = [2**i for i in range(16,20)]
            def order_sdc(k): return k
            def order_mlsdc(k): return 2*k
    else:
        # whatsoever
        figname = "/home/kremling/Documents/Masterarbeit/master-thesis/masterarbeit/daten/graphics/errors_heat1d_spat_discr.pdf"
#        init_val = "random"
#        nu = 0.1
#        freq = 2
#        m = [5,5]
#        n = [255,127]
#        iorder = 8
        niter_arr = range(3,4)
        nsteps_arr = [2**i for i in range(7,8)]

        only_uend = True
        if only_uend:
            def order_sdc(k): return min(k, 2*m[0])-1
            def order_mlsdc(k): return min(k, 2*m[0])-1
        else:
            def order_sdc(k): return min(k, m[0]+1)-1
            def order_mlsdc(k): return min(k, m[0]+1)-1

    run_reference(nsteps_arr, m[0], n[0], freq, eps)
    solve_allencahn(m, n, iorder, freq, eps, initial_guess, niter_arr, nsteps_arr, fname_errors)
    plot_errors(fname_errors, figname, order_sdc=order_sdc, order_mlsdc=order_mlsdc)
#    if random_init:
#        plot_errors(fname_errors, figname, order_sdc=lambda n: n, order_mlsdc=lambda n: n)
#    else:
#        plot_errors(fname_errors, figname, order_sdc=lambda n: n+1, order_mlsdc=lambda n: n if n>1 else 2*n+1)


if __name__ == "__main__":
    for fig in range(1,7):
        main()
