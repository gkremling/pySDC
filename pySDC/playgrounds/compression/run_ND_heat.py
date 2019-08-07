import numpy as np

from pySDC.helpers.stats_helper import filter_stats, sort_stats
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI
from pySDC.implementations.problem_classes.HeatEquation_ND_FD_forced_periodic import heatNd_periodic
from pySDC.implementations.sweeper_classes.imex_1st_order import imex_1st_order
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
# from pySDC.implementations.transfer_classes.TransferMesh_FFT import mesh_to_mesh_fft
from pySDC.playgrounds.compression.HookClass_error_output import error_output


def setup(dt=None, ndim=None, ml=False):

    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 1E-08
    level_params['dt'] = dt  # time-step size
    level_params['nsweeps'] = 1

    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = 3
    sweeper_params['QI'] = ['LU']  # For the IMEX sweeper, the LU-trick can be activated for the implicit part
    # sweeper_params['initial_guess'] = 'zero'

    # initialize problem parameters
    problem_params = dict()
    problem_params['ndim'] = ndim  # will be iterated over
    problem_params['order'] = 8  # order of accuracy for FD discretization in space
    problem_params['nu'] = 0.1  # diffusion coefficient
    problem_params['freq'] = tuple(2 for _ in range(ndim))  # frequencies
    if ml:
        problem_params['nvars'] = [tuple(64 for _ in range(ndim)), tuple(32 for _ in range(ndim))]  # number of dofs
    else:
        problem_params['nvars'] = tuple(64 for _ in range(ndim))  # number of dofs
    problem_params['direct_solver'] = False  # do GMRES instead of LU
    # problem_params['liniter'] = 10  # number of GMRES iterations

    # initialize step parameters
    step_params = dict()
    step_params['maxiter'] = 50
    step_params['err_tol'] = 1E-08

    # initialize space transfer parameters
    space_transfer_params = dict()
    space_transfer_params['rorder'] = 2
    space_transfer_params['iorder'] = 6

    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
    controller_params['use_iteration_estimator'] = True
    controller_params['hook_class'] = error_output

    # fill description dictionary for easy step instantiation
    description = dict()
    description['problem_class'] = heatNd_periodic  # pass problem class
    description['problem_params'] = problem_params  # pass problem parameters
    description['sweeper_class'] = imex_1st_order  # pass sweeper (see part B)
    description['sweeper_params'] = sweeper_params  # pass sweeper parameters
    description['level_params'] = level_params  # pass level parameters
    description['step_params'] = step_params  # pass step parameters
    description['space_transfer_class'] = mesh_to_mesh  # pass spatial transfer class
    # description['space_transfer_class'] = mesh_to_mesh_fft  # pass spatial transfer class
    description['space_transfer_params'] = space_transfer_params  # pass paramters for spatial transfer

    return description, controller_params


def run_simulations(ml=False, nprocs=None):
    """
    A simple test program to do SDC runs for the heat equation in various dimensions
    """

    # set time parameters
    t0 = 0.0
    Tend = 0.125

    nsteps_list = [1]#range(8, 9)
    ndim_list = [1]

    for ndim in ndim_list:
        for nsteps in nsteps_list:

            dt = (Tend - t0) / nsteps

            print(f'Working on {ndim} dimensions with time-step size {dt}...')

            description, controller_params = setup(dt, ndim, ml)

            # instantiate controller
            controller = controller_nonMPI(num_procs=nprocs, controller_params=controller_params, description=description)

            # get initial values on finest level
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)

            # call main function to get things done...
            uend, stats = controller.run(u0=uinit, t0=t0, Tend=Tend)

            # filter statistics by type (number of iterations)
            iter_counts = sort_stats(filter_stats(stats, type='niter'), sortby='time')

            niters = np.array([item[1] for item in iter_counts])
            out = f'   Mean number of iterations: {np.mean(niters):4.2f}'
            print(out)

            # filter statistics by type (error after time-step)
            errors = sort_stats(filter_stats(stats, type='error_after_step'), sortby='time')
            for err in errors:
                out = f'   Error after step {err[0]:8.4f}: {err[1]:8.4e}'
                print(out)

            # filter statistics by type (error after time-step)
            timing = sort_stats(filter_stats(stats, type='timing_run'), sortby='time')
            out = f'...done, took {timing[0][1]} seconds!'
            print(out)

            print()
        print()


if __name__ == "__main__":
    run_simulations(ml=False, nprocs=1)
    run_simulations(ml=True, nprocs=1)
