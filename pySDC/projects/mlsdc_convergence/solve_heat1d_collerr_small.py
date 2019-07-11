# import required classes from pySDC ...
from pySDC.implementations.collocation_classes.gauss_radau_right import CollGaussRadau_Right
from pySDC.implementations.problem_classes.HeatEquation_1D_FD import heat1d
from sweeper_random_initial_guess import sweeper_random_initial_guess
from pySDC.implementations.transfer_classes.TransferMesh import mesh_to_mesh
from pySDC.implementations.controller_classes.controller_nonMPI import controller_nonMPI

# ... and other packages
import pickle
from math import log
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 18})

def solve_heat1d(m, n, nu, freq, random_init, niter_arr, nsteps_arr):
    """
    Run SDC and MLSDC for 1D heat equation with given parameters
    and compare errors for different numbers of iterations and time steps
    """    
    # initialize level parameters
    level_params = dict()
    level_params['restol'] = 0
    
    # initialize sweeper parameters
    sweeper_params = dict()
    sweeper_params['collocation_class'] = CollGaussRadau_Right
    sweeper_params['num_nodes'] = m
    sweeper_params['QI'] = 'IE'
    sweeper_params['initial_guess'] = 'random' if random_init else 'spread'
    
    # initialize problem parameters
    problem_params = dict()
    problem_params['nu'] = nu # diffusion coefficient
    problem_params['freq'] = freq # frequency for the test value
    problem_params['nvars'] = n # number of degrees of freedom
    
    # initialize step parameters
    step_params = dict()
    
    # initialize controller parameters
    controller_params = dict()
    controller_params['logger_level'] = 30
#    controller_params['predict'] = False
    
    # Fill description dictionary for easy hierarchy creation
    description = dict()
    description['problem_class'] = heat1d
    description['problem_params'] = problem_params
#    description_sdc['dtype_u'] = mesh
#    description_sdc['dtype_f'] = mesh
    description['sweeper_class'] = sweeper_random_initial_guess
    description['sweeper_params'] = sweeper_params
    
    # set time parameters
    t0 = 0.
    
    # define error dicts for SDC and MLSDC and save parameters there (needed to plot results afterwards)
    # error_ode:  error of quadrature nodes of the last time step compared to ODE solution
    # error_coll: error of quadrature nodes of the last time step compared to solution of the collocation problem
    # error_uend: error of last quadrature node compared to ODE solution
    error = {'type' : 'SDC', 'niter_arr' : niter_arr, 'nsteps_arr' : nsteps_arr}
    
    # vary length of a time step
    for i,nsteps in enumerate(nsteps_arr):
        # set time step
        dt = 1./nsteps
        level_params['dt'] = dt
        description['level_params'] = level_params
        
        # set end of time interval (only one time step is made)
        Tend=t0+dt
        
        # vary number of iterations
        for niter in niter_arr:
            # set number of iteration
            step_params['maxiter'] = niter
            description['step_params'] = step_params
            
            # print current parameters
            print('niter: %d\tnsteps: %f' % (niter, 1./nsteps))
        
            # instantiate the controller for SDC and MLSDC
            controller = controller_nonMPI(num_procs=1, controller_params=controller_params, description=description)
            
            # get initial values
            P = controller.MS[0].levels[0].prob
            uinit = P.u_exact(t0)
        
            # call main function to get things done...
            uend_sdc, stats_sdc = controller.run(u0=uinit, t0=t0, Tend=Tend)
            
            # save numerical solution as matrix
            L = controller.MS[0].levels[0]
            u_num_sdc = np.array([u.values for u in L.u])
            
            # compute ode solution (by calling the exact function of the problem at all quadrature nodes)
            nodes = [0]
            nodes.extend(L.sweep.coll.nodes)
            u_ode = np.array([P.u_exact(L.time + c*L.dt).values for c in nodes])
            
            # compute, save and print ode error and resulting order in dt
            err_ode_sdc = np.linalg.norm(u_ode - u_num_sdc, ord=np.inf)
            error[(niter, nsteps)] = err_ode_sdc
            order_ode_sdc = log(error[(niter, nsteps_arr[i-1])]/err_ode_sdc)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
            print('SDC:\tu_ode:\terror: %8.6e\torder:%4.2f' % (err_ode_sdc, order_ode_sdc))
            
        # compute solution of the collocation problem
        I_dtQxA = np.eye((m+1)*n) - L.dt*np.kron(L.sweep.coll.Qmat, P.A.toarray())
        u0 = np.kron(np.ones(m+1), uinit.values)
        u_coll = np.linalg.solve(I_dtQxA, u0)
        
        err_coll = np.linalg.norm(u_coll - u_ode.flatten(), ord=np.inf)
        error[(nsteps)] = err_coll
        order_coll = log(error[(nsteps_arr[i-1])]/err_coll)/log(nsteps/nsteps_arr[i-1]) if i > 0 else 0
        print('Coll:\tu_coll:\terror: %8.6e\torder:%4.2f' % (err_coll, order_coll))
            
    return error


def plot_errors(fname_errors, figname, order_sdc, order_coll):
    # Daten einlesen
    fin = open(fname_errors, "rb")
    error = pickle.load(fin)
    fin.close()
    
    # Farben und Symbole fuer Linien und Punkte einstellen
    color = ['red', 'magenta', 'blue', 'teal', 'green']
    marker = ['x','d','o','^','s']
        
    f, axes = plt.subplots(1, 2, sharex=False, sharey=True, figsize=(15,5))
    
    ymin1 = min([err for err in error[0].values() if isinstance(err, np.float64)])
    ymin2 = min([err for err in error[1].values() if isinstance(err, np.float64)])
    ymin = min(ymin1, ymin2)
    ymax1 = max([err for err in error[0].values() if isinstance(err, np.float64)])
    ymax2 = max([err for err in error[1].values() if isinstance(err, np.float64)])
    ymax = max(ymax1, ymax2)
    
    # Plot erstellen
    for i, err in enumerate(error):            
        dt_arr = 1./np.array(err['nsteps_arr'])
        
        for j, niter in enumerate(err['niter_arr']):

            # erreichte Punkte
            axes[i].plot(dt_arr, [err[(niter,n)] for n in err['nsteps_arr']], color=color[j], marker=marker[j], markersize=10, linestyle='None', label='k={}'.format(niter))
            
            # erwartete Linie: err
            axes[i].plot(dt_arr, [(err[(niter,err['nsteps_arr'][0])]/((nstep/err['nsteps_arr'][0])**(order_sdc(niter)))) for nstep in err['nsteps_arr']], color=color[j])
           
        # erreichte Punkte
        axes[i].plot(dt_arr, [err[(n)] for n in err['nsteps_arr']], color=color[-1], marker=marker[-1], markersize=10, linestyle='None', label='coll sol')
        
        # erwartete Linie: err
        axes[i].plot(dt_arr, [(err[(err['nsteps_arr'][0])]/((nstep/err['nsteps_arr'][0])**(order_coll(niter)))) for nstep in err['nsteps_arr']], color=color[-1])
            
        axes[i].set_xscale('log', basex=2)
        axes[i].set_yscale('log')
        axes[i].set_xlabel(r'$\Delta$t')
        axes[i].set_xlim(dt_arr[0]+dt_arr[0]/8, dt_arr[-1]-dt_arr[-1]/8)
#        start, end = axes[i].get_ylim()
#        axes[i].yaxis.set_ticks(np.linspace(start, end, 6))
#        axes[i].set_yticks(axes[i].get_yticks()[::2])
        axes[i].set_yticks(np.power(10.,np.arange(-14,3,2)))
        axes[i].set_ylim([ymin/10, ymax*10])
#        axes[i].set_ylim([1e-14, 1])
    
    axes[0].set_ylabel('error')
    axes[0].set_title(r'$\Delta$x small')
    axes[1].set_title(r'$\Delta$x big')
    
    ## LEGEND
    # reference: https://stackoverflow.com/questions/4700614/how-to-put-the-legend-out-of-the-plot
    # at the bottom
#    handles, labels = axes[0].get_legend_handles_labels()
#    f.legend(handles, labels, loc=8, numpoints=1, ncol=5)
#    plt.subplots_adjust(bottom=0.25)
    # at the right    
    axes[1].legend(numpoints=1, bbox_to_anchor=(1.04,0.5), loc="center left", borderaxespad=0, fontsize='small')
    
    plt.show()  
    
    if figname:
        f.savefig(figname, bbox_inches='tight', format="pdf")
#        plt.savefig('/home10/kremling/Documents/Studium/Gitte/Master/Seminararbeit/seminararbeit/daten/graphics/{}_{}_errors.png'.format(err['type'], dgl))
        print('figure saved in: {}'.format(figname)) 


def main():
    # set params
    nu = 0.1
    freq = 4
    m = 3
    random_init = True
    niter_arr = range(5,12,2)
    
    fname_errors = "data/errors_heat1d.pickle"
    figname = "figures/errors_heat1d.pdf"
    
    order_sdc = lambda k: k
    order_coll = lambda k: m+1
    
    n = 63
    nsteps_arr = [2**i for i in range(12,16)]
    error_dx_small = solve_heat1d(m, n, nu, freq, random_init, niter_arr, nsteps_arr)
    
    n = 15
    nsteps_arr = [2**i for i in range(8,12)]
    error_dx_big = solve_heat1d(m, n, nu, freq, random_init, niter_arr, nsteps_arr)
    
    fout = open(fname_errors, "wb")
    pickle.dump([error_dx_small, error_dx_big], fout)
    fout.close()
    print("results saved in: {}".format(fname_errors))
    
    plot_errors(fname_errors, figname, order_sdc, order_coll)

if __name__ == "__main__":
    main()
    
    