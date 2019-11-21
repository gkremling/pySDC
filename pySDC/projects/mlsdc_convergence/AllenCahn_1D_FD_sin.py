
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve

from pySDC.core.Errors import ParameterError, ProblemError
from pySDC.core.Problem import ptype
from pySDC.implementations.datatype_classes.mesh import mesh, rhs_imex_mesh, rhs_comp2_mesh

class allencahn_periodic_sin_fullyimplicit(ptype):
    """
    Example implementing the Allen-Cahn equation in 1D with finite differences and periodic BC,
    with driving force, 0-1 formulation (Bayreuth example)

    Attributes:
        A: second-order FD discretization of the 1D laplace operator
        dx: distance between two spatial nodes
    """

    def __init__(self, problem_params, dtype_u=mesh, dtype_f=mesh):
        """
        Initialization routine

        Args:
            problem_params (dict): custom parameters for the example
            dtype_u: mesh data type (will be passed parent class)
            dtype_f: mesh data type (will be passed parent class)
        """

        # these parameters will be used later, so assert their existence
        essential_keys = ['nvars', 'dw', 'eps', 'newton_maxiter', 'newton_tol', 'freq']
        for key in essential_keys:
            if key not in problem_params:
                msg = 'need %s to instantiate problem, only got %s' % (key, str(problem_params.keys()))
                raise ParameterError(msg)

        # we assert that nvars looks very particular here.. this will be necessary for coarsening in space later on
        if (problem_params['nvars']) % 2 != 0:
            raise ProblemError('setup requires nvars = 2^p')

        if 'stop_at_nan' not in problem_params:
            problem_params['stop_at_nan'] = True

        # invoke super init, passing number of dofs, dtype_u and dtype_f
        super(allencahn_periodic_sin_fullyimplicit, self).__init__(problem_params['nvars'], dtype_u, dtype_f,
                                                               problem_params)

        # compute dx and get discretization matrix A
        self.dx = 1. / self.params.nvars
        self.xvalues = np.array([i * self.dx for i in range(self.params.nvars)])

        self.A = self.__get_A(self.params.nvars, self.dx)

        self.newton_itercount = 0
        self.lin_itercount = 0
        self.newton_ncalls = 0
        self.lin_ncalls = 0

    @staticmethod
    def __get_A(N, dx):
        """
        Helper function to assemble FD matrix A in sparse format

        Args:
            N (int): number of dofs
            dx (float): distance between two spatial nodes

        Returns:
            scipy.sparse.csc_matrix: matrix A in CSC format
        """

        stencil = [1, -2, 1]
        zero_pos = 2
        dstencil = np.concatenate((stencil, np.delete(stencil, zero_pos - 1)))
        offsets = np.concatenate(([N - i - 1 for i in reversed(range(zero_pos - 1))],
                                  [i - zero_pos + 1 for i in range(zero_pos - 1, len(stencil))]))
        doffsets = np.concatenate((offsets, np.delete(offsets, zero_pos - 1) - N))

        A = sp.diags(dstencil, doffsets, shape=(N, N), format='csc')
        A *= 1.0 / (dx ** 2)

        return A

    def solve_system(self, rhs, factor, u0, t):
        """
        Simple Newton solver

        Args:
            rhs (dtype_f): right-hand side for the nonlinear system
            factor (float): abbrev. for the node-to-node stepsize (or any other factor required)
            u0 (dtype_u): initial guess for the iterative solver
            t (float): current time (required here for the BC)

        Returns:
            dtype_u: solution u
        """

        u = self.dtype_u(u0).values
        eps2 = 1.0 / self.params.eps ** 2 if self.params.eps > 0 else 0
        dw = self.params.dw

        Id = sp.eye(self.params.nvars)

        # start newton iteration
        n = 0
        res = 99
        while n < self.params.newton_maxiter:
            # print(n)
            # form the function g with g(u) = 0
            g = u - rhs.values - factor * (self.A.dot(u) - 2.0 * eps2 * u * (1.0 - u) * (1.0 - 2.0 * u) -
                                           6.0 * dw * u * (1.0 - u))

            # if g is close to 0, then we are done
            res = np.linalg.norm(g, np.inf)

            if res < self.params.newton_tol:
                break

            # assemble dg
            dg = Id - factor * (self.A - 2.0 * eps2 * sp.diags(
                (1.0 - u) * (1.0 - 2.0 * u) - u * ((1.0 - 2.0 * u) + 2.0 * (1.0 - u)), offsets=0) - 6.0 * dw * sp.diags(
                (1.0 - u) - u, offsets=0))

            # newton update: u1 = u0 - g/dg
            u -= spsolve(dg, g)
            # u -= gmres(dg, g, x0=z, tol=self.params.lin_tol)[0]
            # increase iteration count
            n += 1

        if np.isnan(res) and self.params.stop_at_nan:
            raise ProblemError('Newton got nan after %i iterations, aborting...' % n)
        elif np.isnan(res):
            self.logger.warning('Newton got nan after %i iterations...' % n)

        if n == self.params.newton_maxiter:
            self.logger.warning('Newton did not converge after %i iterations, error is %s' % (n, res))

        self.newton_ncalls += 1
        self.newton_itercount += n

        me = self.dtype_u(self.init)
        me.values = u

        return me

    def eval_f(self, u, t):
        """
        Routine to evaluate the RHS

        Args:
            u (dtype_u): current values
            t (float): current time

        Returns:
            dtype_f: the RHS
        """
        eps2 = 1.0 / self.params.eps ** 2 if self.params.eps > 0 else 0
        f = self.dtype_f(self.init)
        f.values = self.A.dot(u.values) - \
            2.0 * eps2 * u.values * (1.0 - u.values) * (1.0 - 2 * u.values) - \
            6.0 * self.params.dw * u.values * (1.0 - u.values)
        return f

    def u_exact(self, t):
        """
        Routine to compute the exact solution at time t

        Args:
            t (float): current time

        Returns:
            dtype_u: exact solution
        """

        me = self.dtype_u(self.init, val=0.0)
        me.values = np.sin(self.params.freq * np.pi * self.xvalues)
        return me