from pySDC.core.Errors import ParameterError
from pySDC.implementations.sweeper_classes.generic_implicit import generic_implicit
import numpy as np

class sweeper_random_initial_guess(generic_implicit):
    
    def predict(self):
        """
        Predictor to fill values at nodes before first sweep
    
        Default prediction for the sweepers, only copies the values to all collocation nodes
        and evaluates the RHS of the ODE there
        """
        
        # get current level and problem description
        L = self.level
        P = L.prob
    
        # evaluate RHS at left point
        L.f[0] = P.eval_f(L.u[0], L.time)
    
        for m in range(1, self.coll.num_nodes + 1):
            # copy u[0] to all collocation nodes, evaluate RHS
            if self.params.initial_guess == 'spread':
                L.u[m] = P.dtype_u(L.u[0])
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])
            # start with zero everywhere
            elif self.params.initial_guess == 'zero':
                L.u[m] = P.dtype_u(init=P.init, val=0.0)
                L.f[m] = P.dtype_f(init=P.init, val=0.0)
            elif self.params.initial_guess == 'random':
                np.random.seed(m*30)
                tmp = P.dtype_u(P.init)
                tmp.values = np.random.rand(P.init)
                L.u[m] = P.dtype_u(tmp)
                L.f[m] = P.eval_f(L.u[m], L.time + L.dt * self.coll.nodes[m - 1])
            else:
                raise ParameterError('initial_guess option {self.params.initial_guess} not implemented')
    
        # indicate that this level is now ready for sweeps
        L.status.unlocked = True
        L.status.updated = True