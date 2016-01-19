import numpy as np
import math
import scipy.sparse.linalg as LA
import scipy.sparse as sp
from ProblemClass import Callback, logging, boussinesq_2d_imex

#
# Runge-Kutta IMEX methods of order 1 to 3
#
class rk_imex:
  
  def __init__(self, problem, order):

    assert order in [1,2,3], "Order must be 1, 2 or 3"
    self.order = order

    if self.order==1:
      self.A      = np.array([[0,0],[0,1]])
      self.A_hat  = np.array([[0,0],[1,0]])
      self.b      = np.array([0,1])
      self.b_hat  = np.array([1,0])
      self.nstages = 2

    elif self.order==2:
      self.A      = np.array([[0,0],[0,0.5]])
      self.A_hat  = np.array([[0,0],[0.5,0]])
      self.b      = np.array([0,1])
      self.b_hat  = np.array([0,1])
      self.nstages = 2

    elif self.order==3:
      # parameter from Pareschi and Russo, J. Sci. Comp. 2005
      alpha = 0.24169426078821
      beta  = 0.06042356519705
      eta   = 0.12915286960590
      self.A_hat   = np.array([ [0,0,0,0], [0,0,0,0], [0,1.0,0,0], [0, 1.0/4.0, 1.0/4.0, 0] ])
      self.A       = np.array([ [alpha, 0, 0, 0], [-alpha, alpha, 0, 0], [0, 1.0-alpha, alpha, 0], [beta, eta, 0.5-beta-eta-alpha, alpha] ])
      self.b_hat   = np.array([0, 1.0/6.0, 1.0/6.0, 2.0/3.0])
      self.b       = self.b_hat
      self.nstages = 4

    self.problem = problem
    self.ndof = np.shape(problem.M)[0]
    self.logger = logging()
    self.stages = np.zeros((self.nstages, self.ndof))

  def timestep(self, u0, dt):

    # Solve for stages
    for i in range(0,self.nstages):

      # Construct RHS
      rhs = np.copy(u0)
      for j in range(0,i):
        rhs += dt*self.A_hat[i,j]*(self.f_slow(self.stages[j,:])) + dt*self.A[i,j]*(self.f_fast(self.stages[j,:]))

      # Solve for stage i
      if self.A[i,i] == 0:
        # Avoid call to spsolve with identity matrix
        self.stages[i,:] = np.copy(rhs)
      else:
        self.stages[i,:] = self.f_fast_solve( rhs, dt*self.A[i,i], u0 )
    
    # Update 
    for i in range(0,self.nstages):
      u0 += dt*self.b_hat[i]*(self.f_slow(self.stages[i,:])) + dt*self.b[i]*(self.f_fast(self.stages[i,:]))

    return u0

  def f_slow(self, u):
    return self.problem.D_upwind.dot(u)

  def f_fast(self, u):
    return self.problem.M.dot(u)

  def f_fast_solve(self, rhs, alpha, u0):
    cb = Callback()
    sol, info = LA.gmres( self.problem.Id - alpha*self.problem.M, rhs, x0=u0, tol=self.problem.gmres_tol, restart=self.problem.gmres_restart, maxiter=self.problem.gmres_maxiter, callback=cb)
    if alpha!=0.0:
      #print "RK-IMEX-%1i: Number of GMRES iterations: %3i --- Final residual: %6.3e" % ( self.order, cb.getcounter(), cb.getresidual() )
      self.logger.add(cb.getcounter())    
    return sol

#
# Trapezoidal rule
#
class trapezoidal:
 
  def __init__(self, problem, alpha=0.5):
    assert isinstance(problem, boussinesq_2d_imex), "problem is wrong type of object"
    self.Ndof = np.shape(problem.M)[0]
    self.order = 2
    self.logger = logging()
    self.problem = problem
    self.alpha = alpha

  def timestep(self, u0, dt):
    B_trap   = sp.eye(self.Ndof) + self.alpha*dt*(self.problem.D_upwind + self.problem.M)
    b         = B_trap.dot(u0)
    return self.f_solve(b, alpha=(1.0-self.alpha)*dt, u0 = u0)

  # 
  # Returns f(u) = c*u
  #  
  def f(self,u):
    return self.problem.D_upwind.dot(u)+self.problem.M.dot(u)
    
  
  #
  # Solves (Id - alpha*c)*u = b for u
  #  
  def f_solve(self, b, alpha, u0):
    cb = Callback()
    sol, info = LA.gmres( self.problem.Id - alpha*(self.problem.D_upwind + self.problem.M), b, x0=u0, tol=self.problem.gmres_tol, restart=self.problem.gmres_restart, maxiter=self.problem.gmres_maxiter, callback=cb)
    if alpha!=0.0:
      #print "BDF-2: Number of GMRES iterations: %3i --- Final residual: %6.3e" % ( cb.getcounter(), cb.getresidual() )
      self.logger.add(cb.getcounter())    
    return sol

#
# A BDF-2 implicit two-step method
#
class bdf2:

  def __init__(self, problem):
    assert isinstance(problem, boussinesq_2d_imex), "problem is wrong type of object"
    self.Ndof = np.shape(problem.M)[0]
    self.order = 2
    self.logger = logging()
    self.problem = problem

  def firsttimestep(self, u0, dt):
    return self.f_solve(b = u0, alpha = dt, u0 = u0)

  def timestep(self, u0, um1, dt):
    b = (4.0/3.0)*u0 - (1.0/3.0)*um1
    return self.f_solve(b = b, alpha = (2.0/3.0)*dt, u0 = u0)

  # 
  # Returns f(u) = c*u
  #  
  def f(self,u):
    return self.problem.D_upwind.dot(u)+self.problem.M.dot(u)
    
  
  #
  # Solves (Id - alpha*c)*u = b for u
  #  
  def f_solve(self, b, alpha, u0):
    cb = Callback()
    sol, info = LA.gmres( self.problem.Id - alpha*(self.problem.D_upwind + self.problem.M), b, x0=u0, tol=self.problem.gmres_tol, restart=self.problem.gmres_restart, maxiter=self.problem.gmres_maxiter, callback=cb)
    if alpha!=0.0:
      #print "BDF-2: Number of GMRES iterations: %3i --- Final residual: %6.3e" % ( cb.getcounter(), cb.getresidual() )
      self.logger.add(cb.getcounter())    
    return sol

#
# A diagonally implicit Runge-Kutta method of order 2, 3 or 4
#
class dirk:

  def __init__(self, problem, order):

    assert isinstance(problem, boussinesq_2d_imex), "problem is wrong type of object"
    self.Ndof = np.shape(problem.M)[0]
    self.order = order
    self.logger = logging()
    self.problem = problem

    assert self.order in [2,22,3,4], 'Order must be 2,22,3,4'
    
    if (self.order==2):
      self.nstages = 1
      self.A       = np.zeros((1,1))
      self.A[0,0]  = 0.5
      self.tau     = [0.5]
      self.b       = [1.0]
    
    if (self.order==22):
      self.nstages = 2
      self.A       = np.zeros((2,2))
      self.A[0,0]  = 1.0/3.0
      self.A[1,0]  = 1.0/2.0
      self.A[1,1]  = 1.0/2.0
      
      self.tau     = np.zeros(2)
      self.tau[0]  = 1.0/3.0
      self.tau[1]  = 1.0

      self.b       = np.zeros(2)
      self.b[0]    = 3.0/4.0
      self.b[1]    = 1.0/4.0
    
    
    if (self.order==3):
      self.nstages = 2 
      self.A       = np.zeros((2,2))
      self.A[0,0]  = 0.5 + 1.0/( 2.0*math.sqrt(3.0) )
      self.A[1,0] = -1.0/math.sqrt(3.0)
      self.A[1,1] = self.A[0,0]
      
      self.tau    = np.zeros(2)
      self.tau[0] = 0.5 + 1.0/( 2.0*math.sqrt(3.0) )
      self.tau[1] = 0.5 - 1.0/( 2.0*math.sqrt(3.0) )
      
      self.b     = np.zeros(2)
      self.b[0]  = 0.5
      self.b[1]  = 0.5
      
    if (self.order==4):
      self.nstages = 3
      alpha = 2.0*math.cos(math.pi/18.0)/math.sqrt(3.0)
      
      self.A      = np.zeros((3,3))
      self.A[0,0] = (1.0 + alpha)/2.0
      self.A[1,0] = -alpha/2.0
      self.A[1,1] = self.A[0,0]
      self.A[2,0] = (1.0 + alpha)
      self.A[2,1] =  -(1.0 + 2.0*alpha)
      self.A[2,2] = self.A[0,0]
      
      self.tau    = np.zeros(3)
      self.tau[0] = (1.0 + alpha)/2.0
      self.tau[1] = 1.0/2.0
      self.tau[2] = (1.0 - alpha)/2.0
      
      self.b      = np.zeros(3)
      self.b[0]   = 1.0/(6.0*alpha*alpha)
      self.b[1]   = 1.0 - 1.0/(3.0*alpha*alpha)
      self.b[2]   = 1.0/(6.0*alpha*alpha)
       
    self.stages  = np.zeros((self.nstages,self.Ndof))

  def timestep(self, u0, dt):
    
      uend           = u0
      for i in range(0,self.nstages):  
        
        b = u0
        
        # Compute right hand side for this stage's implicit step
        for j in range(0,i):
          b = b + self.A[i,j]*dt*self.f(self.stages[j,:])
        
        # Implicit solve for current stage    
        #if i==0:
        self.stages[i,:] = self.f_solve( b, dt*self.A[i,i] , u0 )
        #else:
        #  self.stages[i,:] = self.f_solve( b, dt*self.A[i,i] , self.stages[i-1,:] )
        
        # Add contribution of current stage to final value
        uend = uend + self.b[i]*dt*self.f(self.stages[i,:])
        
      return uend
      
  # 
  # Returns f(u) = c*u
  #  
  def f(self,u):
    return self.problem.D_upwind.dot(u)+self.problem.M.dot(u)
    
  
  #
  # Solves (Id - alpha*c)*u = b for u
  #  
  def f_solve(self, b, alpha, u0):
    cb = Callback()
    sol, info = LA.gmres( self.problem.Id - alpha*(self.problem.D_upwind + self.problem.M), b, x0=u0, tol=self.problem.gmres_tol, restart=self.problem.gmres_restart, maxiter=self.problem.gmres_maxiter, callback=cb)
    if alpha!=0.0:
      #print "DIRK-%1i: Number of GMRES iterations: %3i --- Final residual: %6.3e" % ( self.order, cb.getcounter(), cb.getresidual() )
      self.logger.add(cb.getcounter())    
    return sol
