from numpy import sin, cos
import numpy as np

# These are only for plotting
import matplotlib.animation as animation
from matplotlib.patches import Circle
import matplotlib.pyplot as plt

from pydrake.all import MathematicalProgram, Jacobian, SolutionResult, Variables, sin, cos

import math

class ObstAvoid():

	def __init__(self, ang_discret=5, rng_discret=5):
		fov = np.deg2rad(70)
		self.theta = fov/2.
		self.max_range = 3. 
		self.min_range = 0.5
		self.ang_discret = ang_discret # number of discretizations for angs
		self.rng_discret = rng_discret # number of discretizations for range
		self.angles = np.linspace(-self.theta, self.theta, ang_discret) # discretize
		spacing = (self.max_range - self.min_range)/rng_discret
		self.ranges = np.linspace(self.min_range, self.max_range + spacing, rng_discret) # discretize

	def quad_dynamics(self, state, u, dt):
		# u input here are the velocities xd, yd
		new_state = np.zeros_like(state)
		new_state[2:] = u
		new_state[:2] = state[:2] + dt*u
		# might check the accuracy of this later
		return new_state

	def plot_trajectory(self, trajectory, obst_idx):
		'''
		Given a trajectory and an input_trajectory, plots this trajectory and control inputs over time.

		:param: trajectory: the output of simulate_states_over_time, or equivalent
			Note: see simulate_states_over_time for documentation of the shape of the output
		:param: input_trajectory: the input to simulate_states_over_time, or equivalent
			Note: see simulate_states_over_time for documentation of the shape of the input_trajectory
		:param: obst_idx: where the obstacles are 

		also plots environment 
		'''
		position_x = trajectory[:,0]
		position_y = trajectory[:,1]
		fig, axes = plt.subplots(nrows=1,ncols=1)
		axes.plot(position_x, position_y)

		for i in range(len(obst_idx) - 1):
			if int(obst_idx[i]) < (self.rng_discret - 1): # less than max range measured
				ang_min = self.angles[i] # lower angle bound of obstacle 
				ang_max = self.angles[i+1] # higher angle bound of obstaclee 
				rng_min = self.ranges[int(obst_idx[i])] # where the obst is at at this angle 
				rng_max = self.ranges[int(obst_idx[i] + 1)] 
				ang = (ang_min + ang_max)/2.
				rng = (rng_min + rng_max)/2.
				x = rng*np.cos(ang); y = rng*np.sin(ang); 
				r = (rng_min*np.sin(ang_max) - rng_min*np.sin(ang_min))/2.
				# print(x,y)
				circ = Circle((x,y), radius=r, facecolor="black", edgecolor='black', fill=True ,linewidth = 1.0, linestyle='solid')
				axes.add_patch(circ)

		axes.axis('equal')

		plt.show()

	def compute_trajectory(self, obst_idx, x_out, y_out, ux_out, uy_out, pose_initial=[0.,0.,0.,0.], dt=0.05):
		'''
		Find trajectory with MILP
		input u are tyhe velocities (xd, yd)
		dt 0.05 according to a rate of 20 Hz
		'''
		mp = MathematicalProgram()
		N = 30
		k = 0
		# define input trajectory and state traj
		u = mp.NewContinuousVariables(2, "u_%d" % k) # xd yd
		input_trajectory = u
		st = mp.NewContinuousVariables(4, "state_%d" % k)
		# # binary variables for obstalces constraint 
		c = mp.NewBinaryVariables(4*self.ang_discret, "c_%d" % k)
		obs = c
		states = st
		for k in range(1, N):
			u = mp.NewContinuousVariables(2, "u_%d" % k)
			input_trajectory = np.vstack((input_trajectory, u))
			st = mp.NewContinuousVariables(4, "state_%d" % k)
			states = np.vstack((states, st))
			c = mp.NewBinaryVariables(4*self.ang_discret, "c_%d" % k)
			obs = np.vstack((obs, c))
		st = mp.NewContinuousVariables(4, "state_%d" % (N + 1))
		states = np.vstack((states, st))
		c = mp.NewBinaryVariables(4*self.ang_discret, "c_%d" % k)
		obs = np.vstack((obs, c))
		### define cost
		mp.AddLinearCost(100*(- states[-1,0])) # go as far forward as possible
		# mp.AddQuadraticCost(states[-1,1]*states[-1,1])
		# time constraint 
		M = 1000 # slack var for obst costraint
		# state constraint
		for i in range(2): # initial state constraint x y yaw 
			mp.AddLinearConstraint(states[0,i] <= pose_initial[i])
			mp.AddLinearConstraint(states[0,i] >= pose_initial[i])
		for i in range(2): # initial state constraint xd yd yawd
			mp.AddLinearConstraint(states[0,i] <= pose_initial[2 + i] + 1)
			mp.AddLinearConstraint(states[0,i] >= pose_initial[2 + i] - 1)
		for i in range(N): 
			# state update according to dynamics
			state_next = self.quad_dynamics(states[i,:], input_trajectory[i,:], dt)
			for j in range(4):
				mp.AddLinearConstraint(states[i+1,j] <= state_next[j])
				mp.AddLinearConstraint(states[i+1,j] >= state_next[j])
			# obstacle constraint 
			for j in range(self.ang_discret - 1):
				mp.AddLinearConstraint(sum(obs[i,4*j:4*j+4]) <= 3)
				ang_min = self.angles[j] # lower angle bound of obstacle 
				ang_max = self.angles[j+1] # higher angle bound of obstaclee 
				if int(obst_idx[j]) < (self.rng_discret - 1): # less than max range measured
					rng_min = self.ranges[int(obst_idx[j])] # where the obst is at at this angle 
					rng_max = self.ranges[int(obst_idx[j] + 1)] 
					mp.AddLinearConstraint(states[i,0] <= rng_min - 0.05 + M*obs[i,4*j]) # xi <= xobs,low + M*c
					mp.AddLinearConstraint(states[i,0] >= rng_max + 0.005 - M*obs[i,4*j+1]) # xi >= xobs,high - M*c 
					mp.AddLinearConstraint(states[i,1] <= states[i,0]*np.tan(ang_min) - 0.05 + M*obs[i,4*j+2]) # yi <= xi*tan(ang,min) + M*c
					mp.AddLinearConstraint(states[i,1] >= states[i,0]*np.tan(ang_max) + 0.05 - M*obs[i,4*j+3]) # yi >= ci*tan(ang,max) - M*c
			# environmnt constraint, dont leave fov 
			mp.AddLinearConstraint(states[i,1] >= states[i,0]*np.tan(-self.theta))
			mp.AddLinearConstraint(states[i,1] <= states[i,0]*np.tan(self.theta))
			# bound the inputs 
			# mp.AddConstraint(input_trajectory[i,:].dot(input_trajectory[i,:]) <= 2.5*2.5) # dosnt work with multi int 
			mp.AddLinearConstraint(input_trajectory[i,0] <= 2.5)
			mp.AddLinearConstraint(input_trajectory[i,0] >= -2.5)
			mp.AddLinearConstraint(input_trajectory[i,1] <= 0.5)
			mp.AddLinearConstraint(input_trajectory[i,1] >= -0.5)


		mp.Solve()

		input_trajectory = mp.GetSolution(input_trajectory)
		trajectory = mp.GetSolution(states)
		x_out[:] = trajectory[:,0]
		y_out[:] = trajectory[:,1]
		ux_out[:] = input_trajectory[:,0]
		uy_out[:] = input_trajectory[:,1]
		return trajectory, input_trajectory

def plant(x, u):
	uxd = u[0,0].ToExpression()
	uyd = u[0,0].ToExpression()
	xd = uxd - x[-1]*uyd
	yd = x[-1]*uxd + uyd
	# xd = cos(x[-1])*uxd - sin(x[-1])*uyd
	# yd = sin(x[-1])*uxd + cos(x[-1])*uyd
	thetd = u[2,0].ToExpression()

	return np.array([xd,yd,thetd])

def SOS_traj_optim(S, rho_guess):
	# S provides the initial V guess 
	# STEP 1: search for L and u with fixed V and p
	mp1 = MathematicalProgram()
	x = mp1.NewIndeterminates(3, "x")
	V = x.dot(np.dot(S, x))
	print(S)
	# Define the Lagrange multipliers.
	(lambda_, constraint) = mp1.NewSosPolynomial(Variables(x), 4)
	xd = mp1.NewFreePolynomial(Variables(x), 2)
	yd = mp1.NewFreePolynomial(Variables(x), 2)
	thetd = mp1.NewFreePolynomial(Variables(x), 2)
	u = np.vstack((xd, yd))
	u = np.vstack((u, thetd))
	Vdot = Jacobian([V], x).dot(plant(x,u))[0]
	mp1.AddSosConstraint(-Vdot + lambda_.ToExpression() * (V - rho_guess))
	result = mp1.Solve()
	# print(type(lambda_).__dict__.keys())
	print(type(lambda_.decision_variables()).__dict__.keys())
	L = [mp1.GetSolution(var) for var in lambda_.decision_variables()]
	# print(lambda_.monomial_to_coefficient_map())
	return L, u