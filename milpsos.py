from numpy import sin, cos
import numpy as np

# These are only for plotting
import matplotlib.animation as animation
from matplotlib.patches import Circle, Ellipse
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

from pydrake.all import MathematicalProgram, Jacobian, SolutionResult, Variables, LinearQuadraticRegulator

import math

### MILPSOS operation ### 

class MILPSOS():

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

	def plot_trajectory(self, trajectory, obst_idx, x_margin=None, y_margin=None):
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
				pts = np.zeros([4,2])
				ang_min = self.angles[i] # lower angle bound of obstacle 
				ang_max = self.angles[i+1] # higher angle bound of obstaclee 
				rng_min = self.ranges[int(obst_idx[i])] # where the obst is at at this angle 
				rng_max = self.ranges[int(obst_idx[i] + 1)] 
				ang = (ang_min + ang_max)/2.
				rng = (rng_min + rng_max)/2.
				pts[0,:] = np.array([rng_min*cos(ang_min), rng_min*np.sin(ang_min)])
				pts[1,:] = np.array([rng_max*cos(ang_min), rng_max*np.sin(ang_min)])
				pts[2,:] = np.array([rng_max*cos(ang_max), rng_max*np.sin(ang_max)])
				pts[3,:] = np.array([rng_min*cos(ang_max), rng_min*np.sin(ang_max)])
				poly = Polygon(pts, facecolor="black", edgecolor='black', fill=True ,linewidth = 1.0, linestyle='solid')
				axes.add_patch(poly)

		if x_margin != None and y_margin != None:
			for i in range(1,trajectory.shape[0]):
				rad = np.arctan((trajectory[i,1] - trajectory[i-1,1])/(trajectory[i,0] - trajectory[i-1,0]))
				theta = np.rad2deg(rad)
				x = trajectory[i,0]; y = trajectory[i,1]
				reg = Ellipse((x,y),width=x_margin,height=y_margin, angle=theta, alpha=0.5)
				axes.add_patch(reg)

		axes.axis('equal')

		plt.show()

	def plot_grid(self):
		scan_x1 = np.ones([20])*(0.8*np.cos(-self.theta)) + 0.1*np.random.random([20,])
		scan_y1 = np.linspace(0.8*np.sin(-self.theta),0,20)
		scan_x2 = np.ones([30])*(2.5*np.cos(self.theta)) + 0.1*np.random.random([30,])
		scan_y2 = np.linspace(0, 2.5*np.sin(self.theta),30) 
		fig, axes = plt.subplots(nrows=1,ncols=1)
		axes.plot(scan_x1, scan_y1, color='red', markersize=24)
		axes.plot(scan_x2, scan_y2, color='red', markersize=24)
		for i in range(self.ang_discret - 1):
			for j in range(self.rng_discret - 1): # less than max range measured
				pts = np.zeros([4,2])
				ang_min = self.angles[i] # lower angle bound of obstacle 
				ang_max = self.angles[i+1] # higher angle bound of obstaclee 
				rng_min = self.ranges[j] # where the obst is at at this angle 
				rng_max = self.ranges[j + 1] 
				ang = (ang_min + ang_max)/2.
				rng = (rng_min + rng_max)/2.
				pts[0,:] = np.array([rng_min*cos(ang_min), rng_min*np.sin(ang_min)])
				pts[1,:] = np.array([rng_max*cos(ang_min), rng_max*np.sin(ang_min)])
				pts[2,:] = np.array([rng_max*cos(ang_max), rng_max*np.sin(ang_max)])
				pts[3,:] = np.array([rng_min*cos(ang_max), rng_min*np.sin(ang_max)])
				poly = Polygon(pts, facecolor="black", edgecolor='black', fill=False ,linewidth = 1.0, linestyle='solid')
				axes.add_patch(poly)

		obst_idx = [0, 0, 2, 2, 2]
		for i in range(len(obst_idx) - 1):
			if int(obst_idx[i]) < (self.rng_discret - 1): # less than max range measured
				pts = np.zeros([4,2])
				ang_min = self.angles[i] # lower angle bound of obstacle 
				ang_max = self.angles[i+1] # higher angle bound of obstaclee 
				rng_min = self.ranges[int(obst_idx[i])] # where the obst is at at this angle 
				rng_max = self.ranges[int(obst_idx[i] + 1)] 
				ang = (ang_min + ang_max)/2.
				rng = (rng_min + rng_max)/2.
				pts[0,:] = np.array([rng_min*cos(ang_min), rng_min*np.sin(ang_min)])
				pts[1,:] = np.array([rng_max*cos(ang_min), rng_max*np.sin(ang_min)])
				pts[2,:] = np.array([rng_max*cos(ang_max), rng_max*np.sin(ang_max)])
				pts[3,:] = np.array([rng_min*cos(ang_max), rng_min*np.sin(ang_max)])
				poly = Polygon(pts, facecolor="black", edgecolor='black', fill=True ,linewidth = 1.0, linestyle='solid')
				axes.add_patch(poly)

		axes.axis('equal')

		plt.show()

	def MILP_compute_traj(self, obst_idx, x_out, y_out, dx, dy, pose_initial=[0.,0.]):
		'''
		Find trajectory with MILP
		Outputs trajectory (waypoints) and new K for control 
		'''
		mp = MathematicalProgram()
		N = 8
		k = 0
		# define state traj
		st = mp.NewContinuousVariables(2, "state_%d" % k)
		# # binary variables for obstalces constraint 
		c = mp.NewBinaryVariables(4*self.ang_discret, "c_%d" % k)
		obs = c
		states = st
		for k in range(1, N):
			st = mp.NewContinuousVariables(2, "state_%d" % k)
			states = np.vstack((states, st))
			c = mp.NewBinaryVariables(4*self.ang_discret, "c_%d" % k)
			obs = np.vstack((obs, c))
		st = mp.NewContinuousVariables(2, "state_%d" % (N + 1))
		states = np.vstack((states, st))
		c = mp.NewBinaryVariables(4*self.ang_discret, "c_%d" % k)
		obs = np.vstack((obs, c))
		# variables encoding max x y dist from obstacle 
		x_margin = mp.NewContinuousVariables(1, "x_margin")
		y_margin = mp.NewContinuousVariables(1, "y_margin")
		### define cost
		for i in range(N):
			mp.AddLinearCost(-states[i,0]) # go as far forward as possible
		mp.AddLinearCost(-states[-1,0])
		mp.AddLinearCost(-x_margin[0])
		mp.AddLinearCost(-y_margin[0])
		# bound x y margin so it doesn't explode 
		mp.AddLinearConstraint(x_margin[0] <= 3.)
		mp.AddLinearConstraint(y_margin[0] <= 3.)
		# x y is non ngative adn at least above robot radius
		mp.AddLinearConstraint(x_margin[0] >= 0.05)
		mp.AddLinearConstraint(y_margin[0] >= 0.05)
		M = 1000 # slack var for integer things
		# state constraint
		for i in range(2): # initial state constraint x y
			mp.AddLinearConstraint(states[0,i] <= pose_initial[i])
			mp.AddLinearConstraint(states[0,i] >= pose_initial[i])
		for i in range(N): 
			mp.AddQuadraticCost((states[i+1,1] - states[i,1])**2)
			mp.AddLinearConstraint(states[i+1,0] <= states[i,0] + dx)
			mp.AddLinearConstraint(states[i+1,0] >= states[i,0] - dx)
			mp.AddLinearConstraint(states[i+1,1] <= states[i,1] + dy)
			mp.AddLinearConstraint(states[i+1,1] >= states[i,1] - dy)
			# obstacle constraint 
			for j in range(self.ang_discret - 1):
				mp.AddLinearConstraint(sum(obs[i,4*j:4*j+4]) <= 3)
				ang_min = self.angles[j] # lower angle bound of obstacle 
				ang_max = self.angles[j+1] # higher angle bound of obstaclee 
				if int(obst_idx[j]) < (self.rng_discret - 1): # less than max range measured
					rng_min = self.ranges[int(obst_idx[j])] # where the obst is at at this angle 
					rng_max = self.ranges[int(obst_idx[j] + 1)] 
					mp.AddLinearConstraint(states[i,0] <= rng_min - x_margin[0] + M*obs[i,4*j]) # xi <= xobs,low + M*c
					mp.AddLinearConstraint(states[i,0] >= rng_max + x_margin[0] - M*obs[i,4*j+1]) # xi >= xobs,high - M*c 
					mp.AddLinearConstraint(states[i,1] <= states[i,0]*np.tan(ang_min) - y_margin[0] + M*obs[i,4*j+2]) # yi <= xi*tan(ang,min) + M*c
					mp.AddLinearConstraint(states[i,1] >= states[i,0]*np.tan(ang_max) + y_margin[0] - M*obs[i,4*j+3]) # yi >= ci*tan(ang,max) - M*c
		# obstacle constraint for last state
		for j in range(self.ang_discret - 1):
			mp.AddLinearConstraint(sum(obs[N,4*j:4*j+4]) <= 3)
			ang_min = self.angles[j] # lower angle bound of obstacle 
			ang_max = self.angles[j+1] # higher angle bound of obstaclee 
			if int(obst_idx[j]) < (self.rng_discret - 1): # less than max range measured
				rng_min = self.ranges[int(obst_idx[j])] # where the obst is at at this angle 
				rng_max = self.ranges[int(obst_idx[j] + 1)] 
				mp.AddLinearConstraint(states[N,0] <= rng_min - x_margin[0] + M*obs[N,4*j]) # xi <= xobs,low + M*c
				mp.AddLinearConstraint(states[N,0] >= rng_max + x_margin[0] - M*obs[N,4*j+1]) # xi >= xobs,high - M*c 
				mp.AddLinearConstraint(states[N,1] <= states[N,0]*np.tan(ang_min) - y_margin[0] + M*obs[N,4*j+2]) # yi <= xi*tan(ang,min) + M*c
				mp.AddLinearConstraint(states[N,1] >= states[N,0]*np.tan(ang_max) + y_margin[0] - M*obs[N,4*j+3]) # yi >= ci*tan(ang,max) - M*c


		mp.Solve()

		trajectory = mp.GetSolution(states)
		xm = mp.GetSolution(x_margin)
		ym = mp.GetSolution(y_margin)
		x_out[:] = trajectory[:,0]
		y_out[:] = trajectory[:,1]
		return trajectory, xm[0], ym[0]

	def get_dxdy(self, rho, S, sample=100):
		# from an optimized rho, and the current V (polynomial?), find dx dy 
		# randomly sample dx dy
		upper_y = 0.3
		upper_x = 0.3
		lower_y = 0.
		lower_x = 0.
		lower_vval = 0
		for i in range(sample):
			dx = (upper_x - lower_x)*np.random.random() + lower_x
			dy = (upper_y - lower_y)*np.random.random() + lower_y
			V_val = self.calcV(S, dx, dy)
			if V_val < rho and V_val > lower_vval:
				lower_x = dx
				lower_y = dy
				lower_vval = V_val
		return lower_x, lower_y, lower_vval

	def get_new_S(self, rho, xm, ym):
		# get new S from the xy margins
		S = np.eye(2)
		S[0,0] = rho/(xm*xm)
		S[1,1] = rho/(ym*ym)
		return S

	def calcV(self, S, dx, dy):
		x_ = np.array([dx, dy])
		return x_.dot(np.dot(S, x_))

	def calcS(self, K):
		# in my quadrotor case, S = K
		return K 

	def calcU(self, u_coeff, x):
		a = u_coeff
		return a[5] + a[1]*x[0]*x[1] + a[4]*x[0] + a[2]*x[0]*x[0] + a[3]*x[1] + a[0]*x[1]*x[1]

	def dynamics(self, x, u):
		xd = u[0,0].ToExpression()
		yd = u[1,0].ToExpression()
		return np.array([xd, yd])

	def dynamics_ucoeff(self, x, ux_coeff, uy_coeff):
		xd = self.calcU(ux_coeff, x)
		yd = self.calcU(uy_coeff, x)
		return np.array([xd, yd])

	def dynamics_K(self, x, K):
		xd = -K[0]*x[0]
		yd = -K[1]*x[1]
		return np.array([xd, yd])

	def dynamics_K_sat1(self, x, K):
		# upper saturated xd 
		xd = 2.5
		yd = -K[1]*x[1]
		return np.array([xd, yd])

	def dynamics_K_sat2(self, x, K):
		# lower saturated xd 
		xd = -2.5
		yd = -K[1]*x[1]
		return np.array([xd, yd])

	def dynamics_K_sat3(self, x, K):
		# upper saturated yd 
		xd = -K[0]*x[0]
		yd = 2.5 
		return np.array([xd, yd])

	def dynamics_K_sat4(self, x, K):
		# lower saturated yd 
		xd = -K[0]*x[0]
		yd = -2.5
		return np.array([xd, yd])

	def SOS_compute_1(self, S, rho_prev):
		# fix V and rho, search for L and u
		prog = MathematicalProgram()
		x = prog.NewIndeterminates(2, "x")

		# Define u 
		K = prog.NewContinuousVariables(2, "K")

		# Fixed Lyapunov
		V = x.dot(np.dot(S, x))
		Vdot = Jacobian([V], x).dot(self.dynamics_K(x, K))[0]

		# Define the Lagrange multipliers.
		(lambda_, constraint) = prog.NewSosPolynomial(Variables(x), 2)
		prog.AddLinearConstraint(K[0]*x[0] <= 2.5)
		prog.AddSosConstraint(-Vdot - lambda_.ToExpression() * (rho_prev - V))

		result = prog.Solve()
		# print(lambda_.ToExpression())
		# print(lambda_.decision_variables())
		lc = [prog.GetSolution(var) for var in lambda_.decision_variables()]
		lbda_coeff = np.ones([3,3])
		lbda_coeff[0,0] = lc[0]; lbda_coeff[0,1] = lbda_coeff[1,0] = lc[1]; lbda_coeff[2,0] = lbda_coeff[0,2] = lc[2]
		lbda_coeff[1,1] = lc[3]; lbda_coeff[2,1] = lbda_coeff[1,2] = lc[4]; lbda_coeff[2,2] = lc[5]
		return lbda_coeff

	def SOS_compute_2(self, l_coeff, S, rho_max=10.):
		prog = MathematicalProgram()
		# fix V and lbda, searcu for u and rho
		x = prog.NewIndeterminates(2, "x")
		# get lbda from before 
		l = np.array([x[1], x[0], 1])
		lbda = l.dot(np.dot(l_coeff, l))

		# Define u 
		K = prog.NewContinuousVariables(2, "K")

		# Fixed Lyapunov
		V = x.dot(np.dot(S, x))
		Vdot = Jacobian([V], x).dot(self.dynamics_K(x, K))[0]

		# rho is decision variable now 
		rho = prog.NewContinuousVariables(1, "rho")[0]

		prog.AddSosConstraint(-Vdot - lbda * (rho - V))

		prog.AddLinearConstraint(rho <= rho_max)
		prog.AddLinearCost(-rho)
		prog.Solve()
		rho = prog.GetSolution(rho)
		K = prog.GetSolution(K)
		return rho, K

	def SOS_compute_3(self, K, l_coeff, rho_max=10.):
		prog = MathematicalProgram()
		# fix u and lbda, search for V and rho 
		x = prog.NewIndeterminates(2, "x")

		# get lbda from before 
		l = np.array([x[1], x[0], 1])
		lbda = l.dot(np.dot(l_coeff, l))

		# rho is decision variable now 
		rho = prog.NewContinuousVariables(1, "rho")[0]

		# create lyap V 
		s = prog.NewContinuousVariables(4, "s")
		S = np.array([[s[0], s[1]], [s[2], s[3]]])
		V = x.dot(np.dot(S, x))
		Vdot = Jacobian([V], x).dot(self.dynamics_K(x, K))[0]

		prog.AddSosConstraint(V)
		prog.AddSosConstraint(-Vdot - lbda * (rho - V))

		prog.AddLinearCost(-rho)
		prog.AddLinearConstraint(rho <= rho_max)

		prog.Solve()
		rho = prog.GetSolution(rho)
		s = prog.GetSolution(s)
		return s, rho

	def SOS_compute_4(self, l_coeff, S, rho):
		prog = MathematicalProgram()
		# fix V and lbda, searcu for u and rho
		x = prog.NewIndeterminates(2, "x")
		# get lbda from before 
		l = np.array([x[1], x[0], 1])
		lbda = l.dot(np.dot(l_coeff, l))

		# Define u 
		K = prog.NewContinuousVariables(2, "K")

		# Fixed Lyapunov
		V = x.dot(np.dot(S, x))
		Vdot = Jacobian([V], x).dot(self.dynamics_K(x, K))[0]

		prog.AddSosConstraint(-Vdot - lbda * (rho - V))

		prog.Solve()
		K = prog.GetSolution(K)
		return rho, K

	def create_LQR(self): # as initialization 
		Q = np.eye(2)
		R = np.eye(2)
		A = np.zeros([2,2])
		B = np.eye(2)

		K, S = LinearQuadraticRegulator(A, B, Q, R)

		return K, S