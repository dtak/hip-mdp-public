"""Simple 2D point navigation domain in a grid

Developed to prove out algorithmic concepts the HiP-MDP with embedded latent weights

The concept of this module is that you have a point in a confined 2D grid that can move N,E,S, or W, 
depending on the prescribed action. The start point starts in (-1.25,-1.75)**2. The goal is to get the 
point into the upper right quadrant. Each instance has a single hidden parameter in {1,2} that modifies
the dynamical system for that instance in 3 ways:
	1. The location of a wall blocking entry into the goal state (either on the left or bottom of the goal state)
	2. Whether actions (1,2,3,4) correspond to (W,N,E,S) or (E,S,W,N)
	3. The direction of a drift force (either pushes the agent S or W)
The drift force increases as the agent moves further away from the center of the start region.
"""
import numpy as np

class Grid(object):
	"""
	The environment for the 2D navigation domain.
	"""
	state_names = ('x1', 'x2')
	def __init__(self, perturb_params=False, latent_code=1, start_state=[-1.5,-1.5], beta=0.23, **kw):
		"""
		Initialize the environment

		Keyword arguments:
		perturb_params -- boolean indicating if the start state should be perturbed (default: False)
		latent_code -- hidden parameter (default: 1)
		start_state -- specified start state (default: [-1.5,-1.5])
		beta -- parameter governing the magnitude of the drift force (default: 0.23)
		"""
		self.num_actions = 4
		self.perturb_params = (['latent_code'])
		self.x1_range = [-2, 2] # Construct the box
		self.x2_range = [-2, 2]
		self.target = [0.5,0.5] # Want to encourage the agent to go toward this point in the goal region
		
		# self.beta = 0.23
		self.beta = beta # hyperparameter scales the amount the "drift" in the game
		# designate the wall location
		self.goal_region = [0, 0] # Here we check to see if point is greater than both components of the goal_region

		# Initialize the state
		self.reset(perturb_params, latent_code, start_state, **kw)

	def reset(self, perturb_params = False, latent_code = 1, start_state = [-1.5,-1.5], **kw):
		"""
		Reset the environment.

		Keyword arguments:
		perturb_params -- boolean indicating if the start state should be perturbed (default: False)
		latent_code -- hidden parameter (default: 1)
		start_state -- specified start state (default: [-1.5,-1.5])
		"""
		self.t = 0
		if perturb_params:
			start_state = np.random.uniform(low=-1.75, high=-1.25,size=2)
		self.start_state = start_state
		self.latent_code = latent_code
		self.state = start_state
		# Adjust dynamics based on hidden parameter
		if self.latent_code == 1:
			self.wall_on_left = False
			self.step_size = 0.3
		else: 		
			self.wall_on_left = True
			self.step_size = -0.3
				
	def get_next_state(self, state, action):
		"""
		Return the next state, given the current state and action.
		"""
		next_state = np.copy(state)
		action_effect = self.get_action_effect(action)
		if self.latent_code == 1:
			next_state[0] += self.step_size * action_effect[0] - self.beta*((state[0]+1.5)**2+((state[1]+1.5)**2))**0.5*self.step_size
			next_state[1] += self.step_size * action_effect[1]
		else:
			next_state[0] += self.step_size * action_effect[0] 
			next_state[1] += self.step_size * action_effect[1] + self.beta*((state[0]+1.5)**2+((state[1]+1.5)**2))**0.5*self.step_size
		return next_state

	def observe(self):
		"""Return current state."""
		return self.state

	def is_done(self, episode_length=100, state=None):
		"""Check if episode is over."""
		if state is None:
			s = self.observe()
		else:
			s = state
		if (self.t >= episode_length) or self._in_goal(s):
			return True
		else:
			return False
	
	def get_action_effect(self, action):
		"""Return the state difference caused by the action without accounting for the drift force."""
		action_index = np.array([-1, 1, 1, -1]).reshape(2,2)
		a = np.array([0] * self.num_actions)
		a[action] = 1
		a = a.reshape(2,2)
		act = np.sum(a * action_index, axis=0)
		return act


	def _valid_crossing(self, state=None, next_state=None, action=None):
		"""Check whether taking action from the state is a valid move. 
		There are 2 illegal actions: Leaving the grid area and moving through the wall.
		The keyword arguments are included to allow function to handle arbitrary transitions.
		"""
		if state is None:
			state = self.state
			action = self.action
		if next_state is None:
			# Compute next state
			next_state = self.get_next_state(state,action)
		# Check if agent exited the grid area
		if np.max(np.abs(next_state[0])) > np.max(np.abs(self.x1_range)) or np.max(np.abs(next_state[1])) > np.max(np.abs(self.x2_range)):
			return False
		# Check if agent exited the goal state by moving through a wall
		if self._in_goal(state):
			# Wall on bottom and and agent moved below wall
			if (not self.wall_on_left) and (next_state[1] <= self.goal_region[1]):
				return False
			# Wall on left and and agent moved left of the wall	
			elif self.wall_on_left and (next_state[0 ]<= self.goal_region[0]):
				return False
		# Check if agent entered the goal state by moving through a wall
		elif self._in_goal(next_state):
			# Wall on bottom and and agent was below of the goal state
			if (not self.wall_on_left) and (state[1] <= self.goal_region[1]):
				return False
			# Wall on left and and agent was left of the goal state	
			elif self.wall_on_left and (state[0] <= self.goal_region[0]):
				return False
		# Otherwise, the action was valid
		return True

	def _in_goal(self, state = None):
		"""Check if agent is in the goal region."""
		if state is None:
			state = self.state
		return True if ((state[0] > self.goal_region[0]) and (state[1] > self.goal_region[1])) else False
	

	def calc_reward(self, state=None, action=None, next_state=None, latent_code=0, **kw):
		"""Calculate the reward for the specified transition. The keyword arguments are included
		to allow function to handle arbitrary transitions.
		"""
		if state is None:
			state = self.state
			action = self.action
			latent_code = self.latent_code
		next_state = self.get_next_state(state,action)
		if self._valid_crossing(state=state, next_state=next_state, action=action) and self._in_goal(state=next_state):
			return 1000
		elif self._valid_crossing(state=state, next_state=next_state, action=action) and not self._in_goal(state=next_state):
			return -.1
		else:
			return -5

	def perform_action(self, action, **kw):
		"""Take the specified action and update the environment accordingly."""
		self.t += 1
		self.action = action
		reward = self.calc_reward()
		# Constrain the point to stay in the bounding box or not cross an invalid boundary
		if (not self._valid_crossing()):
			reward = -5
		else:
			self.state = self.get_next_state(self.state,action)
		return reward, self.observe()




