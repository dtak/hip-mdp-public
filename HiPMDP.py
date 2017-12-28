"""
General Framework for Hidden Parameter Markov Decision Processes (HiP-MDPs) and benchmarks.
"""
from __future__ import print_function
import argparse
import tensorflow as tf
import numpy as np
import pickle
from Qnetwork import Qnetwork
from ExperienceReplay import ExperienceReplay
from BayesianNeuralNetwork import *
class HiPMDP(object):
	"""
	The HiP-MDP class can be used to:
	- Create a new batch of experience using agent learning a policy modelfree (run_type='modelfree', create_exp_batch=True)
	- Test one of the following methods on a single test instance:
		- Full HiP-MDP with embedded latent weights (run_type='full' and load pretrained bnn_network_weights)
		- Full HiP-MDP with linear top latent weights (run_type='full_linear' and load pretrained bnn_network_weights)
		- Average model (run_type='onesize' and load pretrained bnn_network_weights)
		- Model from scratch (run_type='modelbased')
		- Model-free (run_type='modelfree')
	"""

	def __init__(self, domain, preset_hidden_params, run_type='full', ddqn_learning_rate=0.0001, 
		episode_count=500, bnn_hidden_layer_size=25, bnn_num_hidden_layers=2, bnn_network_weights=None, 
		eps_min=0.15, test_inst=None, create_exp_batch=False, num_batch_instances=False, save_results=False, 
		grid_beta=0.23, print_output=False):
		"""
		Initialize framework.

		Arguments:
		domain -- the domain the framework will be used on ('grid','hiv', or 'acrobot')
		preset_hidden_params -- List of dictionaries; one dictionary for each instance, where each dictionary contains
			the hidden parameter settings for that instance

		Keyword arguments:
		run_type -- 'full': Constructs a HiP-MDP model through which transfer is facilitated and with which accelerates policy learning, 
					'full_linear': Constructs a HiP-MDP model, but associates the latent weights w_b as a linear weighting of 
						the model features rather than using them as input as is done in the full HiP-MDP model.
					'modelfree': Learns a policy based solely on observed transitions,
					'modelbased': builds a model for accelerating policy learning from only the current instance's data
		ddqn_learning_rate -- DQN ADAM learning rate (default=0.0001)
		episode_count -- Number of episodes per instance (default=500)
		bnn_hidden_layer_size -- Number of units in each hidden layer (default=25)
		bnn_num_hidden_layers -- Number hidden layers (default=2)
		bnn_network -- 1-D numpy array of pretrained BNN network weights (default=None)
		eps_min -- Minimum epsilon value for e-greedy policy (default=0.15)
		test_inst -- Index corresponding to the desired test instance; irrelevant when creating an experience batch (default=None)
		create_exp_batch -- Boolean indicating if this framework is for creating an experience batch (default=False) 
		num_batch_instances -- number of instances desired in constructing a batch of data to train, default is false to cleanly override if not specified
		grid_beta -- Beta hyperparameter for grid domain governing; a weight on the magnitude of the "drift" (default=0.23)
		print_output -- Print verbose output
		"""

		self.__initialize_params()
		
		# Store arguments
		self.domain = domain
		self.ddqn_learning_rate = ddqn_learning_rate
		self.run_type = run_type
		if self.run_type in ['full','full_linear']:
			self.run_type_full = True
		else:
			self.run_type_full = False
		self.preset_hidden_params = preset_hidden_params
		self.bnn_hidden_layer_size = bnn_hidden_layer_size
		self.bnn_num_hidden_layers = bnn_num_hidden_layers
		self.bnn_network_weights = bnn_network_weights
		self.eps_min = eps_min
		self.test_inst = test_inst
		self.create_exp_batch = create_exp_batch
		self.num_batch_instances = num_batch_instances
		self.save_results = save_results
		self.grid_beta = grid_beta
		self.print_output = print_output
		# Set domain specific hyperparameters
		self.__set_domain_hyperparams()
		self.episode_count = episode_count
		# set epsilon step size
		if self.run_type == 'modelfree':
			self.eps_step = (self.eps_max-self.eps_min) / self.episode_count
		else:
			self.eps_step = (self.eps_max-self.eps_min) / self.num_approx_episodes

	def __initialize_params(self):
		"""Initialize standard framework settings."""
		self.instance_count = 1 # number of task instances
		self.episode_count = 500 # number of episodes
		self.weight_count = 5 # number of latent weights
		self.eps_max = 1.0 # initial epsilon value for e-greedy policy
		self.bnn_and_latent_update_interval = 10 # Number of episodes between BNN and latent weight updates
		self.num_strata_samples = 5 # The number of samples we take from each strata of the experience buffer
		self.ddqn_batch_size = 50 # The number of data points pulled from the experience buffer for replay
		self.tau = 0.005 # The transfer rate between our primary DQN and the target DQN
		self.discount_rate = 0.99  # RL discount rate of expected future rewards
		self.beta_zero = 0.5     # Initial bias correction parameter for Importance Sampling when doing prioritized experience replay
		self.bnn_num_samples = 50 # number of samples of network weights drawn to get each BNN prediction
		self.bnn_batch_size = 32
		self.bnn_v_prior = 1.0 # Prior variance on the BNN parameters
		self.bnn_training_epochs = 100 # number of epochs of SGD in each BNN update
		self.num_episodes_avg = 30 # number of episodes used in moving average reward to determine whether to stop DQN training 
		self.num_approx_episodes = 500 # number of approximated rollouts using the BNN to train the DQN
		self.state_diffs = True # Use BNN to predict (s'-s) rather than s'
		self.num_bnn_updates = 3 # number of calls to update_BNN()
		self.wb_learning_rate = 0.0005 # latent weight learning rate
		self.num_batch_updates = 5 # number of minibatch updates to DQN
		self.bnn_alpha = 0.5 # BNN alpha divergence parameter
		self.policy_update_interval = 10 # Main DQN update interval (in timesteps)
		self.target_update_interval = 10 # Target DQN update interval (in timesteps)
		self.ddqn_hidden_layer_sizes = [256,512] # DDQN hidden layer sizes
		self.eps_decay = 0.995 # Epsilon decay rate
		self.grad_clip = 2.5 # DDQN Gradient clip by norm
		self.ddqn_batch_size = 50 # DDQN batch size
		# Prioritized experience replay hyperparameters
		self.PER_alpha = 0.2
		self.PER_beta_zero = 0.1
		self.tau = 0.005 # DQN target network update proportion
		self.wb_num_epochs = 100 # number of epochs of SGD in each latent weight update

	def __set_domain_hyperparams(self):
		"""Set domain specific hyperparameters."""
		self.standardize_rewards = False
		self.standardize_states = False
		# Acrobot settings
		if self.domain == 'acrobot':
			from acrobot_simulator.acrobot import Acrobot as model
			if self.create_exp_batch:
				if self.num_batch_instances:
					self.instance_count = self.num_batch_instances
				else:
					self.instance_count = 8 # number of instances to include in experience batch
			self.max_task_examples = 400 # maximum number of time steps per episode
			self.min_avg_rwd_per_ep = -12 # minimum average reward before stopping DQN training
			self.bnn_learning_rate = 0.00025
			self.num_initial_update_iters = 5 # number of initial updates to the BNN and latent weights
			self.bnn_start = 400 # number of time steps observed before starting BNN training
			self.dqn_start = 400 # number of time steps observed before starting DQN training
		# Grid settings
		elif self.domain == 'grid':
			from grid_simulator.grid import Grid as model
			if self.create_exp_batch:
				if self.num_batch_instances:
					self.instance_count = self.num_batch_instances
				else:
					self.instance_count = 2
			self.max_task_examples = 100
			self.min_avg_rwd_per_ep = 980
			self.bnn_learning_rate = 0.00005
			self.num_initial_update_iters = 10
			self.num_approx_episodes = 1000 # Use extra approximated episodes since finding the goal state takes a bit of luck
			self.bnn_start = 100
			self.dqn_start = 100
			self.wb_num_epochs = 300
			if self.run_type_full:
				self.eps_decay = np.exp(np.log(self.eps_min)/self.num_approx_episodes)
			# In order to learn, model-based from scratch needs some adjustments
			if self.run_type == 'modelbased':
				self.bnn_learning_rate = 0.0005
				self.dqn_start = 400
		# HIV settings
		elif self.domain == 'hiv':
			from hiv_simulator.hiv import HIVTreatment as model
			if self.create_exp_batch:
				if self.num_batch_instances:
					self.instance_count = self.num_batch_instances
				else:
					self.instance_count = 5
			self.max_task_examples = 200
			self.min_avg_rwd_per_ep = 1e15
			self.bnn_learning_rate = 0.00025
			self.num_initial_update_iters = 10 
			self.bnn_start = 200
			self.dqn_start = 200
			self.standardize_rewards = True
			self.bnn_alpha = 0.45 # Alpha divergence hyper parameter
			self.bnn_batch_size = 100 # Draw 500 samples total
			self.standardize_states = True
		else:
			raise NameError('invalid domain')
		# Size of buffer for storing batch of experiences
		self.general_bnn_buffer_size = self.instance_count * self.max_task_examples * self.episode_count 
		# Size of experience buffer for test instance. Note: all experiences are stored
		self.instance_buffer_size = self.max_task_examples * self.episode_count
		# Size of fictional experience buffer
		self.instance_fictional_buffer_size = self.num_approx_episodes * self.episode_count
		if self.domain == 'grid':
			self.task = model(beta=self.grid_beta)
		else:
			self.task = model()
		self.var_params = self.task.perturb_params # names of hidden parameters to be varied
		self.num_actions = self.task.num_actions # number of actions
		self.num_dims = len(self.task.observe()) # number of state dimensions
		# create set of parameters for each experience replay instantiation
		self.experience_replay_param_set = {
			'episode_count': self.episode_count,
			'instance_count': self.instance_count,
			'max_task_examples': self.max_task_examples,
			'ddqn_batch_size': self.ddqn_batch_size,
			'num_strata_samples': self.num_strata_samples,
			'PER_alpha': self.PER_alpha,
			'PER_beta_zero': self.PER_beta_zero,
			'bnn_batch_size': self.bnn_batch_size,
			'dqn_start': self.dqn_start,
			'bnn_start': self.bnn_start
			}

	def __get_instance_param_set(self):
		"""Get preset hidden parameter setting for this instance."""
		if self.create_exp_batch:
			instance_idx = self.instance_iter
		else:
			instance_idx = self.test_inst
		self.instance_param_set = self.preset_hidden_params[instance_idx]

	def __encode_action(self, action):
		"""One-hot encodes the integer action supplied."""
		a = np.array([0] * self.num_actions)
		a[action] = 1
		return a

	def __load_reward_standardization(self):
		"""Load the reward mean and standard deviation."""
		with open('preset_parameters/'+self.domain+'_rewards_standardization','r') as f:
			self.rewards_standardization = pickle.load(f)

	def __load_state_standardization(self):
		"""Load the state mean and standard deviation."""
		with open('preset_parameters/'+self.domain+'_standardization_arrays','r') as f:
			self.state_mean, self.state_std = pickle.load(f)

	def __standardize_state(self,state):
		"""Standardize and return the given state."""
		return (state-self.state_mean) / self.state_std

	def __update_target_graph(self):
		"""Helper function for updating target DQN."""
		self.op_holder=[]
		total_vars = len(self.trainables)
		for idx , var in enumerate(self.trainables[0:total_vars/2]):
			self.op_holder.append(self.trainables[idx + total_vars/2].assign((var.value()*self.tau)+((1-self.tau) * self.trainables[idx + total_vars/2].value())))
		return self.op_holder

	def __update_target(self):
		""" Helper function for updating target DQN."""
		for op in self.op_holder:
			self.sess.run(op)

	def __apply_minibatch_update(self):
		"""Train the main DQN using minibatch updates."""
		if self.run_type == 'modelfree':
			exp_buffer = self.real_buffer
		else:
			exp_buffer = self.fictional_buffer
		for batch_idx in xrange(self.num_batch_updates):
			# Draw experience sample and importance weights
			train_batch, is_weights, indices = exp_buffer.sample(self.instance_steps)
			# Calculate DDQN target
			feed_dict = {self.mainDQN.s:np.vstack(train_batch[:,3])}
			Q1 = self.sess.run(self.mainDQN.predict,feed_dict=feed_dict)
			feed_dict = {self.targetDQN.s:np.vstack(train_batch[:,3])}
			Q2 = self.sess.run(self.targetDQN.output,feed_dict=feed_dict)
			double_Q = Q2[range(train_batch.shape[0]),Q1]
			target_Q = train_batch[:,2] + self.discount_rate*double_Q
			# Calculate TD errors of the sample
			feed_dict = {self.mainDQN.s:np.vstack(train_batch[:,0]), 
				self.mainDQN.next_Q:target_Q, self.mainDQN.action_array:np.vstack(train_batch[:,1])}
			td_loss = self.sess.run(self.mainDQN.td_loss, feed_dict=feed_dict)
			# Update priority queue with the observed td_loss from the selected minibatch and reinsert sampled batch into the priority queue
			if self.run_type == 'modelfree':
				self.real_buffer.update_priorities(np.hstack((np.reshape(td_loss,(len(td_loss),-1)), 
					np.reshape(indices,(len(indices),-1)))))
			else:
				self.fictional_buffer.update_priorities(np.hstack((np.reshape(td_loss,(len(td_loss),-1)), 
					np.reshape(indices,(len(indices),-1)))))
			# Update the DDQN
			feed_dict = {self.mainDQN.s:np.vstack(train_batch[:,0]), 
				self.mainDQN.next_Q:target_Q, 
				self.mainDQN.action_array:np.vstack(train_batch[:,1]),
				self.mainDQN.importance_weights:is_weights, 
				self.mainDQN.learning_rate:self.ddqn_learning_rate
				}
			self.sess.run(self.mainDQN.updateQ, feed_dict=feed_dict)

	def __initialize_BNN(self):
		"""Initialize the BNN and set pretrained network weights (if supplied)."""
		# Generate BNN layer sizes
		if self.run_type != 'full_linear':
			bnn_layer_sizes = [self.num_dims+self.num_actions+self.weight_count] + [self.bnn_hidden_layer_size]*self.bnn_num_hidden_layers + [self.num_dims]
		else:
			bnn_layer_sizes = [self.num_dims+self.num_actions] + [self.bnn_hidden_layer_size]*self.bnn_num_hidden_layers + [self.num_dims*self.weight_count]
		# activation function
		relu = lambda x: np.maximum(x, 0.0)
		# Gather parameters
		param_set = {
			'bnn_layer_sizes': bnn_layer_sizes,
			'weight_count': self.weight_count,
			'num_state_dims': self.num_dims,
			'bnn_num_samples': self.bnn_num_samples,
			'bnn_batch_size': self.bnn_batch_size,
			'num_strata_samples': self.num_strata_samples,
			'bnn_training_epochs': self.bnn_training_epochs,
			'bnn_v_prior': self.bnn_v_prior,
			'bnn_learning_rate': self.bnn_learning_rate,
			'bnn_alpha': self.bnn_alpha,
			'wb_learning_rate': self.wb_learning_rate,
			'wb_num_epochs': self.wb_num_epochs
			}
		if self.run_type != 'full_linear':
			self.network = BayesianNeuralNetwork(param_set, nonlinearity=relu)
		else:
			self.network = BayesianNeuralNetwork(param_set, nonlinearity=relu, linear_latent_weights=True)
		# Use previously trained network weights
		if self.bnn_network_weights is not None:
			self.network.weights = self.bnn_network_weights

	def __initialize_DDQN(self):
		"""Initialize Double DQN."""
		tf.reset_default_graph()
		self.mainDQN = Qnetwork(self.num_dims, self.num_actions, clip=self.grad_clip, activation_fn=tf.nn.relu, hidden_layer_sizes=self.ddqn_hidden_layer_sizes)
		self.targetDQN = Qnetwork(self.num_dims, self.num_actions, clip=self.grad_clip, activation_fn=tf.nn.relu, hidden_layer_sizes=self.ddqn_hidden_layer_sizes)
		init = tf.global_variables_initializer()
		self.trainables = tf.trainable_variables()
		self.targetOps = self.__update_target_graph()
		self.sess = tf.Session()
		self.sess.run(init)
		self.__update_target()

	def __e_greedy_policy(self,state):
		"""Select action using epsilon-greedy policy."""
		if np.random.rand(1) < self.eps:
			action = np.random.randint(0, self.num_actions)
		else:
			action = self.sess.run(self.mainDQN.predict,feed_dict={self.mainDQN.s:state.reshape(1,-1)})[0]
			self.action_counts += self.__encode_action(action)
		return action

	def __update_BNN(self):	
		"""Update BNN using data from test instance."""
		self.network.fit_network(self.instance_bnn_buffer, self.full_task_weights, self.instance_steps, state_diffs=self.state_diffs, use_all_exp=False)
		if self.print_output:
			print('Updated BNN after episode {}'.format(self.episode_iter))

	def __update_latent_weights(self):
		"""Update Latent weights using data from test instance"""
		self.weight_set = self.network.optimize_latent_weighting_stochastic(self.instance_bnn_buffer, self.weight_set, self.instance_steps, state_diffs=self.state_diffs,use_all_exp=False)
		self.full_task_weights[self.instance_iter,:] = self.weight_set
		if self.print_output:
			print('Updated latent weights after episode {}'.format(self.episode_iter))

	def __compute_bnn_training_error(self):
		"""Compute BNN training error on most recent episode."""
		exp = np.reshape(self.episode_buffer_bnn, (len(self.episode_buffer_bnn),-1))
		episode_X = np.array([np.hstack([exp[tt,0],exp[tt,1]]) for tt in xrange(exp.shape[0])])
		episode_Y = np.array([exp[tt,3] for tt in xrange(exp.shape[0])])
		if self.state_diffs:
			# subtract previous state
			episode_Y -= episode_X[:,:self.num_dims]
		l2_errors = self.network.get_td_error(np.hstack([episode_X, np.tile(self.weight_set, (episode_X.shape[0],1))]), episode_Y, 0.0, 1.0)
		self.mean_episode_errors[self.instance_iter,self.episode_iter] = np.mean(l2_errors)
		self.std_episode_errors[self.instance_iter,self.episode_iter] = np.std(l2_errors)
		if self.print_output:
			print('BNN Error: {}'.format(self.mean_episode_errors[self.instance_iter,self.episode_iter]))

	def __sample_start_state(self):
		"""Randomly choose and return a start state from a list of all observed start states"""
		return self.start_states[np.random.randint(0,len(self.start_states))]
	
	def run_fictional_episode(self):
		"""Perform an episode using the BNN to approximate transitions without using the environment. Train DQN using this
		approximate experience data.
		"""
		ep_steps = 0
		ep_reward = 0 
		# Sample start state from previously observed start states.
		state = np.copy(self.__sample_start_state())
		r_fake = 0.0
		# Keep track of the unstandardized reward since it is more interpretable
		if self.standardize_rewards:
			un_std_reward = 0.0
		while ep_steps < self.max_task_examples:
			ep_steps += 1
			self.approx_steps += 1
			action = self.__e_greedy_policy(state)
			aug_state = np.hstack([state, self.__encode_action(action), self.weight_set.reshape(self.weight_set.shape[1],)]).reshape((1,-1))
			# Note: if self.standardize_states==True, then the BNN output is a standardized state 
			next_state = self.network.feed_forward(aug_state).flatten()
			if self.state_diffs:
				next_state += state
			# Undo the state standardization in order calculate the reward
			if self.standardize_states:
				reward_state = state*self.state_std + self.state_mean
				reward_next_state = next_state*self.state_std + self.state_mean
			else:
				reward_state = state
				reward_next_state = next_state

			if self.domain == 'grid':
				# In the grid domain, the reward is calculated as R(s,a)
				reward = self.task.calc_reward(action=action, state=reward_state, latent_code=self.instance_param_set)
			else:
				# In all other domains, the reward is calculated as R(s',a)
				reward = self.task.calc_reward(action=action, state=reward_next_state, latent_code=self.instance_param_set)
			
			if self.standardize_rewards:
				reward = (reward-self.reward_mean) / self.reward_std
				un_std_reward += reward
			r_fake += reward
			self.fictional_buffer.add(np.reshape(np.array([state,self.__encode_action(action),reward,next_state]),(1,4)))
			state = next_state
			if self.approx_steps >= self.dqn_start and self.train_dqn:
				# Update Main DQN
				if self.approx_steps % self.policy_update_interval == 0:
					self.__apply_minibatch_update()
				# Update Target DQN
				if self.approx_steps % self.target_update_interval == 0:
					self.__update_target()
		if self.print_output:
			print('Completed Instance {}, Approx. Episode {}'.format(self.instance_iter,self.approx_episode_iter))
			if self.standardize_rewards:
				print('BNN Reward: {}'.format(un_std_reward))
			else:
				print('BNN Reward: {}'.format(r_fake))
			print('Action counts: {}'.format(self.action_counts))

	def run_episode(self):
		"""Run an episode on the environment (and train DQN if modelfree)."""
		self.task.reset(perturb_params = True, **self.instance_param_set)
		state = self.task.observe()
		if self.standardize_states:
			state = self.__standardize_state(state)
		self.start_states.append(state)
		self.episode_buffer_bnn = []
		ep_reward = 0
		# task is done after max_task_examples timesteps or when the agent enters a terminal state
		while not self.task.is_done(self.max_task_examples):
			self.total_steps += 1
			self.instance_steps += 1
			action = self.__e_greedy_policy(state)
			reward , next_state = self.task.perform_action(action, perturb_params=True, **self.instance_param_set)
			ep_reward += reward
			if self.standardize_rewards:
				reward = (reward-self.reward_mean) / self.reward_std
			if self.standardize_states:
				next_state = self.__standardize_state(next_state)
			if self.run_type == "modelfree":
				self.real_buffer.add(np.reshape(np.array([state, self.__encode_action(action), reward,next_state]),[1,4]))
			self.episode_buffer_bnn.append(np.reshape(np.array([state, self.__encode_action(action), reward,next_state, self.instance_iter]),[1,5]))
			state = next_state
			# For modelfree runs, update DQN using experience from the environment on set interval
			if self.run_type == 'modelfree' and self.instance_steps >= self.dqn_start and self.train_dqn:
				# Update Main DQN
				if self.instance_steps % self.policy_update_interval  == 0:
					self.__apply_minibatch_update()
				# Update Target DQN
				if self.instance_steps % self.target_update_interval == 0:
					self.__update_target()
		# Store results at the end of the episode
		self.rewards[self.instance_iter,self.episode_iter] = ep_reward
		# calculate moving average reward
		last_rwds = self.rewards[self.instance_iter,np.maximum(self.episode_iter - self.num_episodes_avg + 1, 0):self.episode_iter + 1]
		self.avg_rwd_per_ep[self.instance_iter,self.episode_iter] = np.mean(last_rwds)
		# if creating experience batch, store experience in general buffer
		if self.create_exp_batch:
			self.general_bnn_buffer.add(np.reshape(self.episode_buffer_bnn, [-1,5]))
		# If using a model-based approach, store in instance buffer for updating BNN (and latent weights for full runs)
		if self.run_type != 'modelfree':
			self.instance_bnn_buffer.add(np.reshape(self.episode_buffer_bnn, [-1,5]))
		if self.print_output:
			print('Completed Instance {}, Episode {}'.format(self.instance_iter, self.episode_iter))
			print('Total Reward: {}'.format(ep_reward))
			print('Epsilon: {}'.format(self.eps))
			print('Moving Average Reward: {}'.format(self.avg_rwd_per_ep[self.instance_iter, self.episode_iter]))
			print('Action counts: {}'.format(self.action_counts))

	def run_instance(self):
		"""Learn a policy and update the BNN (if desired) over the course of a single instance."""
		# Get hidden parameter setting for this instance
		self.__get_instance_param_set()
		self.sys_param_set.append(self.instance_param_set)
		# Initialize latent weights
		# If full hipmdp run use random latent weights or mean latent weights
		if self.run_type_full:
			self.weight_set = np.atleast_2d(np.random.normal(0, 0.1, self.weight_count))
		# Otherwise use ones
		else:
			self.weight_set = np.atleast_2d(np.ones(self.weight_count))
		self.full_task_weights[self.instance_iter,:] = self.weight_set
		# Initialize DQN and BNN
		self.__initialize_DDQN()
		if self.run_type == 'modelfree':
			self.train_dqn = True
			self.train_bnn = False
			self.initial_bnn_collection = False
		else:
			self.__initialize_BNN()
			self.train_dqn = False
			self.initial_bnn_collection = True # initial collection period for BNN
			self.train_bnn = True
		# Initialize experience buffers
		if self.run_type == 'modelfree':
			# Prioritized experience replay used for training modelfree DQN off environment
			self.real_buffer = ExperienceReplay(self.experience_replay_param_set, buffer_size=self.instance_buffer_size)
		else:
			# Prioritized experience replay used for training model-based DQNs off approximated BNN rollouts
			self.fictional_buffer = ExperienceReplay(self.experience_replay_param_set, buffer_size=self.instance_fictional_buffer_size)
			# Prioritized experience replay used for training BNN (and latent weights) off environment
			self.instance_bnn_buffer = ExperienceReplay(self.experience_replay_param_set, buffer_type='BNN', buffer_size=self.instance_buffer_size)
		# Load Reward Standardization for this instance
		if self.standardize_rewards:
			if self.create_exp_batch:
				instance_idx = self.instance_iter
			else:
				instance_idx = self.test_inst
			self.reward_mean, self.reward_std = self.rewards_standardization[instance_idx]
		# Other initializations
		self.eps = self.eps_max # Set Epsilon
		self.start_states = []
		self.action_counts = np.zeros(self.num_actions) # Store the counts of each on-policy action
		
		self.episode_iter = 0 # episode number
		self.instance_steps = 0 # number of steps this instance on the environment
		self.approx_steps = 0 # number of steps taking using approximated BNN rollout
		self.approx_episode_iter = 0 # approximated rollout number
		just_completed_first_update = False
		# Run episodes
		while self.episode_iter < self.episode_count:
			self.run_episode()
			if self.run_type != 'modelfree' and not self.initial_bnn_collection:
				self.run_fictional_episode()
				self.approx_episode_iter += 1
			if self.run_type != 'modelfree':
				self.__compute_bnn_training_error()
			# Update BNN and latent weights on set interval
			if  self.run_type != 'modelfree':
				if ((self.instance_steps >= self.bnn_start and self.initial_bnn_collection) 
				  or ((self.episode_iter+1) % self.bnn_and_latent_update_interval == 0) or just_completed_first_update) and self.train_bnn:
					# For full runs: oscillate between updating latent weights and updating BNN
					# For all other model-based benchmarks: only update BNN

					# Perform additional BNN/Latent weight updates after first before starting to train DQN
					# and after first set of approximated rollouts
					if self.initial_bnn_collection or just_completed_first_update:
						just_completed_first_update = not just_completed_first_update
						exp = np.reshape(self.episode_buffer_bnn, (len(self.episode_buffer_bnn),-1))
						episode_X = np.array([np.hstack([exp[tt,0],exp[tt,1]]) for tt in range(exp.shape[0])])
						episode_Y = np.array([exp[tt,3] for tt in range(exp.shape[0])])
						if self.state_diffs:
							# subtract previous state
							episode_Y -= episode_X[:,:self.num_dims]
						for update_iter in xrange(self.num_initial_update_iters):	
							if self.run_type_full:
								self.__update_latent_weights()
								l2_errors = self.network.get_td_error(np.hstack([episode_X, np.tile(self.weight_set, (episode_X.shape[0],1))]), episode_Y, 0.0, 1.0)
								if self.print_output:
									print('BNN Error after latent update iter {}: {}'.format(update_iter,np.mean(l2_errors)))
							self.__update_BNN()
							l2_errors = self.network.get_td_error(np.hstack([episode_X, np.tile(self.weight_set,(episode_X.shape[0],1))]), episode_Y, 0.0, 1.0)
							if self.print_output:
								print('BNN Error after BNN update iter {} : {}'.format(update_iter,np.mean(l2_errors)))
					else:
						for update_iter in xrange(self.num_bnn_updates):
							if self.run_type_full:
								self.__update_latent_weights()
							self.__update_BNN()
					
					# Start training DQN after dqn_start steps on the real environment
					if self.initial_bnn_collection and self.instance_steps >= self.dqn_start:
						self.train_dqn = True
						self.initial_bnn_collection = False
						# Approximate Episodes
						while self.approx_episode_iter < self.num_approx_episodes:
							self.run_fictional_episode() 
							self.approx_episode_iter += 1
							# Decay Epsilon
							if self.eps > self.eps_min:
								self.eps *= self.eps_decay
			# Decay Epsilon
			if self.instance_steps > self.dqn_start and self.eps > self.eps_min:
				self.eps *= self.eps_decay
			# Stop training if good policy has been learned
			if (self.avg_rwd_per_ep[self.instance_iter, self.episode_iter] >= self.min_avg_rwd_per_ep) and (self.episode_iter + 1 >= self.num_episodes_avg):
				self.train_dqn = False
				self.eps = self.eps_min
				if self.print_output:
					print('Reached minimum average reward.  Stopping training.')
			self.episode_iter += 1
		# Close TensorFlow Session
		self.sess.close()

	def run_experiment(self):
		"""Run the experiment: either creating a batch of experience or testing a method on a single instance."""
		if self.create_exp_batch:
			self.general_bnn_buffer = ExperienceReplay(self.experience_replay_param_set, 
				buffer_type='BNN', 
				buffer_size=self.general_bnn_buffer_size, 
				general=True,
				mem_priority=False
				)
		# total reward per episode
		self.rewards = np.zeros((self.instance_count, self.episode_count))
		# moving average of total rewards per episode
		self.avg_rwd_per_ep = np.zeros((self.instance_count, self.episode_count))
		self.total_steps = 0
		# Storage for latent weighting of each instance
		self.full_task_weights = np.zeros((self.instance_count, self.weight_count))
		# Storage for hidden parameters for each instance
		self.sys_param_set = []
		# Storage for BNN Training Errors
		self.mean_episode_errors = np.zeros((self.instance_count, self.episode_count))
		self.std_episode_errors = np.zeros((self.instance_count, self.episode_count))
		self.instance_iter = 0

		if self.standardize_rewards:
			self.__load_reward_standardization()

		if self.standardize_states:
			self.__load_state_standardization()

		save_filename = self.domain + '_' + self.run_type + '_results_inst'

		while self.instance_iter < self.instance_count:
			self.run_instance()
			# Save results
			networkweights = None
			if self.run_type != 'modelfree':
				networkweights = self.network.weights
			# Save off current results
			if self.create_exp_batch:
				exp_buffer = self.general_bnn_buffer.exp_buffer
			else:
				if self.run_type != 'modelfree':
					exp_buffer = self.instance_bnn_buffer.exp_buffer
				else:
					exp_buffer = self.real_buffer.exp_buffer
			if self.save_results:
				with open(self.domain + '_' + self.run_type + '_results_inst' + str(self.default_inst) + '_uaiHiP_larger_exp_replay_preload_{}'.format(self.run),'w') as f:
					pickle.dump((exp_buffer, networkweights, self.rewards, self.avg_rwd_per_ep, self.full_task_weights,
						self.sys_param_set, self.mean_episode_errors, self.std_episode_errors), f)
			self.instance_iter += 1
		return (exp_buffer, networkweights, self.rewards, self.avg_rwd_per_ep, self.full_task_weights,
						self.sys_param_set, self.mean_episode_errors, self.std_episode_errors)

