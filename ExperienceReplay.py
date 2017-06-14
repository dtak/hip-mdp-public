"""
Prioritized experience replay implementation adapted from https://arxiv.org/abs/1511.05952
"""
import autograd.numpy.random as npr
import autograd.numpy as np
import math
from PriorityQueue import PriorityQueue

class ExperienceReplay():
	def __init__(self, param_set, buffer_size=200000, buffer_type='Qnetwork', mem_priority=True, general=False):
		"""Initialize the storage containers and parameters relevant for experience replay.
		
		Arguments:
		param_set --  dictionary of parameters which must contain:
			PER_alpha -- hyperparameter governing how much prioritization is used
			PER_beta_zero -- importance sampling parameter initial value
			bnn_start -- number of timesteps before sample will be drawn; i.e the minimum partition size (necessary if buffer_type=='BNN')
			dqn_start -- same as dqn_start (necessary if buffer_type=='Qnetwork')
			episode_count -- number of episodes
			instance_count -- number of instances
			max_task_examples -- maximum number of timesteps per episode
			ddqn_batch_size -- minibatch size for DQN updates (necessary if buffer_type=='Qnetwork')
			bnn_batch_size -- minibatch size for BNN updates (necessary if buffer_type=='BNN')
			num_strata_samples -- number of samples to be drawn from each strata in the prioritized replay buffer
			general_num_partitions -- number of partitions for general experience buffer
			instance_num_partitions -- number of partitions for instance experience buffer

		Keyword arguments:
		buffer_size -- maximum capacity of the experience buffer (default: 200000)
		buffer_type -- string indicating whether experience replay is for training a DQN or a BNN (either 'Qnetwork' or 'BNN'; default: 'Qnetwork')
		mem_priority -- boolean indicating whether the experience replay should be prioritized (default: True)
		general -- boolean indicating if the experience replay is for collecting experiences over multiple instances or a single (default: False)
		"""
		# Extract/Set relevant parameters
		self.mem_priority = mem_priority
		self.alpha = param_set['PER_alpha']
		self.beta_zero = param_set['PER_beta_zero']
		self.capacity = buffer_size
		self.is_full = False
		self.index = 0 # Index number in priority queue where next transition should be inserted
		self.size = 0 # Current size of experience replay buffer
		if buffer_type == 'Qnetwork':
			self.num_init_train = param_set['dqn_start']
			self.tot_steps = param_set['episode_count'] * param_set['max_task_examples']
			self.batch_size = param_set['ddqn_batch_size']
		elif buffer_type == 'BNN':
			self.num_init_train = param_set['bnn_start']
			self.tot_steps = (param_set['episode_count'] * param_set['instance_count']) * param_set['max_task_examples']
			self.batch_size = param_set['bnn_batch_size']
		self.beta_grad = (1-self.beta_zero) / (self.tot_steps-self.num_init_train)
		self.num_strata_samples = param_set['num_strata_samples']
		# Note: at least one partition must be completely filled in order for the sampling procedure to work
		self.num_partitions = self.capacity / (1.0*self.num_init_train)
		# Initialize experience buffer
		self.exp_buffer = []
		# Initialize rank priority distributions and stratified sampling cutoffs if needed
		if self.mem_priority: 
			# Initialize Priority Queue (will be implemented as a binary heap)
			self.pq = PriorityQueue(capacity=buffer_size)
			self.distributions = {}
			partition_num = 1
			partition_division = self.capacity/self.num_partitions
			for n in np.arange(partition_division, self.capacity+0.1, partition_division):
        		# Set up power-law PDF and CDF
				distribution = {}
				distribution['pdf'] = np.power(np.linspace(1, n, n),-1*self.alpha)
				pdf_sum = np.sum(distribution['pdf'])
				distribution['pdf'] = distribution['pdf']/float(pdf_sum) # Normalise PDF
				cdf = np.cumsum(distribution['pdf'])
				# Set up strata for stratified sampling (transitions will have varying TD-error magnitudes)
				distribution['strata_ends'] = np.zeros(self.batch_size + 1)
				distribution['strata_ends'][0] = 0 # First index is 0 (+1)
				distribution['strata_ends'][self.batch_size] = n # Last index is n
				# Use linear search to find strata indices
				stratum = 1.0/self.batch_size
				index = 0
				for s in range(1,self.batch_size):
					if cdf[index] >= stratum:
						index += 1
					while cdf[index] < stratum:
						index = index + 1
					distribution['strata_ends'][s] = index
					stratum = stratum + 1.0/self.batch_size # Set condition for next stratum
				# Store distribution
				self.distributions[partition_num] = distribution
				partition_num = partition_num + 1

	def get_size(self):
		"""Return the number of elements in the experience buffer."""
		return self.size

	def store(self, exp, priority=None):
		"""Stores a single transition tuple in the experience buffer.

		Arguments:
		exp -- numpy array containg single transition

		Keyword arguments:
		priority -- priority for this transition; if None, then maximum priority is used (default: None)
		"""
		# Increment size, circling back to begninning if memory limit is reached
		self.size = np.min((self.size + 1, self.capacity))
		if self.index >= self.capacity:
			self.is_full = True
			self.index = 0
		# If prioritized replay, update priority queue
		if self.mem_priority:
			if priority is None:
				# Store with max priority
				priority = self.pq.find_max() or 1
			# Add to priority queue
			if self.is_full:
				
				self.pq.update_by_val(self.index, priority, self.index)
			else:
				self.pq.insert(priority, self.index)
		# Add experience to buffer
		if self.is_full:
			self.exp_buffer[self.index] = exp
		else:
			self.exp_buffer.append(exp)
		self.index += 1

	def add(self, exp_list):
		"""Store observed transitions.

		Arguments:
		exp_list -- list of transitions (each is a numpy array)
		"""
		# loop over experiences and store each one
		for trans_idx in xrange(len(exp_list)):
			self.store(exp_list[trans_idx])

	def add_with_priorities(self, exp_list, priorities_list):
		"""Store observed transitions and associated priorities.
		
		Arguments:
		exp_list -- list of transitions (each is a numpy array)
		priorities_list -- list of priorities (one for each transition)
		"""
		for trans_idx in xrange(len(exp_list)):
			self.store(exp_list[trans_idx], priorities_list[trans_idx])

	def sample(self, total_steps=0):
		"""Sample from the experience buffer by rank prioritization if specified.
		Otherwise sampling is done uniformly.

		Keyword arguments:
		total_steps -- number of steps taken in experiment (default: 0)
		"""

		N = self.size
		num_samples = np.min((self.batch_size*self.num_strata_samples, self.size))

		# Perform uniform sampling of experience buffer
		if not self.mem_priority: 
			indices = npr.choice(range(N), replace=False, size=num_samples)
			exp_batch = np.array(self.exp_buffer)[indices]
			weights = np.ones(len(indices)) / (len(indices)*1.0)
			return np.reshape(exp_batch,(num_samples,-1)), weights, indices
		# Perform prioritized sampling of experience buffer
		else:
			# Find the closest precomptued distribution by size
			dist_idx = math.floor(N / float(self.capacity) * self.num_partitions) 
			distribution = self.distributions[int(dist_idx)]
			N = dist_idx * 100
			rank_indices_set = set()
			# Perform stratified sampling of priority queue 
			for i_exp in range(num_samples)[::-1]:
				# To increase the training batch size we sample several times from each strata, repeated indices are eliminated
				rank_indices_set.add(npr.randint(distribution['strata_ends'][i_exp/self.num_strata_samples], distribution['strata_ends'][(i_exp/self.num_strata_samples) + 1]))
			rank_indices = list(rank_indices_set)
			exp_indices = self.pq.get_values_by_val(rank_indices)
			exp_batch = [self.exp_buffer[int(exp_idx)] for exp_idx in exp_indices]
			
			# Compute importance sampling weights
			beta = np.min([self.beta_zero + (total_steps-self.num_init_train-1)*self.beta_grad, 1])
			IS_weights = np.power(N*distribution['pdf'][rank_indices], -1*beta)

			# Normalize IS_weights by maximum weight, guarantees that IS weights only scale downwards
			w_max = np.max(IS_weights)
			IS_weights = IS_weights / float(w_max)
			return np.reshape(exp_batch,(len(exp_indices),-1)), IS_weights, exp_indices

	def rand_unif_sample(self, n):
		"""Returns a random uniform sample of n experiences.
		
		Arguments:
		n -- number of transitions to sample
		"""
		indices = npr.choice(range(self.size), replace=False, size=n)
		exp_batch = np.array(self.exp_buffer)[indices]
		return np.reshape(exp_batch,(n, -1))


	def update_priorities(self, pq_insert_list):
		"""Update priorities of sampled transitions.
		
		Arguments:
		pq_insert_list -- list containing [priority, exp_index] for each transition in the sample
		"""
		for i in xrange(len(pq_insert_list)):
			priority = pq_insert_list[i][0]
			exp_idx = pq_insert_list[i][1]
			self.pq.update_by_val(exp_idx, priority, exp_idx)


