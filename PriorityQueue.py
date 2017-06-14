from __future__ import print_function
import numpy as np
"""
Implenentation of a Priority Queue using a (Maximum) Binary Heap.
This code was based on and adapted from:
https://github.com/Kaixhin/Atari/blob/master/structures/BinaryHeap.lua
"""

class PriorityQueue(object):
	"""
	Priority Queue consists of:
		- pq_array: an Nx2 array storing [priority, exp_idx]
			- priority: absolute TD-error
			- exp_idx: experience replay index
		- exp_hash: 
			- key: experience replay index (exp_idx)
			- value: priority queue array index (pq_idx)
		- pq_hash:
			- key: priority queue array index (pq_idx)
			- value: experience replay index (exp_idx)

	Implementation Details:
		- Indices of connected nodes:
			- Parent(i) = ceil(i/2)-1
			- Left_Child(i) = 2i + 1
			- Right_Child(i) = 2i + 2
		- Root node is at priority queue array index (pq_idx) 0

	"""
	def __init__(self, capacity=200000):
		"""
		Initialize Binary Heap
		"""
		# Initialize hash tables
		self.exp_hash = {}
		self.pq_hash = {}
		# Number of elements in the binary heap
		self.size = 0
		# Initialize priority queue array
		self.pq_array = np.zeros((capacity,2))

	def __is_full(self):
		"""
		Check if Priority Queue is full
		"""
		if self.size == self.pq_array.shape[0]:
			return True

	def insert(self, priority, val):
		"""
		Insert a new value
		"""
		# Check if already full
		if self.__is_full():
			print("Priority queue is full. Can not insert ({}, {})".format(priority,val))
			return
		# Insert new value at end
		self.pq_array[self.size,0] = priority
		self.pq_array[self.size,1] = val
		# Update hash tables
		self.exp_hash[val] = self.size
		self.pq_hash[self.size] = val
		# Rebalance
		self.__up_heap(self.size)
		# Increment element counter
		self.size += 1

	def __up_heap(self, i):
		"""
		Rebalance the heap by moving large values up
		"""
		# Calculate parent index
		p = int(np.ceil(i/2.0)) - 1
		if i > 0:
			# If parent i smaller than child, then swap
			if self.pq_array[p,0] < self.pq_array[i,0]:
				self.pq_array[p], self.pq_array[i] = np.copy(self.pq_array[i]), np.copy(self.pq_array[p])
				# Update hash tables
				self.exp_hash[self.pq_array[i,1]], self.exp_hash[self.pq_array[p,1]], self.pq_hash[i], self.pq_hash[p] = i, p, self.pq_array[i,1], self.pq_array[p,1]
				# Continue rebalancing
				self.__up_heap(p)

	def pop(self):
		"""
		Removes and returns [priority, exp_idx] for the 
		the maxmimum priority element
		"""
		if self.size == 0:
			return None
		# Get max element (first element in pq_array)
		max_elt = np.copy(self.pq_array[0])
		# Most the last value (not necessarily the smallest) to the root
		self.pq_array[0] = self.pq_array[self.size-1]
		self.size -= 1
		# Update hash tables
		self.exp_hash[self.pq_array[0,1]], self.pq_hash[0] = 0, self.pq_array[0,1]
		# Rebalance
		self.__down_heap(0)
		return max_elt

	def __down_heap(self, i):
		"""
		Rebalances the heap (by moving small values down)
		"""
		# Calculate left and right child indices
		l = 2*i+1
		r = 2*i+2
		# Find index of the greatest of these elements
		if l < self.size and self.pq_array[l,0] > self.pq_array[i,0]:
			greatest = l
		else:
			greatest = i
		if r < self.size and self.pq_array[r,0] > self.pq_array[greatest,0]:
			greatest = r
		# Continue rebalancing if necessary
		if greatest != i:
			# swap elements at indices i, greatest
			self.pq_array[i], self.pq_array[greatest] = np.copy(self.pq_array[greatest]), np.copy(self.pq_array[i])
			# Update hash tables
			self.exp_hash[self.pq_array[i,1]], self.exp_hash[self.pq_array[greatest,1]], self.pq_hash[i], self.pq_hash[greatest] = i, greatest, self.pq_array[i,1], self.pq_array[greatest,1]

			self.__down_heap(greatest)

	def update(self, i, priority, val):
		"""
		Updates a value (exp_idx) (and rebalances).  i is the pq_idx
		"""
		if i >= self.size:
			print("Error: index {} is greater than the current size of the heap".format(i))
			return
		# Replace value
		self.pq_array[i,0] = priority
		self.pq_array[i,1] = val
		# Update hash tables
		self.exp_hash[val] = i
		self.pq_hash[i] = val
		# Rebalance
		self.__down_heap(i)
		self.__up_heap(i)

	def update_by_val(self, exp_idx, priority, val):
		"""
		Updates a value by using the value 
		(using the exp_hash table to translate exp_idx-->pq_idx)
		"""
		self.update(self.exp_hash[exp_idx], priority, val)

	def find_max(self):
		"""
		Returns the maxmimum priority of any element in the priority queue
		Note: this method modify the priority queue
		"""
		if self.size == 0:
			return None
		return self.pq_array[0,0]

	def get_priorities(self, order='pq'):
		"""
		Retrieves the priorities
		If order == 'pq': the priorities are ordered by pq_idx
		If order == 'exp': the priorities are ordered by exp_idx
		"""
		if order == 'exp':
			priorities = np.zeros(self.size)
			for exp_idx in xrange(self.size):
				pq_idx = self.exp_hash[exp_idx]
				priorities[exp_idx] = self.pq_array[pq_idx,0]
			return priorities
		return self.pq_array[:,0]

	def get_values(self):
		"""
		Retrieves the values (exp_idx's)
		"""
		return self.pq_array[:,1]

	def get_value_by_val(self, pq_idx):
		"""
		Retrieves the exp_idx by using the pq_idx using pq_hash table
		"""
		return self.pq_hash[pq_idx]

	def get_values_by_val(self, pq_indices):
		"""
		Retrieves the exp_idx for each pq_idx in pq_indices
		"""
		return [self.pq_hash[pq_idx] for pq_idx in pq_indices]

	def rebalance(self):
		"""
		Rebalances the binary heap.  Takes O(n log n) time to run.
		Avoid using, when possible.
		"""
		# Sort array by priority
		sorted_indices_by_priority = np.argsort(-self.pq_array[:,0])
		self.pq_array = self.pq_array[sorted_indices_by_priority]
		pq_indices = range(self.size)
		# Create hash tables
		self.pq_hash = dict(zip(pq_indices,self.pq_array[:,1]))
		self.exp_hash = dict(zip(self.pq_array[:,1],pq_indices))





