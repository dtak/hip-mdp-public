import tensorflow as tf
import tensorflow.contrib.slim as slim
"""
A DQN class to implement Deep Q Learning. The implementation is of 
a Double DQN with 2 fully connected hidden layers.with the option to clip the gradient by its L2 norm.
The implementation is derived from Juliani's implementation in TensorFlow: 
https://github.com/awjuliani/DeepRL-Agents
"""

class Qnetwork():
	"""The network receives a batch of states and is tasked with learning the Q-function
	that provides values for each possible action."""
	def __init__(self, state_dims, num_actions, clip = 2.5, activation_fn=tf.nn.relu, hidden_layer_sizes=(256,512)):
		# Create the model/establish the feed-forward part of the network
		self.s = tf.placeholder(tf.float32,[None,state_dims])
		self.next_Q = tf.placeholder(shape=[None],dtype=tf.float32)
		self.action_array = tf.placeholder(shape=[None,num_actions],dtype=tf.float32)
		self.importance_weights = tf.placeholder(shape=[None],dtype=tf.float32)
		self.learning_rate = tf.placeholder(dtype=tf.float32)
		self.activation_fn = activation_fn

		self.hidden1_units = hidden_layer_sizes[0]
		self.hidden2_units  = hidden_layer_sizes[1]
		# Hidden layer 1
		self.h1_weights = tf.Variable(tf.truncated_normal([state_dims,self.hidden1_units], stddev=1.0/float(state_dims*self.hidden1_units)), name='weights')
		self.h1_biases = tf.Variable(tf.zeros([self.hidden1_units]))
		self.hidden1 = self.activation_fn(tf.matmul(self.s,self.h1_weights) + self.h1_biases)
		# Hidden layer 2
		self.h2_weights = tf.Variable(tf.truncated_normal([self.hidden1_units, self.hidden2_units], stddev=1.0/float(self.hidden1_units*self.hidden2_units)),name='weights')
		self.h2_biases = tf.Variable(tf.zeros([self.hidden2_units]))
		self.hidden2 = self.activation_fn(tf.matmul(self.hidden1,self.h2_weights) + self.h2_biases)
		# Output layer
		self.out_weights = tf.Variable(tf.truncated_normal([self.hidden2_units, num_actions], stddev=1.0/float(self.hidden2_units*num_actions)),name='weights')
		self.out_biases = tf.Variable(tf.zeros([num_actions]))
		self.output = tf.matmul(self.hidden2,self.out_weights) + self.out_biases
		self.predict = tf.argmax(self.output,1)
		self.curate_output = tf.reduce_sum(tf.mul(self.action_array,self.output),1)
		self.td_error = self.next_Q - tf.reduce_sum(tf.mul(self.action_array,self.output),1)
		# Define the loss according the the bellman equations
		self.td_loss = 0.5*tf.square(self.td_error)

		self.loss = tf.reduce_sum(tf.mul(self.importance_weights, self.td_loss))
		if clip > 0: # Train and backpropagate clipped errors

			def ClipIfNotNone(gradient):
				if gradient is None:
					return gradient
				return tf.clip_by_norm(gradient, clip)

			self.objective = tf.train.AdamOptimizer(learning_rate = self.learning_rate,epsilon=0.01) 
			self.grads_and_vars = self.objective.compute_gradients(self.loss)
			self.clipped_grads = [ (ClipIfNotNone(grad), var) for grad, var in self.grads_and_vars ]
			self.updateQ = self.objective.apply_gradients(self.clipped_grads)
		else: # Update the network according to gradient descent minimization of the weighted loss
			self.updateQ = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss)


