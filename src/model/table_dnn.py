import torch
import numpy as np
from model.util import sanitize_sacred_arguments

class MultitaskTabularNet(torch.nn.Module):

	def __init__(
		self, num_tasks, input_dim, t_limit=1, hidden_dims=[256, 256, 256],
		shared_layers=[True, True, False, False], time_embed_std=30,
		embed_size=256
	):
		"""
		Initialize a time-dependent dense neural network for tabular data.
		Arguments:
			`num_tasks`: number of tasks to output, T
			`input_dim`: dimension of input data, D
			`t_limit`: maximum time horizon
			`hidden_dims`: dimensions of each hidden layer; this is of length
				one fewer than the number of layers
			`shared_layers`: which layers of the network to share weights
				between tasks
			`time_embed_std`: standard deviation of random weights to sample for
				time embeddings
			`embed_size`: size of the time embeddings and input embeddings
		"""
		super().__init__()
		
		assert len(shared_layers) == len(hidden_dims) + 1
		assert embed_size % 2 == 0

		self.creation_args = locals()
		del self.creation_args["self"]
		del self.creation_args["__class__"]
		self.creation_args = sanitize_sacred_arguments(self.creation_args)
		
		self.num_tasks = num_tasks
		self.t_limit = t_limit
		self.shared_layers = shared_layers

		layer_to_iter = lambda layer_i: range(
			1 if shared_layers[layer_i] else num_tasks
		)

		# Random embedding layer for time; the random weights are set at the
		# start and are not trainable
		self.time_embed_rand_weights = torch.nn.Parameter(
			torch.randn(embed_size // 2) * time_embed_std,
			requires_grad=False
		)
		
		# Initial dense layers to generate input embedding (always shared)
		self.time_dense_1 = torch.nn.Linear(embed_size, embed_size)
		self.time_dense_2 = torch.nn.Linear(embed_size, embed_size)
		self.input_dense = torch.nn.Linear(input_dim, embed_size)
	   
		# Dense layers that form the bulk of the network
		self.dense_layers = torch.nn.ModuleList()  # List of lists
		for i in range(len(shared_layers)):
			if i == 0:
				in_size, out_size = embed_size, hidden_dims[i]
			elif i < len(shared_layers) - 1:
				in_size, out_size = hidden_dims[i - 1], hidden_dims[i]
			else:
				in_size, out_size = hidden_dims[i - 1], input_dim
			layer_list = []
			self.dense_layers.append(torch.nn.ModuleList([
				torch.nn.Linear(in_size, out_size)
				for _ in layer_to_iter(i)
			]))

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)
		self.relu = torch.nn.ReLU()

	def forward(self, xt, t, task_inds=None):
		"""
		Forward pass of the network.
		Arguments:
			`xt`: B x D tensor containing the images to train on
			`t`: B-tensor containing the times to train the network for each
				input
			`task_inds`: an iterable of task indices to generate predictions
				for; if specified, the output tensor will be
				B x `len(task_inds)` x D instead of B x T x D
		Returns a B x T x D tensor which consists of the prediction.
		"""
		# Get the time embeddings for `t`
		# We embed the time as cos((t/T) * (2pi) * z) and sin((t/T) * (2pi) * z)
		time_embed_args = (t[:, None] / self.t_limit) * (2 * np.pi) * \
			self.time_embed_rand_weights[None, :]
		# Shape: B x (E / 2)
		time_embed = self.swish(
			torch.cat([
				torch.sin(time_embed_args), torch.cos(time_embed_args)
			], dim=1)
		)
		# Shape: B x E

		time_embed_out = self.time_dense_2(
			self.swish(self.time_dense_1(time_embed))
		)
		# Shape: B x E

		x = self.input_dense(xt) + time_embed_out  # Shape: B x E
		
		if task_inds is None:
			layer_to_iter = lambda layer_i: (
				enumerate([0] * self.num_tasks) if self.shared_layers[layer_i]
				else enumerate(range(self.num_tasks))
			)
		else:
			layer_to_iter = lambda layer_i: (
				enumerate([0] * len(task_inds)) if self.shared_layers[layer_i]
				else enumerate(task_inds)
			)

		for i in range(len(self.dense_layers)):
			if i == 0:
				dense_outs = [
					self.relu(self.dense_layers[i][l_i](x))
					for _, l_i in layer_to_iter(i)
				]
			elif i < len(self.dense_layers) - 1:
				dense_outs = [
					self.relu(self.dense_layers[i][l_i](dense_outs[o_i]))
					for o_i, l_i in layer_to_iter(i)
				]
			else:
				dense_outs = [
					self.dense_layers[i][l_i](dense_outs[o_i])
					for o_i, l_i in layer_to_iter(i)
				]

		out = torch.stack(dense_outs, dim=1)  # Shape: B x T x D
		return out
		
	def loss(self, pred_values, true_values, task_inds, weights=None):
		"""
		Computes the loss of the neural network.
		Arguments:
			`pred_values`: a B x T x D tensor of predictions from the network
			`true_values`: a B x D tensor of true values to predict
			`task_inds`: a B-tensor of indices (0 through T - 1) that determine
				which predicted values to compare to true values
			`weights`: if provided, a tensor broadcastable with B x D to weight
				the squared error by, prior to summing or averaging across
				dimensions
		Returns a scalar loss of mean-squared-error values, summed across the
		D dimension and averaged across the batch dimension.
		"""
		pred_values_subset = torch.stack([
			pred_values[i, task_inds[i]] for i in range(len(task_inds))
		])	# Shape: B x D
		
		# Compute loss as MSE
		squared_error = torch.square(true_values - pred_values_subset)
		if weights is not None:
			squared_error = squared_error / weights
			
		return torch.mean(torch.sum(
			squared_error,
			dim=tuple(range(1, len(squared_error.shape)))
		))


class LabelGuidedTabularNet(torch.nn.Module):

	def __init__(

		self, num_classes, input_dim, t_limit=1, hidden_dims=[256, 256, 256],
		time_embed_std=30, embed_size=256, label_embed_size=32
	):
		"""
		Initialize a time-dependent dense neural network for tabular data.
		Labels are included as part of the input.
		Arguments:
			`num_classes`: number of classification tasks possible, C
			`input_dim`: dimension of input data, D
			`t_limit`: maximum time horizon
			`hidden_dims`: dimensions of each hidden layer; this is of length
				one fewer than the number of layers
			`time_embed_std`: standard deviation of random weights to sample for
				time embeddings
			`embed_size`: size of the time embeddings and input embeddings
		"""
		super().__init__()
		
		assert embed_size % 2 == 0

		self.creation_args = locals()
		del self.creation_args["self"]
		del self.creation_args["__class__"]
		self.creation_args = sanitize_sacred_arguments(self.creation_args)
		
		self.num_classes = num_classes
		self.t_limit = t_limit

		# Map labels to embeddings
		self.label_embedder = torch.nn.Embedding(num_classes, label_embed_size)

		# Random embedding layer for time; the random weights are set at the
		# start and are not trainable
		self.time_embed_rand_weights = torch.nn.Parameter(
			torch.randn(embed_size // 2) * time_embed_std,
			requires_grad=False
		)
		
		# Initial dense layers to generate input embedding (always shared)
		self.time_dense_1 = torch.nn.Linear(embed_size, embed_size)
		self.time_dense_2 = torch.nn.Linear(embed_size, embed_size)
		self.input_dense = torch.nn.Linear(input_dim, embed_size)
	   
		# Dense layers that form the bulk of the network
		self.dense_layers = torch.nn.ModuleList()  # List of lists
		for i in range(len(hidden_dims) + 1):
			if i == 0:
				in_size, out_size = \
					embed_size + label_embed_size, hidden_dims[i]
			elif i < len(hidden_dims):
				in_size, out_size = hidden_dims[i - 1], hidden_dims[i]
			else:
				in_size, out_size = hidden_dims[i - 1], input_dim
			layer_list = []
			self.dense_layers.append(torch.nn.Linear(in_size, out_size))

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)
		self.relu = torch.nn.ReLU()

	def forward(self, xt, t, label):
		"""
		Forward pass of the network.
		Arguments:
			`xt`: B x D tensor containing the images to train on
			`t`: B-tensor containing the times to train the network for each
				input
			`label`: B-tensor containing class indices
		Returns a B x D tensor which consists of the prediction.
		"""
		# Get the time embeddings for `t`
		# We embed the time as cos((t/T) * (2pi) * z) and sin((t/T) * (2pi) * z)
		time_embed_args = (t[:, None] / self.t_limit) * (2 * np.pi) * \
			self.time_embed_rand_weights[None, :]
		# Shape: B x (E / 2)
		time_embed = self.swish(
			torch.cat([
				torch.sin(time_embed_args), torch.cos(time_embed_args)
			], dim=1)
		)
		# Shape: B x E

		time_embed_out = self.time_dense_2(
			self.swish(self.time_dense_1(time_embed))
		)
		# Shape: B x E

		# Get label embeddings
		label_embed = self.label_embedder(label)

		x = self.input_dense(xt) + time_embed_out  # Shape: B x E
				
		for i in range(len(self.dense_layers)):
			if i == 0:
				dense_out = self.relu(self.dense_layers[i](
					torch.cat([x, label_embed], dim=1)
				))
			elif i < len(self.dense_layers) - 1:
				dense_out = self.relu(self.dense_layers[i](dense_out))
			else:
				dense_out = self.dense_layers[i](dense_out)

		return dense_out
		
	def loss(self, pred_values, true_values, weights=None):
		"""
		Computes the loss of the neural network.
		Arguments:
			`pred_values`: a B x D tensor of predictions from the network
			`true_values`: a B x D tensor of true values to predict
			`weights`: if provided, a tensor broadcastable with B x D to weight
				the squared error by, prior to summing or averaging across
				dimensions
		Returns a scalar loss of mean-squared-error values, summed across the
		D dimension and averaged across the batch dimension.
		"""
		# Compute loss as MSE
		squared_error = torch.square(true_values - pred_values)
		if weights is not None:
			squared_error = squared_error / weights
			
		return torch.mean(torch.sum(
			squared_error,
			dim=tuple(range(1, len(squared_error.shape)))
		))
