import torch
import numpy as np
from model.util import sanitize_sacred_arguments

class MultitaskAutoencoderTimeConcat(torch.nn.Module):
	def __init__(
		self, num_tasks, input_dim, t_limit=1,
		enc_dims=[256, 128, 32], dec_dims=[128, 256],
		shared_layers=[True, True, True, False, False, False],
		time_embed_size=32, dropout=0.5
	):
		"""
		Initialize a time-dependent autoencoder for scRNA-seq data, where time
		embeddings are concatenated onto intermediate representations.
		Arguments:
			`num_tasks`: number of tasks to output, T
			`input_dim`: dimension of input data, D
			`t_limit`: maximum time horizon
			`enc_dims`: the number of channels in each encoding layer; the
				latent dimension is the last entry
			`dec_dims`: the number of channels in each decoding layer (given in
				reverse order of usage); note that the last decoding layer will
				map back to `input_dim`
			`shared_layers`: which layers of the network to share weights
				between tasks
			`time_embed_size`: size of the time embeddings
			`dropout`: droput probability after each layer
		"""
		super().__init__()

		assert len(shared_layers) == len(enc_dims) + len(dec_dims) + 1

		hidden_dims = enc_dims + dec_dims[::-1]

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
		
		# Layers that form the bulk of the network
		self.layers = torch.nn.ModuleList()  # List of lists of layer modules
		self.time_embedders = torch.nn.ModuleList()  # List of lists
		for i in range(len(shared_layers)):
			if i == 0:
				in_size, out_size = input_dim, hidden_dims[i]
			elif i < len(shared_layers) - 1:
				in_size, out_size = hidden_dims[i - 1], hidden_dims[i]
			else:
				in_size, out_size = hidden_dims[i - 1], input_dim
			
			layer_tasks = torch.nn.ModuleList()
			for _ in layer_to_iter(i):
				s = torch.nn.Sequential()

				if i < len(shared_layers) - 1:
					dense_in_size = in_size + time_embed_size
				else:
					dense_in_size = in_size
				s.append(torch.nn.Linear(dense_in_size, out_size))

				if i < len(shared_layers) - 1:
					s.append(torch.nn.BatchNorm1d(out_size))
					s.append(torch.nn.ReLU())
					if dropout > 0:
						s.append(torch.nn.Dropout(dropout))

				layer_tasks.append(s)

			self.layers.append(layer_tasks)
			
			if i < len(shared_layers) - 1:
				self.time_embedders.append(torch.nn.ModuleList([
					torch.nn.Linear(2, time_embed_size)
					for _ in layer_to_iter(i)
				]))

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)

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
		# We embed the time as cos((t/T) * (2pi)) and sin((t/T) * (pi / 2))
		time_embed_args = (t[:, None] / self.t_limit) * (np.pi / 2)
		# Shape: B x 1
		time_embed = self.swish(
			torch.cat([
				torch.sin(time_embed_args), torch.cos(time_embed_args)
			], dim=1)
		)
		# Shape: B x 2

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

		for i in range(len(self.layers)):
			if i == 0:
				time_embed_outs = [
					self.time_embedders[i][l_i](time_embed)
					for _, l_i in layer_to_iter(i)
				]
				layer_outs = [
					self.layers[i][l_i](torch.cat(
						[xt, time_embed_outs[o_i]], dim=1
					))
					for o_i, l_i in layer_to_iter(i)
				]
			elif i < len(self.layers) - 1:
				time_embed_outs = [
					self.time_embedders[i][l_i](time_embed)
					for _, l_i in layer_to_iter(i)
				]
				layer_outs = [
					self.layers[i][l_i](torch.cat(
						[layer_outs[o_i], time_embed_outs[o_i]], dim=1
					))
					for o_i, l_i in layer_to_iter(i)
				]
			else:
				layer_outs = [
					self.layers[i][l_i](layer_outs[o_i])
					for o_i, l_i in layer_to_iter(i)
				]

		out = torch.stack(layer_outs, dim=1)  # Shape: B x T x D
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


class MultitaskAutoencoderTimeAdd(torch.nn.Module):
	def __init__(
		self, num_tasks, input_dim, t_limit=1,
		enc_dims=[256, 128, 32], dec_dims=[128, 256],
		shared_layers=[True, True, True, False, False, False],
		time_embed_std=30, time_embed_size=32, dropout=0.5
	):
		"""
		Initialize a time-dependent autoencoder for scRNA-seq data, where time
		embeddings are added onto intermediate representations.
		Arguments:
			`num_tasks`: number of tasks to output, T
			`input_dim`: dimension of input data, D
			`t_limit`: maximum time horizon
			`enc_dims`: the number of channels in each encoding layer; the
				latent dimension is the last entry
			`dec_dims`: the number of channels in each decoding layer (given in
				reverse order of usage); note that the last decoding layer will
				map back to `input_dim`
			`shared_layers`: which layers of the network to share weights
				between tasks
			`time_embed_std`: standard deviation of random weights to sample for
				time embeddings
			`time_embed_size`: size of the time embeddings
			`dropout`: droput probability after each layer
		"""
		super().__init__()

		assert len(shared_layers) == len(enc_dims) + len(dec_dims) + 1
		assert time_embed_size % 2 == 0

		hidden_dims = enc_dims + dec_dims[::-1]

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
			torch.randn(time_embed_size // 2) * time_embed_std,
			requires_grad=False
		)
		
		# Layers that form the bulk of the network
		self.layers = torch.nn.ModuleList()  # List of lists of layer modules
		self.time_embedders = torch.nn.ModuleList()  # List of lists
		for i in range(len(shared_layers)):
			if i == 0:
				in_size, out_size = input_dim, hidden_dims[i]
			elif i < len(shared_layers) - 1:
				in_size, out_size = hidden_dims[i - 1], hidden_dims[i]
			else:
				in_size, out_size = hidden_dims[i - 1], input_dim
			
			layer_tasks = torch.nn.ModuleList()
			for _ in layer_to_iter(i):
				s = torch.nn.Sequential()
				s.append(torch.nn.Linear(in_size, out_size))

				if i < len(shared_layers) - 1:
					s.append(torch.nn.BatchNorm1d(out_size))
					s.append(torch.nn.ReLU())
					if dropout > 0:
						s.append(torch.nn.Dropout(dropout))

				layer_tasks.append(s)

			self.layers.append(layer_tasks)
			
			if i < len(shared_layers) - 1:
				self.time_embedders.append(torch.nn.ModuleList([
					torch.nn.Linear(time_embed_size, in_size)
					for _ in layer_to_iter(i)
				]))

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)

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
		# We had sampled a vector z from a zero-mean Gaussian of fixed variance
		# We embed the time as cos((t/T) * (pi / 2) * z) and
		# sin((t/T) * (pi / 2) * z)
		time_embed_args = (t[:, None] / self.t_limit) * \
			self.time_embed_rand_weights[None, :] * (np.pi / 2)
		# Shape: B x (E / 2)

		time_embed = self.swish(torch.cat([
			torch.sin(time_embed_args), torch.cos(time_embed_args)
		], dim=1))
		# Shape: B x E

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

		for i in range(len(self.layers)):
			if i == 0:
				layer_outs = [
					self.layers[i][l_i](
						xt + self.time_embedders[i][l_i](time_embed)
					)
					for o_i, l_i in layer_to_iter(i)
				]
			elif i < len(self.layers) - 1:
				layer_outs = [
					self.layers[i][l_i](
						layer_outs[o_i] +
						self.time_embedders[i][l_i](time_embed)
					)
					for o_i, l_i in layer_to_iter(i)
				]
			else:
				layer_outs = [
					self.layers[i][l_i](layer_outs[o_i])
					for o_i, l_i in layer_to_iter(i)
				]

		out = torch.stack(layer_outs, dim=1)  # Shape: B x T x D
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


class MultitaskResNet(torch.nn.Module):
	def __init__(
		self, num_tasks, input_dim, t_limit=1, hidden_dim=8192,
		shared_layers=[True, True, True, False, False],
		time_embed_std=30, time_embed_size=32
	):
		"""
		Initialize a time-dependent residual network for scRNA-seq data, where
		time embeddings are added onto intermediate representations.
		Arguments:
			`num_tasks`: number of tasks to output, T
			`input_dim`: dimension of input data, D
			`t_limit`: maximum time horizon
			`hidden_dim`: dimensions of each hidden layer
			`shared_layers`: which layers of the network to share weights
				between tasks; this defines the number of layers
			`time_embed_std`: standard deviation of random weights to sample for
				time embeddings
			`time_embed_size`: size of the time embeddings
		"""
		super().__init__()

		assert time_embed_size % 2 == 0

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
			torch.randn(time_embed_size // 2) * time_embed_std,
			requires_grad=False
		)
		
		# Layers that form the bulk of the network
		self.layers = torch.nn.ModuleList()  # List of lists of layer modules
		self.time_embedders = torch.nn.ModuleList()  # List of lists
		for i in range(len(shared_layers)):
			if i == 0:
				in_size, out_size = input_dim, hidden_dim
			elif i < len(shared_layers) - 1:
				in_size, out_size = hidden_dim, hidden_dim
			else:
				in_size, out_size = hidden_dim, input_dim
			
			layer_tasks = torch.nn.ModuleList()
			for _ in layer_to_iter(i):
				s = torch.nn.Sequential()
				s.append(torch.nn.Linear(in_size, out_size))

				if i < len(shared_layers) - 1:
					s.append(torch.nn.BatchNorm1d(out_size))
					s.append(torch.nn.ReLU())

				layer_tasks.append(s)

			self.layers.append(layer_tasks)
			
			if i < len(shared_layers) - 1:
				self.time_embedders.append(torch.nn.ModuleList([
					torch.nn.Linear(time_embed_size, in_size)
					for _ in layer_to_iter(i)
				]))

		# Last layer that applies a linear function to each output
		self.last_linear = torch.nn.Conv1d(
			input_dim, input_dim, num_tasks, groups=input_dim
		)

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)

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
		# We had sampled a vector z from a zero-mean Gaussian of fixed variance
		# We embed the time as cos((t/T) * (pi / 2) * z) and
		# sin((t/T) * (pi / 2) * z)
		time_embed_args = (t[:, None] / self.t_limit) * \
			self.time_embed_rand_weights[None, :] * (np.pi / 2)
		# Shape: B x (E / 2)

		time_embed = self.swish(torch.cat([
			torch.sin(time_embed_args), torch.cos(time_embed_args)
		], dim=1))
		# Shape: B x E

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

		for i in range(len(self.layers)):
			if i == 0:
				running_sum = [
					self.layers[i][l_i](
						xt + self.time_embedders[i][l_i](time_embed)
					)
					for o_i, l_i in layer_to_iter(i)
				]
			elif i < len(self.layers) - 1:
				layer_outs = [
					self.layers[i][l_i](
						running_sum[o_i] +
						self.time_embedders[i][l_i](time_embed)
					)
					for o_i, l_i in layer_to_iter(i)
				]
				running_sum = [
					running_sum[i] + layer_outs[i]
					for i in range(len(running_sum))
				]
			else:
				layer_outs = [
					self.layers[i][l_i](running_sum[o_i])
					for o_i, l_i in layer_to_iter(i)
				]

		body_out = torch.stack(layer_outs, dim=1)  # Shape: B x T x D

		# For each output scalar, apply a distinct linear function
		out = torch.permute(
			self.last_linear(torch.permute(body_out, (0, 2, 1))),
			(0, 2, 1)
		)
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
