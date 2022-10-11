import torch
import numpy as np
from model.util import sanitize_sacred_arguments

class MultitaskMNISTUNetTimeConcat(torch.nn.Module):

	def __init__(
		self, num_tasks, t_limit=1,
		enc_dims=[32, 64, 128, 256], dec_dims=[32, 64, 128],
		shared_layers=[True, True, True, True, False, False, False, False],
		time_embed_size=32, data_channels=1
	):
		"""
		Initialize a time-dependent U-net for MNIST, where time embeddings are
		concatenated to image representations.
		Arguments:
			`num_tasks`: number of tasks to output, T
			`t_limit`: maximum time horizon
			`enc_dims`: the number of channels in each encoding layer
			`dec_dims`: the number of channels in each decoding layer (given in
				reverse order of usage)
			`shared_layers`: which layers of the UNet to share weights between
				tasks
			`time_embed_size`: size of the time embeddings
			`data_channels`: number of channels in input image
		"""
		super().__init__()

		assert len(enc_dims) == 4
		assert len(dec_dims) == 3
		assert len(shared_layers) == 8

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
		
		# Encoders: receptive field increases and depth increases
		self.conv_e1_tasks = torch.nn.ModuleList()
		self.time_dense_e1_tasks = torch.nn.ModuleList()
		self.norm_e1_tasks = torch.nn.ModuleList()
		self.conv_e2_tasks = torch.nn.ModuleList()
		self.time_dense_e2_tasks = torch.nn.ModuleList()
		self.norm_e2_tasks = torch.nn.ModuleList()
		self.conv_e3_tasks = torch.nn.ModuleList()
		self.time_dense_e3_tasks = torch.nn.ModuleList()
		self.norm_e3_tasks = torch.nn.ModuleList()
		self.conv_e4_tasks = torch.nn.ModuleList()
		self.time_dense_e4_tasks = torch.nn.ModuleList()
		self.norm_e4_tasks = torch.nn.ModuleList()
		for _ in layer_to_iter(0):
			self.conv_e1_tasks.append(
				torch.nn.Conv2d(
					data_channels + time_embed_size, enc_dims[0], kernel_size=3,
					stride=1, bias=False
				)
			)
			self.time_dense_e1_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_e1_tasks.append(
				torch.nn.GroupNorm(4, num_channels=enc_dims[0])
			)
		for _ in layer_to_iter(1):
			self.conv_e2_tasks.append(
				torch.nn.Conv2d(
					enc_dims[0] + time_embed_size, enc_dims[1], kernel_size=3,
					stride=2, bias=False
				)
			)
			self.time_dense_e2_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_e2_tasks.append(
				torch.nn.GroupNorm(32, num_channels=enc_dims[1])
			)
		for _ in layer_to_iter(2):
			self.conv_e3_tasks.append(
				torch.nn.Conv2d(
					enc_dims[1] + time_embed_size, enc_dims[2], kernel_size=3,
					stride=2, bias=False
				)
			)
			self.time_dense_e3_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_e3_tasks.append(
				torch.nn.GroupNorm(32, num_channels=enc_dims[2])
			)
		for _ in layer_to_iter(3):
			self.conv_e4_tasks.append(
				torch.nn.Conv2d(
					enc_dims[2] + time_embed_size, enc_dims[3], kernel_size=3,
					stride=2, bias=False
				)
			)
			self.time_dense_e4_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_e4_tasks.append(
				torch.nn.GroupNorm(32, num_channels=enc_dims[3])
			)

		# Decoders: depth decreases		   
		self.conv_d4_tasks = torch.nn.ModuleList()
		self.time_dense_d4_tasks = torch.nn.ModuleList()
		self.norm_d4_tasks = torch.nn.ModuleList()
		self.conv_d3_tasks = torch.nn.ModuleList()
		self.time_dense_d3_tasks = torch.nn.ModuleList()
		self.norm_d3_tasks = torch.nn.ModuleList()
		self.conv_d2_tasks = torch.nn.ModuleList()
		self.time_dense_d2_tasks = torch.nn.ModuleList()
		self.norm_d2_tasks = torch.nn.ModuleList()
		self.conv_d1_tasks = torch.nn.ModuleList()
		for _ in layer_to_iter(4):
			self.conv_d4_tasks.append(
				torch.nn.ConvTranspose2d(
					enc_dims[3] + time_embed_size, dec_dims[2], 3, stride=2,
					bias=False
				)
			)
			self.time_dense_d4_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_d4_tasks.append(
				torch.nn.GroupNorm(32, num_channels=dec_dims[2])
			)
		for _ in layer_to_iter(5):
			self.conv_d3_tasks.append(
				torch.nn.ConvTranspose2d(
					dec_dims[2] + enc_dims[2] + time_embed_size, dec_dims[1], 3,
					stride=2, output_padding=1, bias=False
				)
			)
			self.time_dense_d3_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_d3_tasks.append(
				torch.nn.GroupNorm(32, num_channels=dec_dims[1])
			)
		for _ in layer_to_iter(6):
			self.conv_d2_tasks.append(
				torch.nn.ConvTranspose2d(
					dec_dims[1] + enc_dims[1] + time_embed_size, dec_dims[0], 3,
					stride=2, output_padding=1, bias=False
				)
			)
			self.time_dense_d2_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_d2_tasks.append(
				torch.nn.GroupNorm(32, num_channels=dec_dims[0])
			)
		for _ in layer_to_iter(7):
			self.conv_d1_tasks.append(
				torch.nn.ConvTranspose2d(
					dec_dims[0] + enc_dims[0], data_channels, 3, stride=1,
					bias=True
				)
			)

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)

	def forward(self, xt, t):
		"""
		Forward pass of the network.
		Arguments:
			`xt`: B x 1 x H x W tensor containing the images to train on
			`t`: B-tensor containing the times to train the network for each
				image
		Returns a B x T x 1 x H x W tensor which consists of the prediction.
		"""
		# Get the time embeddings for `t`
		# We embed the time as cos((t/T) * (2pi)) and sin((t/T) * (2pi))
		time_embed_args = (t[:, None] / self.t_limit) * (2 * np.pi)
		# Shape: B x 1
		time_embed = self.swish(
			torch.cat([
				torch.sin(time_embed_args), torch.cos(time_embed_args)
			], dim=1)
		)
		# Shape: B x 2

		layer_to_iter = lambda layer_i: (
			[0] * self.num_tasks if self.shared_layers[layer_i] else
			range(self.num_tasks)
		)
		
		# Encoding
		enc_1_outs = [
			self.swish(self.norm_e1_tasks[i](self.conv_e1_tasks[i](
				torch.cat([
					xt,
					torch.tile(
						self.time_dense_e1_tasks[i](
							time_embed
						)[:, :, None, None],
						(1, 1) + xt.shape[2:]
					)
				], dim=1)
			))) for i in layer_to_iter(0)
		]
		enc_2_outs = [
			self.swish(self.norm_e2_tasks[i](self.conv_e2_tasks[i](
				torch.cat([
					enc_1_outs[i],
					torch.tile(
						self.time_dense_e2_tasks[i](
							time_embed
						)[:, :, None, None],
						(1, 1) + enc_1_outs[i].shape[2:]
					)
				], dim=1)
			))) for i in layer_to_iter(1)
		]
		enc_3_outs = [
			self.swish(self.norm_e3_tasks[i](self.conv_e3_tasks[i](
				torch.cat([
					enc_2_outs[i],
					torch.tile(
						self.time_dense_e3_tasks[i](
							time_embed
						)[:, :, None, None],
						(1, 1) + enc_2_outs[i].shape[2:]
					)
				], dim=1)
			))) for i in layer_to_iter(2)
		]
		enc_4_outs = [
			self.swish(self.norm_e4_tasks[i](self.conv_e4_tasks[i](
				torch.cat([
					enc_3_outs[i],
					torch.tile(
						self.time_dense_e4_tasks[i](
							time_embed
						)[:, :, None, None],
						(1, 1) + enc_3_outs[i].shape[2:]
					)
				], dim=1)
			))) for i in layer_to_iter(3)
		]
		
		# Decoding
		dec_4_outs = [
			self.swish(self.norm_d4_tasks[i](self.conv_d4_tasks[i](
				torch.cat([
					enc_4_outs[i],
					torch.tile(
						self.time_dense_d4_tasks[i](
							time_embed
						)[:, :, None, None],
						(1, 1) + enc_4_outs[i].shape[2:]
					)
				], dim=1)
			))) for i in layer_to_iter(4)
		]
		dec_3_outs = [
			self.swish(self.norm_d3_tasks[i](self.conv_d3_tasks[i](
				torch.cat([
					dec_4_outs[i], enc_3_outs[i],
					torch.tile(
						self.time_dense_d3_tasks[i](
							time_embed
						)[:, :, None, None],
						(1, 1) + dec_4_outs[i].shape[2:]
					)
				], dim=1)
			))) for i in layer_to_iter(5)
		]
		dec_2_outs = [
			self.swish(self.norm_d2_tasks[i](self.conv_d2_tasks[i](
				torch.cat([
					dec_3_outs[i], enc_2_outs[i],
					torch.tile(
						self.time_dense_d2_tasks[i](
							time_embed
						)[:, :, None, None],
						(1, 1) + dec_3_outs[i].shape[2:]
					)
				], dim=1)
			))) for i in layer_to_iter(6)
		]
		dec_1_out = [
			self.conv_d1_tasks[i](
				torch.cat([dec_2_outs[i], enc_1_outs[i]], dim=1)
			) for i in layer_to_iter(7)
		]
		dec_1_out = torch.stack(dec_1_out, dim=1)  # Shape: B x T x 1 x H x W
		
		return dec_1_out
	
	def loss(self, pred_values, true_values, task_inds, weights=None):
		"""
		Computes the loss of the neural network.
		Arguments:
			`pred_values`: a B x T x 1 x H x W tensor of predictions from the
				network
			`true_values`: a B x 1 x H x W tensor of true values to predict
			`task_inds`: a B-tensor of indices (0 through T - 1) that determine
				which predicted values to compare to true values
			`weights`: if provided, a tensor broadcastable with B x 1 x H x W to
				weight the squared error by, prior to summing or averaging
				across dimensions
		Returns a scalar loss of mean-squared-error values, summed across the
		1 x H x W dimensions and averaged across the batch dimension.
		"""
		pred_values_subset = torch.stack([
			pred_values[i, task_inds[i]] for i in range(len(task_inds))
		])	# Shape: B x 1 x H x W
		
		# Compute loss as MSE
		squared_error = torch.square(true_values - pred_values_subset)
		if weights is not None:
			squared_error = squared_error / weights
			
		return torch.mean(torch.sum(
			squared_error,
			dim=tuple(range(1, len(squared_error.shape)))
		))


class MultitaskMNISTUNetTimeAdd(torch.nn.Module):

	def __init__(
		self, num_tasks, t_limit=1,
		enc_dims=[32, 64, 128, 256], dec_dims=[32, 64, 128],
		shared_layers=[True, True, True, True, False, False, False, False],
		time_embed_size=32, time_embed_std=30, use_time_embed_dense=False,
		data_channels=1
	):
		"""
		Initialize a time-dependent U-net for MNIST, where time embeddings are
		concatenated to image representations.
		Arguments:
			`num_tasks`: number of tasks to output, T
			`t_limit`: maximum time horizon
			`enc_dims`: the number of channels in each encoding layer
			`dec_dims`: the number of channels in each decoding layer (given in
				reverse order of usage)
			`shared_layers`: which layers of the UNet to share weights between
				tasks
			`time_embed_size`: size of the time embeddings
			`time_embed_std`: standard deviation of random weights to sample for
				time embeddings
			`use_time_embed_dense`: if True, have a dense layer on top of time
				embeddings initially
			`data_channels`: number of channels in input image
		"""
		super().__init__()

		assert len(enc_dims) == 4
		assert len(dec_dims) == 3
		assert len(shared_layers) == 8
		assert time_embed_size % 2 == 0

		self.creation_args = locals()
		del self.creation_args["self"]
		del self.creation_args["__class__"]
		self.creation_args = sanitize_sacred_arguments(self.creation_args)
		
		self.num_tasks = num_tasks
		self.t_limit = t_limit
		self.shared_layers = shared_layers
		self.use_time_embed_dense = use_time_embed_dense

		layer_to_iter = lambda layer_i: range(
			1 if shared_layers[layer_i] else num_tasks
		)

		# Random embedding layer for time; the random weights are set at the
		# start and are not trainable
		self.time_embed_rand_weights = torch.nn.Parameter(
			torch.randn(time_embed_size // 2) * time_embed_std,
			requires_grad=False
		)
		if use_time_embed_dense:
			self.time_embed_dense = torch.nn.Linear(
				time_embed_size, time_embed_size
			)

		# Encoders: receptive field increases and depth increases
		self.conv_e1_tasks = torch.nn.ModuleList()
		self.time_dense_e1_tasks = torch.nn.ModuleList()
		self.norm_e1_tasks = torch.nn.ModuleList()
		self.conv_e2_tasks = torch.nn.ModuleList()
		self.time_dense_e2_tasks = torch.nn.ModuleList()
		self.norm_e2_tasks = torch.nn.ModuleList()
		self.conv_e3_tasks = torch.nn.ModuleList()
		self.time_dense_e3_tasks = torch.nn.ModuleList()
		self.norm_e3_tasks = torch.nn.ModuleList()
		self.conv_e4_tasks = torch.nn.ModuleList()
		self.time_dense_e4_tasks = torch.nn.ModuleList()
		self.norm_e4_tasks = torch.nn.ModuleList()
		for _ in layer_to_iter(0):
			self.conv_e1_tasks.append(
				torch.nn.Conv2d(
					data_channels + time_embed_size, enc_dims[0], kernel_size=3,
					stride=1, bias=False
				)
			)
			self.time_dense_e1_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_e1_tasks.append(
				torch.nn.GroupNorm(4, num_channels=enc_dims[0])
			)
		for _ in layer_to_iter(1):
			self.conv_e2_tasks.append(
				torch.nn.Conv2d(
					enc_dims[0] + time_embed_size, enc_dims[1], kernel_size=3,
					stride=2, bias=False
				)
			)
			self.time_dense_e2_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_e2_tasks.append(
				torch.nn.GroupNorm(32, num_channels=enc_dims[1])
			)
		for _ in layer_to_iter(2):
			self.conv_e3_tasks.append(
				torch.nn.Conv2d(
					enc_dims[1] + time_embed_size, enc_dims[2], kernel_size=3,
					stride=2, bias=False
				)
			)
			self.time_dense_e3_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_e3_tasks.append(
				torch.nn.GroupNorm(32, num_channels=enc_dims[2])
			)
		for _ in layer_to_iter(3):
			self.conv_e4_tasks.append(
				torch.nn.Conv2d(
					enc_dims[2] + time_embed_size, enc_dims[3], kernel_size=3,
					stride=2, bias=False
				)
			)
			self.time_dense_e4_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_e4_tasks.append(
				torch.nn.GroupNorm(32, num_channels=enc_dims[3])
			)

		# Decoders: depth decreases		   
		self.conv_d4_tasks = torch.nn.ModuleList()
		self.time_dense_d4_tasks = torch.nn.ModuleList()
		self.norm_d4_tasks = torch.nn.ModuleList()
		self.conv_d3_tasks = torch.nn.ModuleList()
		self.time_dense_d3_tasks = torch.nn.ModuleList()
		self.norm_d3_tasks = torch.nn.ModuleList()
		self.conv_d2_tasks = torch.nn.ModuleList()
		self.time_dense_d2_tasks = torch.nn.ModuleList()
		self.norm_d2_tasks = torch.nn.ModuleList()
		self.conv_d1_tasks = torch.nn.ModuleList()
		for _ in layer_to_iter(4):
			self.conv_d4_tasks.append(
				torch.nn.ConvTranspose2d(
					enc_dims[3] + time_embed_size, dec_dims[2], 3, stride=2,
					bias=False
				)
			)
			self.time_dense_d4_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_d4_tasks.append(
				torch.nn.GroupNorm(32, num_channels=dec_dims[2])
			)
		for _ in layer_to_iter(5):
			self.conv_d3_tasks.append(
				torch.nn.ConvTranspose2d(
					dec_dims[2] + enc_dims[2] + time_embed_size, dec_dims[1], 3,
					stride=2, output_padding=1, bias=False
				)
			)
			self.time_dense_d3_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_d3_tasks.append(
				torch.nn.GroupNorm(32, num_channels=dec_dims[1])
			)
		for _ in layer_to_iter(6):
			self.conv_d2_tasks.append(
				torch.nn.ConvTranspose2d(
					dec_dims[1] + enc_dims[1] + time_embed_size, dec_dims[0], 3,
					stride=2, output_padding=1, bias=False
				)
			)
			self.time_dense_d2_tasks.append(torch.nn.Linear(2, time_embed_size))
			self.norm_d2_tasks.append(
				torch.nn.GroupNorm(32, num_channels=dec_dims[0])
			)
		for _ in layer_to_iter(7):
			self.conv_d1_tasks.append(
				torch.nn.ConvTranspose2d(
					dec_dims[0] + enc_dims[0], data_channels, 3, stride=1,
					bias=True
				)
			)

		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)

	def forward(self, xt, t):
		"""
		Forward pass of the network.
		Arguments:
			`xt`: B x 1 x H x W tensor containing the images to train on
			`t`: B-tensor containing the times to train the network for each
				image
		Returns a B x T x 1 x H x W tensor which consists of the prediction.
		"""
		# Get the time embeddings for `t`
		# We had sampled a vector z from a zero-mean Gaussian of fixed variance
		# We embed the time as cos((t/T) * (2pi) * z) and sin((t/T) * (2pi) * z)
		time_embed_args = (t[:, None] / self.t_limit) * \
			self.time_embed_rand_weights[None, :] * (2 * np.pi)
		# Shape: B x (E / 2)

		time_embed = torch.cat([
			torch.sin(time_embed_args), torch.cos(time_embed_args)
		], dim=1)
		if self.use_time_embed_dense:
			time_embed = self.swish(self.time_embed_dense(time_embed))
		else:
			time_embed = self.swish(time_embed)
		# Shape: B x E

		layer_to_iter = lambda layer_i: (
			[0] * self.num_tasks if self.shared_layers[layer_i] else
			range(self.num_tasks)
		)
		
		# Encoding
		enc_1_outs = [
			self.swish(self.norm_e1_tasks[i](
				self.conv_e1_tasks[i](xt) +
				self.time_dense_e1_tasks[i](time_embed)[:, :, None, None]
			)) for i in layer_to_iter(0)
		]
		enc_2_outs = [
			self.swish(self.norm_e2_tasks[i](
				self.conv_e2_tasks[i](enc_1_outs[i]) +
				self.time_dense_e2_tasks[i](time_embed)[:, :, None, None]
			)) for i in layer_to_iter(1)
		]
		enc_3_outs = [
			self.swish(self.norm_e3_tasks[i](
				self.conv_e3_tasks[i](enc_2_outs[i]) +
				self.time_dense_e3_tasks[i](time_embed)[:, :, None, None]
			)) for i in layer_to_iter(2)
		]
		enc_4_outs = [
			self.swish(self.norm_e4_tasks[i](
				self.conv_e4_tasks[i](enc_3_outs[i]) +
				self.time_dense_e4_tasks[i](time_embed)[:, :, None, None]
			)) for i in layer_to_iter(3)
		]
		
		# Decoding
		dec_4_outs = [
			self.swish(self.norm_d4_tasks[i](
				self.conv_d4_tasks[i](enc_4_outs[i]) +
				self.time_dense_d4_tasks[i](time_embed)[:, :, None, None]
			)) for i in layer_to_iter(4)
		]
		dec_3_outs = [
			self.swish(self.norm_d3_tasks[i](
				self.conv_d3_tasks[i](
					torch.cat([dec_4_outs[i], enc_3_outs[i]], dim=1)
				) +
				self.time_dense_d3_tasks[i](time_embed)[:, :, None, None]
			)) for i in layer_to_iter(5)
		]
		dec_2_outs = [
			self.swish(self.norm_d2_tasks[i](
				self.conv_d2_tasks[i](
					torch.cat([dec_3_outs[i], enc_2_outs[i]], dim=1)
				) +
				self.time_dense_d2_tasks[i](time_embed)[:, :, None, None]
			)) for i in layer_to_iter(6)
		]
		dec_1_out = [
			self.conv_d1_tasks[i](
				torch.cat([dec_2_outs[i], enc_1_outs[i]], dim=1)
			)
			for i in layer_to_iter(7)
		]
		dec_1_out = torch.stack(dec_1_out, dim=1)  # Shape: B x T x 1 x H x W
		
		return dec_1_out

	def loss(self, pred_values, true_values, task_inds, weights=None):
		"""
		Computes the loss of the neural network.
		Arguments:
			`pred_values`: a B x T x 1 x H x W tensor of predictions from the
				network
			`true_values`: a B x 1 x H x W tensor of true values to predict
			`task_inds`: a B-tensor of indices (0 through T - 1) that determine
				which predicted values to compare to true values
			`weights`: if provided, a tensor broadcastable with B x 1 x H x W to
				weight the squared error by, prior to summing or averaging
				across dimensions
		Returns a scalar loss of mean-squared-error values, summed across the
		1 x H x W dimensions and averaged across the batch dimension.
		"""
		pred_values_subset = torch.stack([
			pred_values[i, task_inds[i]] for i in range(len(task_inds))
		])	# Shape: B x 1 x H x W
		
		# Compute loss as MSE
		squared_error = torch.square(true_values - pred_values_subset)
		if weights is not None:
			squared_error = squared_error / weights
			
		return torch.mean(torch.sum(
			squared_error,
			dim=tuple(range(1, len(squared_error.shape)))
		))
