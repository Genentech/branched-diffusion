import torch
import numpy as np
import tqdm
import os
import sacred
import model.util as util
import model.sdes as sdes
import model.graph_net as graph_net

MODEL_DIR = os.environ.get(
	"MODEL_DIR",
	"/gstore/data/resbioai/tsenga5/branched_diffusion/models/trained_models/misc"
)

train_ex = sacred.Experiment("train")

train_ex.observers.append(
	sacred.observers.FileStorageObserver.create(MODEL_DIR)
)

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


@train_ex.config
def config():
	# Number of training epochs
	num_epochs = 30

	# Learning rate
	learning_rate = 0.001


@train_ex.command
def train_model(
	model, sde, data_loader, model_type, class_mapper, num_epochs,
	learning_rate, _run, loss_weighting_type="empirical_norm",
	weight_func=None, t_limit=1
):
	"""
	Trains a diffusion model using the given instantiated model and SDE object.
	Arguments:
		`model`: an instantiated score model which takes in x, t and predicts
			score
		`sde`: an SDE object
		`data_loader`: a DataLoader object that yields batches of data as
			tensors in pairs: x, y; x is both the adjacency matrix and node
			features
		`model_type`: either "branched" or "labelguided"
		`class_mapper`: for "branched" model types, a function that takes in
			B-tensors of class and time and maps to a B-tensor of branch
			indices; for "labelguided" model types, a function that takes in
			B-tensors of class and maps to a B-tensor of class indices
		`num_epochs`: number of epochs to train for
		`learning_rate`: learning rate to use for training
		`loss_weighting_type`: method for weighting the loss; can be "ml" to
			weight by g^2, "expected_norm" to weight by the expected mean
			magnitude of the loss, "empirical_norm" to weight by the observed
			true norm, or None to do no weighting at all
		`weight_func`: if given, a function mapping a batch of inputs x0 to a
			broadcastable tensor of weights which will multiply into the loss
			for each predicted feature (in addition to any loss weighting
			specified by `loss_weighting_type`)
		`t_limit`: training will occur between time 0 and `t_limit`
	"""
	assert model_type in ("branched", "labelguided")
	
	run_num = _run._id
	output_dir = os.path.join(MODEL_DIR, str(run_num))
	
	model.train()
	torch.set_grad_enabled(True)
	optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch_num in range(num_epochs):
		batch_losses = []
		t_iter = tqdm.tqdm(data_loader)
		for x0, y in t_iter:
			x0 = x0.to(DEVICE).float()
			
			# NEW: Get node "flags" to mask out parts that overrun molecule boundaries
			adj_x0 = x0[:, :, :x0.shape[1]]
			node_flags = graph_net.node_flags(adj_x0)
			
			# Sample random times from 0 to 1
			t = (torch.rand(x0.shape[0]) * t_limit).to(DEVICE)
	
			# Run SDE forward to get xt and the true score at xt
			xt, true_score = sde.forward(x0, t)
			
			# NEW: Mask out overrunning parts of xt and true score
			adj_xt, x_xt = xt[:, :, :xt.shape[1]], xt[:, :, xt.shape[1]:]
			adj_xt = graph_net.mask_adjs(adj_xt, node_flags)
			x_xt = graph_net.mask_x(x_xt, node_flags)
			xt = torch.cat([adj_xt, x_xt], dim=2)
			adj_true_score, x_true_score = \
				true_score[:, :, :true_score.shape[1]], true_score[:, :, true_score.shape[1]:]
			adj_true_score = graph_net.mask_adjs(adj_true_score, node_flags)
			x_true_score = graph_net.mask_x(x_true_score, node_flags)
			true_score = torch.cat([adj_true_score, x_true_score], dim=2)
			
			# Get model-predicted score
			# NEW: include node flags so predicted score masks out overruns
			if model_type == "branched":
				pred_score = model(xt, t, node_flags)
			else:
				class_inds = class_mapper(y).long()
				pred_score = model(xt, t, node_flags, class_inds)
			
			# Get weighting factor
			if loss_weighting_type == "ml":
				loss_weight = 1 / sde.diff_coef_func(xt, t)
			elif loss_weighting_type == "expected_norm":
				loss_weight = sde._inflate_dims(sde.mean_score_mag(t))
			elif loss_weighting_type == "empirical_norm":
				loss_weight = sde._inflate_dims(torch.mean(
					torch.square(true_score), dim=tuple(range(1, len(x0.shape)))
				))
			elif loss_weighting_type is None:
				loss_weight = torch.ones_like(x0)

			if weight_func is not None:
				# Division here, as `loss_weight` itself is the divisor
				extra_weights = weight_func(x0)
				loss_weight = loss_weight / extra_weights

			# Compute loss
			if model_type == "branched":
				# Compute branch indices
				branch_inds = class_mapper(y, t)
				loss = model.loss(
					pred_score, true_score, branch_inds, loss_weight
				)
			else:
				loss = model.loss(pred_score, true_score, loss_weight)
			loss_val = loss.item()
			t_iter.set_description("Loss: %.2f" % loss_val)

			if not np.isfinite(loss_val):
				continue

			optim.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
			optim.step()
			
			batch_losses.append(loss_val)
		
		epoch_loss = np.mean(batch_losses)
		print("Epoch %d average Loss: %.2f" % (epoch_num + 1, epoch_loss))
		
		_run.log_scalar("train_epoch_loss", epoch_loss)
		_run.log_scalar("train_batch_losses", batch_losses)

		model_path = os.path.join(
			output_dir, "epoch_%d_ckpt.pth" % (epoch_num + 1)
		)
		link_path = os.path.join(output_dir, "last_ckpt.pth")

		# Save model
		util.save_model(model, model_path)

		# Create symlink to last epoch
		if os.path.islink(link_path):
			os.remove(link_path)
		os.symlink(os.path.basename(model_path), link_path)

@train_ex.command
def train_branched_model(
	model, sde, data_loader, class_time_to_branch_index, num_epochs,
	learning_rate, _run, loss_weighting_type="empirical_norm", t_limit=1
):
	"""
	Wrapper for `train_model`.
	"""
	train_model(
		model, sde, data_loader, "branched", class_time_to_branch_index,
		num_epochs, learning_rate, _run,
		loss_weighting_type=loss_weighting_type, t_limit=t_limit
	)

	
@train_ex.command
def train_label_guided_model(
	model, sde, data_loader, class_to_class_index, num_epochs, learning_rate,
	_run, loss_weighting_type="empirical_norm", t_limit=1
):
	"""
	Wrapper for `train_model`.
	"""
	train_model(
		model, sde, data_loader, "labelguided", class_to_class_index,
		num_epochs, learning_rate, _run,
		loss_weighting_type=loss_weighting_type, t_limit=t_limit
	)


class VarianceExplodingSDE(sdes.SDE):
	
	def __init__(self, sigma_min, sigma_max, input_shape, symmetrize=False):
		super().__init__(input_shape)
		
		self.sigma_min = torch.tensor(sigma_min).to(DEVICE)
		self.sigma_max = torch.tensor(sigma_max).to(DEVICE)
		self.symmetrize = symmetrize
		if symmetrize:
			assert len(input_shape) == 2 and input_shape[0] == input_shape[1]
		
	def drift_coef_func(self, xt, t):
		return torch.zeros_like(xt)
	
	def diff_coef_func(self, xt, t):
		return self._inflate_dims(
			self.sigma_min * torch.pow(self.sigma_max / self.sigma_min, t) *
			torch.sqrt(2 * torch.log(self.sigma_max / self.sigma_min))
		)

	def forward(self, x0, t, return_score=True):
		z = torch.normal(torch.zeros_like(x0), torch.ones_like(x0))
		if self.symmetrize:
			z = torch.triu(z, diagonal=1)
			z = z + torch.transpose(z, 1, 2)
		std = self.sigma_min * torch.pow(self.sigma_max / self.sigma_min, t)
		std = self._inflate_dims(std)
		xt = x0 + (std * z)
		if return_score:
			score = -z / std
			return xt, score
		else:
			return xt
		
	def mean_score_mag(self, t):
		variance = torch.square(self.sigma_min * torch.pow(self.sigma_max / self.sigma_min, t))
		return 1 / variance
		
	def sample_prior(self, num_sample, t):
		shape = torch.Size([num_sample]) + torch.Size(self.input_shape)
		z = torch.normal(torch.zeros(shape).to(DEVICE), torch.ones(shape).to(DEVICE))
		if self.symmetrize:
			z = torch.triu(z, diagonal=1)
			z = z + torch.transpose(z, 1, 2)
		std = self.sigma_min * torch.pow(self.sigma_max / self.sigma_min, t)
		return z * self._inflate_dims(std)

class VariancePreservingSDE(sdes.SDE):
	
	def __init__(self, beta_0, beta_1, input_shape, symmetrize=False):
		super().__init__(input_shape)
		
		self.beta_0 = torch.tensor(beta_0).to(DEVICE)
		self.delta_beta = torch.tensor(beta_1 - beta_0).to(DEVICE)
		self.symmetrize = symmetrize
		if symmetrize:
			assert len(input_shape) == 2 and input_shape[0] == input_shape[1]

	def _beta(self, t):
		return self.beta_0 + (self.delta_beta * t)
		
	def _beta_bar(self, t):
		return (self.beta_0 * t) + (0.5 * self.delta_beta * torch.square(t))
	
	def drift_coef_func(self, xt, t):
		return -0.5 * self._inflate_dims(self._beta(t)) * xt
	
	def diff_coef_func(self, xt, t):
		return self._inflate_dims(torch.sqrt(self._beta(t)))
	
	def forward(self, x0, t, return_score=True):
		z = torch.normal(
			torch.zeros_like(x0), torch.ones_like(x0), generator=self.rng
		)  # Shape: B x ...
		if self.symmetrize:
			z = torch.triu(z, diagonal=1)
			z = z + torch.transpose(z, 1, 2)
		
		mean = x0 * torch.exp(-0.5 * self._inflate_dims(self._beta_bar(t)))
		variance = 1 - torch.exp(-self._beta_bar(t))
		std = self._inflate_dims(torch.sqrt(variance))	# Shape: B x ...
		
		xt = mean + (std * z)
		
		if return_score:
			score = -z / std
			return xt, score
		else:
			return xt
		
	def mean_score_mag(self, t):
		variance = 1 - torch.exp(-self._beta_bar(t))
		return 1 / variance  # Shape: B
		
	def sample_prior(self, num_samples, t):
		# We will sample in the limit as t approaches infinity
		shape = torch.Size([num_samples]) + torch.Size(self.input_shape)
		z = torch.normal(
			torch.zeros(shape).to(DEVICE), torch.ones(shape).to(DEVICE),
			generator=self.rng
		)  # Shape: B x ...
		if self.symmetrize:
			z = torch.triu(z, diagonal=1)
			z = z + torch.transpose(z, 1, 2)
		return z

class AXJointSDE(sdes.SDE):
	
	def __init__(self, vp_beta_0, vp_beta_1, ve_sigma_min, ve_sigma_max, input_shape):
		super().__init__(input_shape)
		
		self.adj_shape = (input_shape[0], input_shape[0])
		self.x_shape = (input_shape[0], input_shape[1] - input_shape[0])
		
		self.adj_sde = VarianceExplodingSDE(ve_sigma_min, ve_sigma_max, self.adj_shape, symmetrize=True)
		self.x_sde = VariancePreservingSDE(vp_beta_0, vp_beta_1, self.x_shape)
		
	def forward(self, x0, t, return_score=True):
		adj_x0, x_x0 = x0[:, :, :x0.shape[1]], x0[:, :, x0.shape[1]:]
		if return_score:
			adj_xt, adj_score = self.adj_sde.forward(adj_x0, t, True)
			x_xt, x_score = self.x_sde.forward(x_x0, t, True)
			xt = torch.cat([adj_xt, x_xt], dim=2)
			score = torch.cat([adj_score, x_score], dim=2)
			return xt, score
		else:
			adj_xt = self.adj_sde.forward(adj_x0, t, False)
			x_xt = self.x_sde.forward(x_x0, t, False)
			return torch.cat([adj_xt, x_xt], dim=2)
		
	def mean_score_mag(self, t):
		adj_mag = self._inflate_dims(self.adj_sde.mean_score_mag(t))
		x_mag = self._inflate_dims(self.x_sde.mean_score_mag(t))
		adj_mag = torch.tile(adj_mag, (1,) + self.adj_shape)
		x_mag = torch.tile(x_mag, (1,) + self.x_shape)
		return torch.cat([adj_mag, x_mag], dim=2)
		
	def sample_prior(self, num_samples, t):
		adj_prior = self.adj_sde.sample_prior(num_samples, t)
		x_prior = self.x_sde.sample_prior(num_samples, t)
		return torch.cat([adj_prior, x_prior], dim=2)

def generate_continuous_samples(
	model, sde, class_to_sample, model_type, class_mapper, node_flags,
	num_samples=64, num_steps=500, t_start=0.001, t_limit=1,
	initial_samples=None, verbose=False
):
	"""
	Generates samples from a trained score model and SDE. This first generates a
	sample from the SDE's prior distribution a `t_limit`, then steps backward
	through time to generate new data points.
	Arguments:
		`model`: a trained score model which takes in x, t and predicts score
		`sde`: an SDE object
		`class_to_sample`: class to sample from (will be an argument in tensor
			form to `class_mapper`)
		`model_type`: either "branched" or "labelguided"
		`class_mapper`: for "branched" model types, a function that takes in
			B-tensors of class and time and maps to a B-tensor of branch
			indices; for "labelguided" model types, a function that takes in
			B-tensors of class and maps to a B-tensor of class indices
		`node_flags`: a set of binary flags that denote the size of each graph
		`num_samples`: number of objects to return
		`num_steps`: number of steps to take for Euler-Maruyama and
			predictor-corrector algorithms
		`t_start`: last time step to stop at (a smaller positive number than
			`t_limit`)
		`t_limit`: the time step to start generating at (a larger positive
			number than `t_start`)
		`initial_samples`: if given, this is a tensor which contains the samples
			to start from initially, to be used instead of sampling from the
			SDE's defined prior
		`verbose`: if True, print out progress bar and/or number of ODE
			evaluations
	Returns a tensor of size `num_samples` x ...
	"""
	assert model_type in ("branched", "labelguided")

	# First, sample from the prior distribution at some late time t
	if initial_samples is not None:
		xt = initial_samples
	else:
		t = (torch.ones(num_samples) * t_limit).to(DEVICE)
		xt = sde.sample_prior(num_samples, t)
		
	# NEW: mask out overrun
	adj_xt, x_xt = xt[:, :, :xt.shape[1]], xt[:, :, xt.shape[1]:]
	adj_xt = graph_net.mask_adjs(adj_xt, node_flags)
	x_xt = graph_net.mask_x(x_xt, node_flags)
	xt = torch.cat([adj_xt, x_xt], dim=2)

	if model_type == "branched":
		class_tens = torch.tensor([class_to_sample], device=DEVICE)
	else:
		class_tens = torch.tile(
			torch.tensor([class_to_sample], device=DEVICE), (num_samples,)
		)
	
	# Disable gradient computation in model
	torch.set_grad_enabled(False)
	
	time_steps = torch.linspace(t_limit, t_start, num_steps).to(DEVICE)
	# (descending order)
	step_size = time_steps[0] - time_steps[1]

	# Step backward through time starting at xt
	x = xt
	t_iter = tqdm.tqdm(time_steps) if verbose else time_steps
	for time_step in t_iter:
		t = torch.ones(num_samples).to(DEVICE) * time_step\

		# Take Langevin MCMC step
		# NEW: include node flags so predicted score masks out overruns
		if model_type == "branched":
			branch_index = class_mapper(class_tens, time_step[None])[0]
			score = model(x, t, node_flags, [branch_index])[:, 0]
		else:
			class_index = class_mapper(class_tens).long()
			score = model(x, t, node_flags, class_index)

		snr = 0.1
		score_norm = torch.mean(
			torch.norm(score.reshape(score.shape[0], -1), dim=-1)
		)
		alpha = snr * (
			torch.prod(torch.tensor(x.shape[1:])) / torch.square(score_norm)
		)

		z = torch.randn_like(x)
		# NEW: symmetrize noise and mask out overrun
		adj_z, x_z = z[:, :, :z.shape[1]], z[:, :, z.shape[1]:]
		adj_z = graph_net.mask_adjs(adj_z, node_flags)
		adj_z = torch.triu(adj_z, diagonal=1)
		adj_z = adj_z + torch.transpose(adj_z, 1, 2)
		x_z = graph_net.mask_x(x_z, node_flags)
		z = torch.cat([adj_z, x_z], dim=2)
		
		x = x + ((alpha / 2) * score) + \
			(torch.sqrt(alpha) * z)

		# Take SDE step
		f = sde.drift_coef_func(x, t)
		g = sde.diff_coef_func(x, t)
		dw = torch.randn_like(x)
		# NEW: symmetrize noise and mask out overrun
		adj_dw, x_dw = dw[:, :, :dw.shape[1]], dw[:, :, dw.shape[1]:]
		adj_dw = graph_net.mask_adjs(adj_dw, node_flags)
		adj_dw = torch.triu(adj_dw, diagonal=1)
		adj_dw = adj_dw + torch.transpose(adj_dw, 1, 2)
		x_dw = graph_net.mask_x(x_dw, node_flags)
		dw = torch.cat([adj_dw, x_dw], dim=2)

		# NEW: include node flags so predicted score masks out overruns
		if model_type == "branched":
			score = model(x, t, node_flags, [branch_index])[:, 0]
		else:
			score = model(x, t, node_flags, class_index)

		drift = (f - (torch.square(g) * score)) * step_size
		diff = g * torch.sqrt(step_size) * dw

		mean_x = x - drift	# Subtract because step size is really negative
		x = mean_x + diff
		
	return mean_x  # Last step: don't include the diffusion/randomized term


def generate_continuous_branched_samples(
	model, sde, class_to_sample, class_time_to_branch_index, node_flags,
	num_samples=64, num_steps=500, t_start=0.001, t_limit=1,
	initial_samples=None, verbose=False
):
	"""
	Wrapper for `generate_continuous_samples`.
	"""
	return generate_continuous_samples(
		model, sde, class_to_sample, "branched", class_time_to_branch_index,
		node_flags=node_flags, num_samples=num_samples, num_steps=num_steps,
		t_start=t_start, t_limit=t_limit, initial_samples=initial_samples,
		verbose=verbose
	)


def generate_continuous_label_guided_samples(
	model, sde, class_to_sample, class_to_class_index, node_flags,
	num_samples=64, num_steps=500, t_start=0.001, t_limit=1,
	initial_samples=None, verbose=False
):
	"""
	Wrapper for `generate_continuous_samples`.
	"""
	return generate_continuous_samples(
		model, sde, class_to_sample, "labelguided", class_to_class_index,
		node_flags=node_flags, num_samples=num_samples, num_steps=num_steps,
		t_start=t_start, t_limit=t_limit, initial_samples=initial_samples,
		verbose=verbose
	)
