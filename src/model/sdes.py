import torch

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


class SDE:
	# Base class for SDEs
	def __init__(self, input_shape, seed=None):
		"""
		Arguments:
			`input_shape`: a tuple of ints which is the shape of input tensors
				x; does not include batch dimension
			`seed`: random seed for sampling and running the SDE
		"""
		self.input_shape = input_shape
		self.rng = torch.Generator(device=DEVICE)
		if seed:
			self.rng.manual_seed(seed)
			
	def _inflate_dims(self, v):
		"""
		Given a tensor vector `v`, appends dimensions of size 1 so that it has
		the same number of dimensions as `self.input_shape`. For example, if
		`self.input_shape` is (3, 50, 50), then this function turns `v` from a
		B-tensor to a B x 1 x 1 x 1 tensor. This is useful for combining the
		tensor with things shaped like the input later.
		Arguments:
			`v`: a B-tensor
		Returns a B x `self.input_shape` tensor.
		"""
		return v[(slice(None),) + ((None,) * len(self.input_shape))]
	
	def drift_coef_func(self, xt, t):
		"""
		Definition of drift coefficient f(xt, t).
		Arguments:
			`xt`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE for each input
		Returns a B x `self.input_shape` tensor.
		"""
		return torch.zeros_like(xt)
	
	def diff_coef_func(self, xt, t):
		"""
		Definition of diffusion coefficient g(t).
		Arguments:
			`xt`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE
		Returns a B x `self.input_shape` tensor.
		"""
		return torch.zeros_like(xt)
	
	def forward(self, x0, t, return_score=True):
		"""
		Runs the SDE forward given starting point `x0` and a time `t`.
		Optionally returns the score: the gradient (with respect to x) of the
		log conditional probability, pt(xt | x0).
		Arguments:
			`x0`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE for each input
		Returns a B x `self.input_shape` tensor to represent xt. If
		`return_score` is True, then also returns a B x `self.input_shape`
		tensor which is the gradient of the log conditional probability (with
		respect to `xt`).
		"""
		if return_score:
			return torch.zeros_like(x0), torch.zeros_like(x0)
		else:
			return torch.zeros_like(x0)
		
	def mean_score_mag(self, t):
		"""
		Returns the average magnitude (squared L2 norm) of the score (averaged
		over sampling and data distribution), divided by the size of `x`. That
		is, this should be ||score||_2^2 / d, where d is the size of `x`.
		Arguments:
			`t`: a B-tensor containing the time in the SDE
		Returns a B-tensor containing the expected magnitude of the score
		function at each time `t`.
		"""
		return torch.ones_like(t)
		
	def sample_prior(self, num_samples, t):
		"""
		Samples from the prior distribution specified by the SDE at time `t`.
		Arguments:
			`num_samples`: B, the number of samples to return
			`t`: a B-tensor containing the times in the SDE to sample for
		Returns a B x `self.input_shape` tensor for the `xt` values that are
		sampled.
		"""
		return torch.zeros(torch.Size([num_samples] + list(self.input_shape)))
	
	def __str__(self):
		return "Base SDE"


class VarianceExplodingSDE(SDE):
	
	def __init__(self, sigma, input_shape, seed=None):
		"""
		Arguments:
			`sigma`: the sigma in dx = sigma^t dw
			`input_shape`: a tuple of ints which is the shape of input
				tensors x; does not include batch dimension
			`seed`: random seed for sampling and running the SDE
		"""
		super().__init__(input_shape, seed)
		
		self.sigma = torch.tensor(sigma).to(DEVICE)
		self.string = "Variance Exploding SDE (beta(t) = %.2f^t)" % sigma
	
	def drift_coef_func(self, xt, t):
		"""
		Definition of drift coefficient f(xt, t).
		Arguments:
			`xt`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE for each input
		Returns a B x `self.input_shape` tensor.
		"""
		return torch.zeros_like(xt)
	
	def diff_coef_func(self, xt, t):
		"""
		Definition of diffusion coefficient g(t).
		Arguments:
			`xt`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE
		Returns a B x `self.input_shape` tensor.
		"""
		return self._inflate_dims(torch.pow(self.sigma, t))
	
	def forward(self, x0, t, return_score=True):
		"""
		Runs the SDE forward given starting point `x0` and a time `t`.
		Optionally returns the score: the gradient (with respect to x) of the
		log conditional probability, pt(xt | x0).
		Arguments:
			`x0`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE for each input
		Returns a B x `self.input_shape` tensor to represent xt. If
		`return_score` is True, then also returns a B x `self.input_shape`
		tensor which is the gradient of the log conditional probability (with
		respect to `xt`).
		"""
		z = torch.normal(
			torch.zeros_like(x0), torch.ones_like(x0), generator=self.rng
		)  # Shape: B x ...
		
		variance = (torch.pow(self.sigma, 2 * t) - 1) / \
			(2 * torch.log(self.sigma))
		std = self._inflate_dims(torch.sqrt(variance))	# Shape: B x ...		
		xt = x0 + (std * z)
		
		if return_score:
			score = -z / std
			return xt, score
		else:
			return xt

	def mean_score_mag(self, t):
		"""
		Returns the average magnitude (squared L2 norm) of the score (averaged
		over sampling and data distribution), divided by the size of `x`. That
		is, this should be ||score||_2^2 / d, where d is the size of `x`.
		Arguments:
			`t`: a B-tensor containing the time in the SDE
		Returns a B-tensor containing the expected magnitude of the score
		function at each time `t`.
		"""
		variance = (torch.pow(self.sigma, 2 * t) - 1) / \
			(2 * torch.log(self.sigma))
		return 1 / variance  # Shape: B
		
	def sample_prior(self, num_samples, t):
		"""
		Samples from the prior distribution specified by the SDE at time `t`.
		Arguments:
			`num_samples`: B, the number of samples to return
			`t`: a B-tensor containing the times in the SDE to sample for
		Returns a B x `self.input_shape` tensor for the `xt` values that are
		sampled.
		"""
		shape = torch.Size([num_samples]) + torch.Size(self.input_shape)
		z = torch.normal(
			torch.zeros(shape).to(DEVICE), torch.ones(shape).to(DEVICE),
			generator=self.rng
		)  # Shape: B x ...
		
		variance = (torch.pow(self.sigma, 2 * t) - 1) / \
			(2 * torch.log(self.sigma))
		std = self._inflate_dims(torch.sqrt(variance))	# Shape: B x ...
		
		return z * std
	
	def __str__(self):
		return self.string


class VariancePreservingSDE(SDE):
	
	def __init__(self, beta_0, beta_1, input_shape, seed=None):
		"""
		Arguments:
			`beta_0`: beta(0); see below
			`beta_1`: beta(1); beta(t) will be linearly interpolated
				between beta(0) and beta(1)
			`input_shape`: a tuple of ints which is the shape of input
				tensors x; does not include batch dimension
			`seed`: random seed for sampling and running the SDE
		"""
		super().__init__(input_shape, seed)
		
		self.beta_0 = torch.tensor(beta_0).to(DEVICE)
		self.delta_beta = torch.tensor(beta_1 - beta_0).to(DEVICE)
		self.string = "Variance Preserving SDE (beta(t) = %.2f + %.2ft)" % (
			beta_0, beta_1 - beta_0
		)
		
	def _beta(self, t):
		"""
		Computes beta(t).
		Arguments:
			`t`: a B-tensor of times
		Returns a B-tensor of beta values.
		"""
		return self.beta_0 + (self.delta_beta * t)
		
	def _beta_bar(self, t):
		"""
		Computes the integral of beta(0) to beta(t).
		Arguments:
			`t`: a B-tensor of times
		Returns a B-tensor of beta-bar values.
		"""
		return (self.beta_0 * t) + (0.5 * self.delta_beta * torch.square(t))
	
	def drift_coef_func(self, xt, t):
		"""
		Definition of drift coefficient f(xt, t).
		Arguments:
			`xt`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE for each input
		Returns a B x `self.input_shape` tensor.
		"""
		return -0.5 * self._inflate_dims(self._beta(t)) * xt
	
	def diff_coef_func(self, xt, t):
		"""
		Definition of diffusion coefficient g(t).
		Arguments:
			`xt`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE
		Returns a B x `self.input_shape` tensor.
		"""
		return self._inflate_dims(torch.sqrt(self._beta(t)))
	
	def forward(self, x0, t, return_score=True):
		"""
		Runs the SDE forward given starting point `x0` and a time `t`.
		Optionally returns the score: the gradient (with respect to x) of the
		log conditional probability, pt(xt | x0).
		Arguments:
			`x0`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the SDE for each input
		Returns a B x `self.input_shape` tensor to represent xt. If
		`return_score` is True, then also returns a B x `self.input_shape`
		tensor which is the gradient of the log conditional probability (with
		respect to `xt`).
		"""
		z = torch.normal(
			torch.zeros_like(x0), torch.ones_like(x0), generator=self.rng
		)  # Shape: B x ...
		
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
		"""
		Returns the average magnitude (squared L2 norm) of the score (averaged
		over sampling and data distribution), divided by the size of `x`. That
		is, this should be ||score||_2^2 / d, where d is the size of `x`.
		Arguments:
			`t`: a B-tensor containing the time in the SDE
		Returns a B-tensor containing the expected magnitude of the score
		function at each time `t`.
		"""
		variance = 1 - torch.exp(-self._beta_bar(t))
		return 1 / variance  # Shape: B
		
	def sample_prior(self, num_samples, t):
		"""
		Samples from the prior distribution specified by the SDE at time `t`.
		Arguments:
			`num_samples`: B, the number of samples to return
			`t`: a B-tensor containing the times in the SDE to sample for
		Returns a B x `self.input_shape` tensor for the `xt` values that are
		sampled.
		"""
		# We will sample in the limit as t approaches infinity
		shape = torch.Size([num_samples]) + torch.Size(self.input_shape)
		return torch.normal(
			torch.zeros(shape).to(DEVICE), torch.ones(shape).to(DEVICE),
			generator=self.rng
		)  # Shape: B x ...
	
	def __str__(self):
		return self.string


if __name__ == "__main__":
	input_shape = (1, 28, 28)
	sde = VarianceExplodingSDE(25.0, input_shape)
	x0 = torch.empty((32,) + input_shape, device=DEVICE)
	t = torch.rand(32, device=DEVICE)
	xt, score = sde.forward(x0, t)
