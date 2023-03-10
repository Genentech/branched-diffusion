import torch

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


class DiscreteDiffuser:
	# Base class for discrete diffusers
	def __init__(self, input_shape, seed=None):
		"""
		Arguments:
			`input_shape`: a tuple of ints which is the shape of input tensors
				x; does not include batch dimension
			`seed`: random seed for sampling and running the diffusion process
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
	
	def forward(self, x0, t, return_posterior=True):
		"""
		Runs diffusion process forward given starting point `x0` and a time `t`.
		Optionally also returns a tensor which represents the posterior of
		x_{t-1} given xt and/or x0 (e.g. probability, mean, noise, etc.)
		Arguments:
			`x0`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the diffusion process for
				each input
		Returns a B x `self.input_shape` tensor to represent xt. If
		`return_posterior` is True, then also returns a B x `self.input_shape`
		tensor which is a parameter of the posterior.
		"""
		if return_posterior:
			return torch.zeros_like(x0), torch.zeros_like(x0)
		else:
			return torch.zeros_like(x0)

	def reverse_step(self, xt, t, post):
		"""
		Performs a reverse sampling step to compute x_{t-1} given xt and the
		posterior quantity (or an estimate of it) defined in `posterior`.
		Arguments:
			`xt`: a B x `self.input_shape` tensor containing the data at time t
			`t`: a B-tensor containing the time in the diffusion process for
				each input
			`post`: a B x `self.input_shape` tensor containing the posterior
				quantity (or a model-predicted estimate of it) as defined in
				`posterior`
		Returns a B x `self.input_shape` tensor for x_{t-1}.
		"""
		return torch.zeros_like(x0)
		
	def sample_prior(self, num_samples, t):
		"""
		Samples from the prior distribution specified by the diffusion process
		at time `t`.
		Arguments:
			`num_samples`: B, the number of samples to return
			`t`: a B-tensor containing the time in the diffusion process for
				each input
		Returns a B x `self.input_shape` tensor for the `xt` values that are
		sampled.
		"""
		return torch.zeros(torch.Size([num_samples] + list(self.input_shape)))
	
	def __str__(self):
		return "Base Discrete Diffuser"


class GaussianDiffuser(DiscreteDiffuser):

	def __init__(self, beta_1, delta_beta, input_shape, seed=None):
		"""
		Arguments:
			`beta_1`: beta(1), the first value of beta at t = 1
			`delta_beta`: beta(t) will be linear with this slope
			`input_shape`: a tuple of ints which is the shape of input tensors
				x; does not include batch dimension
			`seed`: random seed for sampling and running the diffusion process
		"""
		super().__init__(input_shape, seed)

		self.beta_1 = torch.tensor(beta_1).to(DEVICE)
		self.delta_beta = torch.tensor(delta_beta).to(DEVICE)
		self.string = "Gaussian Diffuser (beta(t) = %.2f + %.2ft)" % (
			beta_1, delta_beta
		)

	def _beta(self, t):
		"""
		Computes beta(t).
		Arguments:
			`t`: a B-tensor of times
		Returns a B-tensor of beta values.
		"""
		# Subtract 1 from t: when t = 1, beta(t) = beta_1
		return self.beta_1 + (self.delta_beta * (t - 1))
		
	def _alpha(self, t):
		"""
		Computes alpha(t).
		Arguments:
			`t`: a B-tensor of times
		Returns a B-tensor of alpha values.
		"""
		return 1 - self._beta(t)
	
	def _alpha_bar(self, t):
		"""
		Computes alpha-bar(t).
		Arguments:
			`t`: a B-tensor of times
		Returns a B-tensor of alpha-bar values.
		"""
		max_t = torch.max(t)
		t_range = torch.arange(max_t.int() + 1, device=DEVICE)
		alphas = self._alpha(t_range)
		alphas_prod = torch.cumprod(alphas, dim=0)
		return alphas_prod[t.long()]

	def forward(self, x0, t, return_posterior=True):
		"""
		Runs diffusion process forward given starting point `x0` and a time `t`.
		Optionally also returns a tensor which represents the posterior of
		x_{t-1} given xt and/or x0 (e.g. probability, mean, noise, etc.)
		Arguments:
			`x0`: a B x `self.input_shape` tensor containing the data at some
				time points
			`t`: a B-tensor containing the time in the diffusion process for
				each input
		Returns a B x `self.input_shape` tensor to represent xt. If
		`return_posterior` is True, then also returns a B x `self.input_shape`
		tensor which is a parameter of the posterior.
		"""
		z = torch.normal(
			torch.zeros_like(x0), torch.ones_like(x0), generator=self.rng
		)  # Shape: B x ...

		alpha_bar = self._inflate_dims(self._alpha_bar(t))
		xt = (torch.sqrt(alpha_bar) * x0) + (torch.sqrt(1 - alpha_bar) * z)

		if return_posterior:
			return xt, z
		return xt

	def reverse_step(self, xt, t, post):
		"""
		Performs a reverse sampling step to compute x_{t-1} given xt and the
		posterior quantity (or an estimate of it) defined in `posterior`.
		Arguments:
			`xt`: a B x `self.input_shape` tensor containing the data at time t
			`t`: a B-tensor containing the time in the diffusion process for
				each input
			`post`: a B x `self.input_shape` tensor containing the posterior
				quantity (or a model-predicted estimate of it) as defined in
				`posterior`
		Returns a B x `self.input_shape` tensor for x_{t-1}.
		"""
		beta = self._beta(t)
		alpha = self._alpha(t)
		alpha_bar = self._alpha_bar(t)
		alpha_bar_1 = self._alpha_bar(t - 1)
		beta_tilde = beta * (1 - alpha_bar_1) / (1 - alpha_bar)

		z = torch.normal(
			torch.zeros_like(xt), torch.ones_like(xt), generator=self.rng
		)

		d = xt - (post * self._inflate_dims(beta / torch.sqrt(1 - alpha_bar)))
		d = d / self._inflate_dims(torch.sqrt(alpha))
		std = torch.sqrt(beta_tilde)
		std[t == 1] = 0  # No noise for the last step
		return d + (z * self._inflate_dims(std))
		
	def sample_prior(self, num_samples, t):
		"""
		Samples from the prior distribution specified by the diffusion process
		at time `t`.
		Arguments:
			`num_samples`: B, the number of samples to return
			`t`: a B-tensor containing the time in the diffusion process for
				each input
		Returns a B x `self.input_shape` tensor for the `xt` values that are
		sampled.
		"""
		# Ignore t
		size = torch.Size([num_samples] + list(self.input_shape))
		return torch.normal(
			torch.zeros(size, device=DEVICE), torch.ones(size, device=DEVICE),
			generator=self.rng
		)
	
	def __str__(self):
		return self.string


if __name__ == "__main__":
	input_shape = (1, 28, 28)
	diffuser = GaussianDiffuser(1e-4, 1.2e-4, input_shape)
	x0 = torch.empty((32,) + input_shape, device=DEVICE)
	t = torch.randint(1, 1001, (32,), device=DEVICE)
	xt, z = diffuser.forward(x0, t)
	xt1 = diffuser.reverse_step(xt, t, z)
	xT = diffuser.sample_prior(32, None)
