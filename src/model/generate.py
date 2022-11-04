import torch
import numpy as np
import scipy.integrate
import tqdm

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


def generate_continuous_samples(
	model, sde, class_to_sample, class_time_to_branch_index, sampler="em",
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
			form to `class_time_to_branch_index`)
		`class_time_to_branch_index`: function that takes in B-tensors of class
			and time and maps to a B-tensor of branch indices
		`sampler`: one of "em", "pc", or "ode" for Euler-Maruyama,
			predictor-corrector, or ordinary differential equation, respectively
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
	# First, sample from the prior distribution at some late time t
	if initial_samples is not None:
		xt = initial_samples
	else:
		t = (torch.ones(num_samples) * t_limit).to(DEVICE)
		xt = sde.sample_prior(num_samples, t)

	class_tens = torch.tensor([class_to_sample], device=DEVICE)
	
	# Disable gradient computation in model
	torch.set_grad_enabled(False)
	
	if sampler == "em":
		# Euler-Maruyama
		time_steps = torch.linspace(t_limit, t_start, num_steps).to(DEVICE)
		# (descending order)
		step_size = time_steps[0] - time_steps[1]
		
		# Step backward through time starting at xt, simulating the reverse SDE
		x = xt
		t_iter = tqdm.tqdm(time_steps) if verbose else time_steps
		for time_step in t_iter:
			branch_index = class_time_to_branch_index(
				class_tens, time_step[None]
			)[0]
			t = torch.ones(num_samples).to(DEVICE) * time_step
			f = sde.drift_coef_func(x, t)
			g = sde.diff_coef_func(x, t)
			dw = torch.randn_like(x)
			
			drift = (
				f - (torch.square(g) * model(x, t, [branch_index])[:, 0])
			) * step_size
			diff = g * torch.sqrt(step_size) * dw
			
			mean_x = x - drift	# Subtract because step size is really negative
			x = mean_x + diff
		return mean_x  # Last step: don't include the diffusion/randomized term
	
	elif sampler == "pc":
		# Predictor-Corrector
		time_steps = torch.linspace(t_limit, t_start, num_steps).to(DEVICE)
		# (descending order)
		step_size = time_steps[0] - time_steps[1]
		
		# Step backward through time starting at xt
		x = xt
		t_iter = tqdm.tqdm(time_steps) if verbose else time_steps
		for time_step in t_iter:
			branch_index = class_time_to_branch_index(
				class_tens, time_step[None]
			)[0]
			t = torch.ones(num_samples).to(DEVICE) * time_step
			
			# Take Langevin MCMC step
			score = model(x, t, [branch_index])[:, 0]
			
			snr = 0.1
			score_norm = torch.mean(
				torch.norm(score.reshape(score.shape[0], -1), dim=-1)
			)
			alpha = snr * (
				torch.prod(torch.tensor(x.shape[1:])) / torch.square(score_norm)
			)
			
			x = x + ((alpha / 2) * score) + \
				(torch.sqrt(alpha) * torch.randn_like(x))
			
			# Take SDE step
			f = sde.drift_coef_func(x, t)
			g = sde.diff_coef_func(x, t)
			dw = torch.randn_like(x)
			
			drift = (
				f - (torch.square(g) * model(x, t, [branch_index])[:, 0])
			) * step_size
			diff = g * torch.sqrt(step_size) * dw
			
			mean_x = x - drift	# Subtract because step size is really negative
			x = mean_x + diff
			
		return mean_x  # Last step: don't include the diffusion/randomized term
	
	elif sampler == "ode":
		# ODE
		t = torch.ones(num_samples).to(DEVICE)
		x_shape = tuple(torch.tensor(xt.shape).numpy())
		
		# Define function used by ODE solver:
		def ode_func(t, x):
			# t is a scalar; x is a 1D NumPy array
			x_tens = torch.tensor(x).float().to(DEVICE).view(x_shape)
			t_tens = torch.ones(num_samples).to(DEVICE) * t

			time_step_tens = torch.tensor([t], device=DEVICE)
			branch_index = class_time_to_branch_index(
				class_tens, time_step_tens
			)[0]
			
			score_tens = model(x_tens, t_tens, [branch_index])[:, 0]
			f_tens = sde.drift_coef_func(x_tens, t_tens)
			g_tens = sde.diff_coef_func(x_tens, t_tens)
			
			step_tens = f_tens - (0.5 * torch.square(g_tens) * score_tens)
			return step_tens.reshape(-1).cpu().numpy().astype(np.float64)
		
		# Run the ODE solver
		result = scipy.integrate.solve_ivp(
			ode_func, (t_limit, t_start),
			xt.reshape(-1).cpu().numpy(),
			rtol=0.00001, atol=0.00001,
			method="RK45"
		)
		if verbose:
			print("Number of ODE function evaluations: %d" % result.nfev)

		return torch.tensor(result.y[:, -1]).to(DEVICE).reshape(x_shape)


def generate_discrete_samples(
	model, diffuser, class_to_sample, class_time_to_branch_index,
	num_samples=64, t_start=0, t_limit=1000, initial_samples=None, verbose=False
):
	"""
	Generates samples from a trained score model and discrete diffuser. This
	first generates a sample from the prior distribution a `t_limit`, then steps
	backward through time to generate new data points.
	Arguments:
		`model`: a trained score model which takes in x, t and predicts score
		`diffuser`: a DiscreteDiffuser object
		`class_to_sample`: class to sample from (will be an argument in tensor
			form to `class_time_to_branch_index`)
		`class_time_to_branch_index`: function that takes in B-tensors of class
			and time and maps to a B-tensor of branch indices
		`num_samples`: number of objects to return
		`t_start`: last time step to stop at (a smaller positive integer) than
			`t_limit`
		`t_limit`: the time step to start generating at (a larger positive
			integer than `t_start`)
		`initial_samples`: if given, this is a tensor which contains the samples
			to start from initially, to be used instead of sampling from the
			diffuser's defined prior
		`verbose`: if True, print out progress bar
	Returns a tensor of size `num_samples` x ...
	"""
	# First, sample from the prior distribution at some late time t
	if initial_samples is not None:
		xt = initial_samples
	else:
		t = (torch.ones(num_samples) * t_limit).to(DEVICE)
		xt = diffuser.sample_prior(num_samples, t)

	class_tens = torch.tensor([class_to_sample], device=DEVICE)
	
	# Disable gradient computation in model
	torch.set_grad_enabled(False)
	
	time_steps = torch.arange(t_limit, t_start, step=-1).to(DEVICE)
	# (descending order)
	
	# Step backward through time starting at xt
	x = xt
	t_iter = tqdm.tqdm(time_steps) if verbose else time_steps
	for time_step in t_iter:
		branch_index = class_time_to_branch_index(
			class_tens, time_step[None]
		)[0]
		t = torch.ones(num_samples).to(DEVICE) * time_step
		z = model(xt, t, branch_index)[:, 0]
		xt = diffuser.reverse_step(xt, t, z)
	return xt
