import torch
import numpy as np
import tqdm
import os
import sacred
import model.util as util

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
	model, sde, data_loader, class_time_to_branch_index, num_epochs,
	learning_rate, _run, loss_weighting_type="empirical_norm", t_limit=1
):
	"""
	Trains a diffusion model using the given instantiated model and SDE object.
	Arguments:
		`model`: an instantiated score model which takes in x, t and predicts
			score
		`sde`: an SDE object
		`data_loader`: a DataLoader object that yields batches of data as
			tensors in pairs: x, y
		`class_time_to_branch_index`: function that takes in B-tensors of class
			and time and maps to a B-tensor of branch indices
		`num_epochs`: number of epochs to train for
		`learning_rate`: learning rate to use for training
		`loss_weighting_type`: method for weighting the loss; can be "ml" to
			weight by g^2, "expected_norm" to weight by the expected mean
			magnitude of the loss, "empirical_norm" to weight by the observed
			true norm, or None to do no weighting at all
		`t_limit`: training will occur between time 0 and `t_limit`
	"""
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
			
			# Sample random times from 0 to 1
			t = (torch.rand(x0.shape[0]) * t_limit).to(DEVICE)

			# Compute branch indices
			branch_inds = class_time_to_branch_index(y, t)
			
			# Run SDE forward to get xt and the true score at xt
			xt, true_score = sde.forward(x0, t)
			
			# Get model-predicted score
			pred_score = model(xt, t)
			
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
			
			# Compute loss as MSE
			loss = model.loss(pred_score, true_score, branch_inds, loss_weight)
			loss_val = loss.item()
			t_iter.set_description("Loss: %.2f" % loss_val)

			if np.isnan(loss_val):
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
