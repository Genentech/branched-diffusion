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
	model, sde, data_loader, model_type, class_mapper, num_epochs,
	learning_rate, _run, loss_weighting_type="empirical_norm",
	weight_func=None, t_limit=1, data_loader_returns_t_and_b=False
):
	"""
	Trains a diffusion model using the given instantiated model and SDE object.
	Arguments:
		`model`: an instantiated score model which takes in x, t and predicts
			score
		`sde`: an SDE object
		`data_loader`: a DataLoader object that yields batches of data as
			tensors in pairs: x, y
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
		`data_loader_returns_t_and_b`: if True, data loader returns the time
			tensor and branch index along with data along with the data, instead
			of the class label
	"""
	assert model_type in ("branched", "labelguided")

	run_num = _run._id
	output_dir = os.path.join(MODEL_DIR, str(run_num))

	model.train()
	torch.set_grad_enabled(True)
	optim = torch.optim.Adam(model.parameters(), lr=learning_rate)

	for epoch_num in range(num_epochs):
		if "on_epoch_start" in dir(data_loader.dataset):
			data_loader.dataset.on_epoch_start()

		batch_losses = []
		t_iter = tqdm.tqdm(data_loader)
		for batch in t_iter:
			if data_loader_returns_t_and_b:
				x0, t, branch_inds = batch
				t = t.to(DEVICE).float()
				branch_inds = branch_inds.to(DEVICE)
			else:
				x0, y = batch
				# Sample random times from 0 to 1
				t = (torch.rand(x0.shape[0]) * t_limit).to(DEVICE)
			x0 = x0.to(DEVICE).float()
			
			# Run SDE forward to get xt and the true score at xt
			xt, true_score = sde.forward(x0, t)
			
			# Get model-predicted score
			if model_type == "branched":
				pred_score = model(xt, t)
			else:
				class_inds = class_mapper(y).long()
				pred_score = model(xt, t, class_inds)
			
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
				if not data_loader_returns_t_and_b:
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
