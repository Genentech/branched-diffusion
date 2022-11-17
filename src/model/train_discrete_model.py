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
	model, diffuser, data_loader, model_type, class_mapper, num_epochs,
	learning_rate, _run, t_limit=1000
):
	"""
	Trains a diffusion model using the given instantiated model and discrete
	diffuser object.
	Arguments:
		`model`: an instantiated model which takes in x, t and predicts a
			posterior
		`diffuser`: a DiscreteDiffuser object
		`data_loader`: a DataLoader object that yields batches of data as
			tensors in pairs: x, y
		`model_type`: either "branched" or "labelguided"
		`class_mapper`: for "branched" model types, a function that takes in
			B-tensors of class and time and maps to a B-tensor of branch
			indices; for "labelguided" model types, a function that takes in
			B-tensors of class and maps to a B-tensor of class indices
		`class_time_to_branch_index`: function that takes in B-tensors of class
			and time and maps to a B-tensor of branch indices
		`num_epochs`: number of epochs to train for
		`learning_rate`: learning rate to use for training
		`t_limit`: training will occur between time 1 and `t_limit`
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
			
			# Sample random times between 1 and t_limit (inclusive)
			t = torch.randint(
			    t_limit, size=(x0.shape[0],), device=DEVICE
			) + 1

			# Run diffusion forward to get xt and the posterior parameter to
			# predict
			xt, true_post = diffuser.forward(x0, t)
			
			# Get model-predicted posterior parameter
			if model_type == "branched":
				pred_post = model(xt, t)
			else:
				class_inds = class_mapper(y).long()
				pred_post = model(xt, t, class_inds)

			# Compute loss
			if model_type == "branched":
				# Compute branch indices
				branch_inds = class_mapper(y, t)
				loss = model.loss(pred_post, true_post, branch_inds)
			else:
				loss = model.loss(pred_post, true_post)

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
	model, diffuser, data_loader, class_time_to_branch_index, num_epochs,
	learning_rate, _run, t_limit=1
):
	"""
	Wrapper for `train_model`.
	"""
	train_model(
		model, diffuser, data_loader, "branched", class_time_to_branch_index,
		num_epochs, learning_rate, _run, t_limit=t_limit
	)

	
@train_ex.command
def train_label_guided_model(
	model, diffuser, data_loader, class_to_class_index, num_epochs, learning_rate,
	_run, t_limit=1
):
	"""
	Wrapper for `train_model`.
	"""
	train_model(
		model, diffuser, data_loader, "labelguided", class_to_class_index,
		num_epochs, learning_rate, _run, t_limit=t_limit
	)
