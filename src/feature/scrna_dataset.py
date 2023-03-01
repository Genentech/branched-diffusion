import numpy as np
import torch
import h5py

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


class SingleCellDataset(torch.utils.data.Dataset):
	def __init__(self, scrna_h5_path, autoencoder_path=None):
		"""
		Create a PyTorch IterableDataset which yields (tabular) single-cell
		expression data.
		Arguments:
			`scrna_h5_path`: path to saved scRNA-seq dataset, which contains the
				preprocessed and normalized data, cell cluster IDs, and gene
				names
			`autoencoder_path`: optional path to trained scVI autoencoder, which
				causes all outputs to be encoded in latent space
		"""
		super().__init__()

		self.scrna_h5_path = scrna_h5_path
		with h5py.File(scrna_h5_path, "r") as f:
			self.data = f["x"][:]
			self.cell_cluster = f["cell_cluster_id"][:]

		if autoencoder_path:
			import scvi
			self.autoencoder = scvi.model.LinearSCVI.load(autoencoder_path)
		else:
			self.autoencoder = None

	def encode_batch(self, x):
		"""
		Encodes a batch of gene expressions into latent space. Requires the
		dataset to be initialized with a trained autoencoder.
		Arguments:
			`x`: a B x D tensor of expressions; note that D must match the N x D
				array `x` stored at`self.scrna_h5_path`
		Returns a B x D' tensor of latent-space representations, where D' is the
		size of the latent space.
		"""
		if not self.autoencoder:
			raise ValueError("No initialized autoencoder found")
		enc_pred = self.autoencoder.module.inference(
			x, torch.ones(len(x), device=DEVICE)
		)
		return enc_pred["qz"].loc.detach()

	def decode_batch(self, z):
		"""
		Decodes a batch of latent-space representations into gene expressions.
		Requires the dataset to be initialized with a trained autoencoder.
		Arguments:
			`z`: a B x D' tensor of latent vectors, where D' is the latent-space
				size
		Returns a B x D tensor of gene expressions, where D matches the N x D
		array `x` stored at`self.scrna_h5_path`.
		"""
		if not self.autoencoder:
			raise ValueError("No initialized autoencoder found")
		dec_pred = self.autoencoder.module.generative(
			z,
		    torch.ones((len(z), 1), device=DEVICE),
			torch.zeros((len(z), 1), device=DEVICE)
		)
		return dec_pred["px"].scale.detach()

	def __getitem__(self, index):
		"""
		Returns a pair of tensors for the item at the given index: a D-tensor of
		expression counts, and a scalar tensor of the cell ID.
		"""
		x = torch.tensor(self.data[index]).to(DEVICE)
		y = torch.tensor(self.cell_cluster[index]).to(DEVICE)

		if self.autoencoder:
			z = self.encode_batch(x[None])[0]
			return z, y
		else:
			return x, y

	def __len__(self):
		return len(self.data)


if __name__ == "__main__":
	scrna_h5_path = "/gstore/data/resbioai/tsenga5/branched_diffusion/data/scrna/covid_flu/processed/covid_flu_processed_reduced_genes.h5"
	autoencoder_path = "/gstore/data/resbioai/tsenga5/branched_diffusion/models/trained_models/scrna_vaes/covid_flu/covid_flu_processed_reduced_genes_ldvae/"
	dataset = SingleCellDataset(scrna_h5_path, autoencoder_path)
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=32, shuffle=False
	)
	enc_batch = next(iter(data_loader))
	z = enc_batch[0]
	x_hat = dataset.decode_batch(z)
