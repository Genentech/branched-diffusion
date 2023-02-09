import numpy as np
import torch
import h5py

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


class SingleCellDataset(torch.utils.data.Dataset):
	def __init__(self, scrna_h5_path):
		"""
		Create a PyTorch IterableDataset which yields (tabular) single-cell
		expression data.
		Arguments:
			`scrna_h5_path`: path to saved scRNA-seq dataset, which contains the
				preprocessed and normalized data, cell cluster IDs, and gene
				names
		"""
		super().__init__()

		self.scrna_h5_path = scrna_h5_path
		with h5py.File(scrna_h5_path, "r") as f:
			self.data = f["x"][:]
			self.cell_cluster = f["cell_cluster_id"][:]

	def __getitem__(self, index):
		"""
		Returns a pair of tensors for the item at the given index: a D-tensor of
		expression counts, and a scalar tensor of the cell ID.
		"""
		x = torch.tensor(self.data[index])
		y = torch.tensor(self.cell_cluster[index])
		return x.to(DEVICE), y.to(DEVICE)

	def __len__(self):
		return len(self.data)


if __name__ == "__main__":
	scrna_h5_path = "/gstore/data/resbioai/tsenga5/branched_diffusion/data/scrna/covid_flu/covid_flu_processed.h5"
	dataset = SingleCellDataset(scrna_h5_path)
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=32, shuffle=False
	)
	batch = next(iter(data_loader))
