import pandas as pd
import rdkit.Chem
import torch
import torch_geometric
import networkx as nx

# Define device
if torch.cuda.is_available():
	DEVICE = "cuda"
else:
	DEVICE = "cpu"


ZINC250K_PATH = "/gstore/home/tsenga5/discrete_graph_diffusion/data/250k_rndm_zinc_drugs_clean_3.csv"


def smiles_to_networkx(smiles, kekulize=True):
	"""
	Converts a SMILES string to a NetworkX graph. The graph will retain the
	atomic number and bond type for nodes and edges (respectively), under the
	keys `atomic_num` and `bond_type` (respectively).
	Arguments:
		`smiles`: a SMILES string
		`kekulize`: if True, denote single/double bonds separately in aromatic
			structures instead of using the aromatic bond type
	Returns a NetworkX graph.
	"""
	mol = rdkit.Chem.MolFromSmiles(smiles)
	if kekulize:
		rdkit.Chem.Kekulize(mol)
	g = nx.Graph()
	for atom in mol.GetAtoms():
		g.add_node(
			atom.GetIdx(),
			atomic_num=atom.GetAtomicNum()
		)
	for bond in mol.GetBonds():
		g.add_edge(
			bond.GetBeginAtomIdx(),
			bond.GetEndAtomIdx(),
			bond_type=bond.GetBondType()
		)
	return g


MAX_NUM_NODES = 38
ATOM_MAP = torch.tensor([6, 7, 8, 9, 16, 17, 35, 53, 15])
BOND_MAP = torch.tensor([1, 2, 3, 12])

def smiles_to_matrices(smiles, max_num_nodes=MAX_NUM_NODES):
	"""
	Converts a SMILES string to an adjacency matrix and node-feature matrix.
	Arguments:
		`smiles`: a SMILES string
		`max_num_nodes`: the size of the matrices returned, M
	Returns an M x M adjacency matrix with bond orders as edge weights, and an
	M x D matrix of one-hot-encoded atom features, with D being the size of
	`ATOM_MAP`.
	"""
	g = smiles_to_networkx(smiles)
	assert len(g) <= max_num_nodes
	data = torch_geometric.utils.from_networkx(g) 
	
	# Get atom features
	atom_inds = torch.argmax(
		(data.atomic_num.view(-1, 1) == ATOM_MAP).int(), dim=1
	)
	x = torch.nn.functional.one_hot(atom_inds, num_classes=len(ATOM_MAP))

	# Get bond features
	# For aromatic bonds, set them to be 1.5
	edge_attr = data.bond_type.float()
	edge_attr[edge_attr == BOND_MAP[-1]] = 1.5
	
	adj = torch_geometric.utils.to_dense_adj(
		data.edge_index, edge_attr=edge_attr, max_num_nodes=max_num_nodes
	)[0]
	x = torch.cat(
		[x, torch.zeros(max_num_nodes - x.shape[0], x.shape[1])], dim=0
	)
	
	return adj, x


def label_aromaticity(smiles):
	"""
	Assigns a molecule an integer label based on aromaticity.
	Arguments:
		`smiles`: a SMILES string
	Returns 1 if the molecule is aromatic, and 0 otherwise.
	"""
	mol = rdkit.Chem.MolFromSmiles(smiles)
	return 1 if any(a.GetIsAromatic() for a in mol.GetAtoms()) else 0


def label_element_presence(smiles, elements=(9, 17, 35, 53)):
	"""
	Assigns a molecule an integer label based on whether or not the molecule has
	one of the elements in `elements`.
	Arguments:
		`smiles`: a SMILES string
		`elements`: an iterable of atomic numbers to check for; by default
			checks for the presence of halogens F through I
	Returns 1 if the molecule contains an element in `elements`, and 0
	otherwise.
	"""
	mol = rdkit.Chem.MolFromSmiles(smiles)
	return 1 if any(a.GetAtomicNum() in elements for a in mol.GetAtoms()) else 0


def label_bicyclicity(smiles):
	"""
	Assigns a molecule an integer label based on bicyclicity.
	Arguments:
		`smiles`: a SMILES string
	Returns 1 if the molecule is bicyclic, and 0 otherwise.
	"""
	g = smiles_to_networkx(smiles)
	cycles = nx.cycle_basis(g)
	if len(cycles) < 2:
		return 0
	cycles = [set(x) for x in cycles]
	for i in range(len(cycles)):
		for j in range(i):
			if cycles[i] & cycles[j]:
				return 1
	return 0


def label_cycle_num(smiles, nums_to_label=None):
	"""
	Assigns a molecule an integer label of the number of cycles.
	Arguments:
		`smiles`: a SMILES string
		`nums_to_label`: a list of which cycle numbers to given labels to;
			molecules with a number of cycles not in this list get a label of
			-1; by default labels all molecules
	Returns the number of cycles in the molecule, or -1 if the number is not in
	`nums_to_label`.
	"""
	g = smiles_to_networkx(smiles)
	cycles = nx.cycle_basis(g)
	if not nums_to_label or len(cycles) in nums_to_label:
		return len(cycles)
	else:
		return -1


def label_macrocyclicity(smiles):
	"""
	Assigns a molecule an integer label based on macrocyclicity, irrespective of
	aromatic cycles.
	Arguments:
		`smiles`: a SMILES string
	Returns 1 if the molecule has a non-aromatic cycle, and 0 otherwise.
	"""
	mol = rdkit.Chem.MolFromSmiles(smiles)
	for ring in mol.GetRingInfo().AtomRings():
		aromatic = all(
			mol.GetAtomWithIdx(atom_ind).GetIsAromatic() for atom_ind in ring
		)
		if not aromatic:
			return 1
	return 0


def label_heteroaromaticity(smiles):
	"""
	Assigns a molecule an integer label based on heteroaromaticity.
	Arguments:
		`smiles`: a SMILES string
	Returns 1 if the molecule has an heteroaromatic cycle (i.e. aromatic cycle
	with atom other than C), 0 if the molecule has only carbon aromatic cycles,
	and -1 if the molecule is not aromatic at all.
	"""
	mol = rdkit.Chem.MolFromSmiles(smiles)
	found_aromatic = False
	for ring in mol.GetRingInfo().AtomRings():
		aromatic = all(
			mol.GetAtomWithIdx(atom_ind).GetIsAromatic() for atom_ind in ring
		)
		if aromatic:
			heteroaromatic = any(
				mol.GetAtomWithIdx(atom_ind).GetAtomicNum() != 6 for atom_ind in ring
			)
			if heteroaromatic:
				return 1
			found_aromatic = True
	if found_aromatic:
		return 0
	else:
		return -1


class ZINCDataset(torch.utils.data.Dataset):
	def __init__(self, label_method=None, **label_kwargs):
		"""
		Create a PyTorch IterableDataset which yields molecular graphs as pairs
		of adjacency and data matrices.
		Arguments:
			`label_method`: the method used to label molecules; if not provided,
				every molecule will be given a label of 0
			`label_kwargs`: provided to the labeling method
		"""
		super().__init__()

		zinc_table = pd.read_csv(ZINC250K_PATH, sep=",", header=0)
		zinc_table["smiles"] = zinc_table["smiles"].str.strip()
		self.all_smiles = zinc_table["smiles"]
		if label_method == "bicyclicity":
			label_func = label_bicyclicity
		elif label_method == "num_cycles":
			label_func = label_cycle_num
		elif label_method == "aromaticity":
			label_func = label_aromaticity
		elif label_method == "elements":
			label_func = label_element_presence
		elif label_method == "macrocyclicity":
			label_func = label_macrocyclicity
		elif label_method == "heteroaromaticity":
			label_func = label_heteroaromaticity
		elif label_method is None:
			label_func = lambda s, **label_kwargs: 0
		else:
			raise ValueError("Unknown label method: %s" % label_method)

		self.target = torch.tensor([
			label_func(s, **label_kwargs) for s in self.all_smiles	
		])

		# Filter out things that didn't get a label
		mask = (self.target != -1).numpy()
		self.all_smiles = self.all_smiles[mask].reset_index(drop=True)
		self.target = self.target[mask]

	def __getitem__(self, index):
		"""
		Returns a single tensor representing the molecule at index `index` in
		`self.all_smiles`, and a scalar tensor of the label/class. The first
		returned tensor is M x (M + D), where the first M x M is the adjacency
		matrix and the next M x D is the node-feature matrix.
		"""
		adj, x = smiles_to_matrices(self.all_smiles[index])
		adj, x = adj.to(DEVICE), x.to(DEVICE)
		return torch.cat([adj, x], dim=1), self.target[index]

	def __len__(self):
		return len(self.all_smiles)


if __name__ == "__main__":
	dataset = ZINCDataset()
	data_loader = torch.utils.data.DataLoader(
		dataset, batch_size=32, shuffle=False
	)
	batch = next(iter(data_loader))
