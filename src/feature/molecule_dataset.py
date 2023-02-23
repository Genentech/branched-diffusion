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


class ZINCDataset(torch.utils.data.Dataset):
	def __init__(self):
		"""
		Create a PyTorch IterableDataset which yields molecular graphs as pairs
		of adjacency and data matrices.
		"""
		super().__init__()

		zinc_table = pd.read_csv(ZINC250K_PATH, sep=",", header=0)
		zinc_table["smiles"] = zinc_table["smiles"].str.strip()
		self.all_smiles = zinc_table["smiles"]	
		self.target = torch.zeros(len(self.all_smiles))

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
