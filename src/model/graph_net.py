import torch
import math
import numpy as np
from model.util import sanitize_sacred_arguments

def node_flags(adj, eps=1e-5):

	flags = torch.abs(adj).sum(-1).gt(eps).to(dtype=torch.float32)

	if len(flags.shape)==3:
		flags = flags[:,0,:]
	return flags


# -------- Mask batch of node features with 0-1 flags tensor --------
def mask_x(x, flags):

	if flags is None:
		flags = torch.ones((x.shape[0], x.shape[1]), device=x.device)
	return x * flags[:,:,None]


# -------- Mask batch of adjacency matrices with 0-1 flags tensor --------
def mask_adjs(adjs, flags):
	"""
	:param adjs:  B x N x N or B x C x N x N
	:param flags: B x N
	:return:
	"""
	if flags is None:
		flags = torch.ones((adjs.shape[0], adjs.shape[-1]), device=adjs.device)

	if len(adjs.shape) == 4:
		flags = flags.unsqueeze(1)	# B x 1 x N
	adjs = adjs * flags.unsqueeze(-1)
	adjs = adjs * flags.unsqueeze(-2)
	return adjs


# -------- Create higher order adjacency matrices --------
def pow_tensor(x, cnum):
	# x : B x N x N
	x_ = x.clone()
	xc = [x.unsqueeze(1)]
	for _ in range(cnum-1):
		x_ = torch.bmm(x_, x)
		xc.append(x_.unsqueeze(1))
	xc = torch.cat(xc, dim=1)

	return xc

def glorot(tensor):
	if tensor is not None:
		stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
		tensor.data.uniform_(-stdv, stdv)

def zeros(tensor):
	if tensor is not None:
		tensor.data.fill_(0)

def reset(value):
	if hasattr(value, 'reset_parameters'):
		value.reset_parameters()
	else:
		for child in value.children() if hasattr(value, 'children') else []:
			reset(child)

# -------- GCN layer --------
class DenseGCNConv(torch.nn.Module):
	r"""See :class:`torch_geometric.nn.conv.GCNConv`.
	"""
	def __init__(self, in_channels, out_channels, improved=False, bias=True):
		super(DenseGCNConv, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.improved = improved

		self.weight = torch.nn.Parameter(torch.Tensor(self.in_channels, out_channels))

		if bias:
			self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)

		self.reset_parameters()

	def reset_parameters(self):
		glorot(self.weight)
		zeros(self.bias)


	def forward(self, x, adj, mask=None, add_loop=True):
		r"""
		Args:
			x (Tensor): Node feature tensor :math:`\mathbf{X} \in \mathbb{R}^{B
				\times N \times F}`, with batch-size :math:`B`, (maximum)
				number of nodes :math:`N` for each graph, and feature
				dimension :math:`F`.
			adj (Tensor): Adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B
				\times N \times N}`. The adjacency tensor is broadcastable in
				the batch dimension, resulting in a shared adjacency matrix for
				the complete batch.
			mask (BoolTensor, optional): Mask matrix
				:math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
				the valid nodes for each graph. (default: :obj:`None`)
			add_loop (bool, optional): If set to :obj:`False`, the layer will
				not automatically add self-loops to the adjacency matrices.
				(default: :obj:`True`)
		"""
		x = x.unsqueeze(0) if x.dim() == 2 else x
		adj = adj.unsqueeze(0) if adj.dim() == 2 else adj
		B, N, _ = adj.size()

		if add_loop:
			adj = adj.clone()
			idx = torch.arange(N, dtype=torch.long, device=adj.device)
			adj[:, idx, idx] = 1 if not self.improved else 2

		out = torch.matmul(x, self.weight)
		deg_inv_sqrt = adj.sum(dim=-1).clamp(min=1).pow(-0.5)

		adj = deg_inv_sqrt.unsqueeze(-1) * adj * deg_inv_sqrt.unsqueeze(-2)
		out = torch.matmul(adj, out)

		if self.bias is not None:
			out = out + self.bias

		if mask is not None:
			out = out * mask.view(B, N, 1).to(x.dtype)

		return out


	def __repr__(self):
		return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
								   self.out_channels)

# -------- MLP layer --------
class MLP(torch.nn.Module):
	def __init__(self, num_layers, input_dim, hidden_dim, output_dim, use_bn=False, activate_func=torch.nn.functional.relu):
		"""
			num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
			input_dim: dimensionality of input features
			hidden_dim: dimensionality of hidden units at ALL layers
			output_dim: number of classes for prediction
			num_classes: the number of classes of input, to be treated with different gains and biases,
					(see the definition of class `ConditionalLayer1d`)
		"""

		super(MLP, self).__init__()

		self.linear_or_not = True  # default is linear model
		self.num_layers = num_layers
		self.use_bn = use_bn
		self.activate_func = activate_func

		if num_layers < 1:
			raise ValueError("number of layers should be positive!")
		elif num_layers == 1:
			# Linear model
			self.linear = torch.nn.Linear(input_dim, output_dim)
		else:
			# Multi-layer model
			self.linear_or_not = False
			self.linears = torch.nn.ModuleList()

			self.linears.append(torch.nn.Linear(input_dim, hidden_dim))
			for layer in range(num_layers - 2):
				self.linears.append(torch.nn.Linear(hidden_dim, hidden_dim))
			self.linears.append(torch.nn.Linear(hidden_dim, output_dim))

			if self.use_bn:
				self.batch_norms = torch.nn.ModuleList()
				for layer in range(num_layers - 1):
					self.batch_norms.append(torch.nn.BatchNorm1d(hidden_dim))


	def forward(self, x):
		"""
		:param x: [num_classes * batch_size, N, F_i], batch of node features
			note that in self.cond_layers[layer],
			`x` is splited into `num_classes` groups in dim=0,
			and then treated with different gains and biases
		"""
		if self.linear_or_not:
			# If linear model
			return self.linear(x)
		else:
			# If MLP
			h = x
			for layer in range(self.num_layers - 1):
				h = self.linears[layer](h)
				if self.use_bn:
					h = self.batch_norms[layer](h)
				h = self.activate_func(h)
			return self.linears[self.num_layers - 1](h)

class Attention(torch.nn.Module):

	def __init__(self, in_dim, attn_dim, out_dim, num_heads=4, conv='GCN'):
		super(Attention, self).__init__()
		self.num_heads = num_heads
		self.attn_dim = attn_dim
		self.out_dim = out_dim
		self.conv = conv

		self.gnn_q, self.gnn_k, self.gnn_v = self.get_gnn(in_dim, attn_dim, out_dim, conv)
		self.activation = torch.tanh 
		self.softmax_dim = 2

	def forward(self, x, adj, flags, attention_mask=None):

		if self.conv == 'GCN':
			Q = self.gnn_q(x, adj) 
			K = self.gnn_k(x, adj) 
		else:
			Q = self.gnn_q(x) 
			K = self.gnn_k(x)

		V = self.gnn_v(x, adj) 
		dim_split = self.attn_dim // self.num_heads
		Q_ = torch.cat(Q.split(dim_split, 2), 0)
		K_ = torch.cat(K.split(dim_split, 2), 0)

		if attention_mask is not None:
			attention_mask = torch.cat([attention_mask for _ in range(self.num_heads)], 0)
			attention_score = Q_.bmm(K_.transpose(1,2))/math.sqrt(self.out_dim)
			A = self.activation( attention_mask + attention_score )
		else:
			A = self.activation( Q_.bmm(K_.transpose(1,2))/math.sqrt(self.out_dim) ) # (B x num_heads) x N x N
		
		# -------- (B x num_heads) x N x N --------
		A = A.view(-1, *adj.shape)
		A = A.mean(dim=0)
		A = (A + A.transpose(-1,-2))/2 

		return V, A 

	def get_gnn(self, in_dim, attn_dim, out_dim, conv='GCN'):

		if conv == 'GCN':
			gnn_q = DenseGCNConv(in_dim, attn_dim)
			gnn_k = DenseGCNConv(in_dim, attn_dim)
			gnn_v = DenseGCNConv(in_dim, out_dim)

			return gnn_q, gnn_k, gnn_v

		elif conv == 'MLP':
			num_layers=2
			gnn_q = MLP(num_layers, in_dim, 2*attn_dim, attn_dim, activate_func=torch.tanh)
			gnn_k = MLP(num_layers, in_dim, 2*attn_dim, attn_dim, activate_func=torch.tanh)
			gnn_v = DenseGCNConv(in_dim, out_dim)

			return gnn_q, gnn_k, gnn_v

		else:
			raise NotImplementedError(f'{conv} not implemented.')


# -------- Layer of ScoreNetworkA --------
class AttentionLayer(torch.nn.Module):

	def __init__(self, num_linears, conv_input_dim, attn_dim, conv_output_dim, input_dim, output_dim, 
					num_heads=4, conv='GCN'):

		super(AttentionLayer, self).__init__()

		self.attn = torch.nn.ModuleList()
		for _ in range(input_dim):
			self.attn_dim =  attn_dim 
			self.attn.append(Attention(conv_input_dim, self.attn_dim, conv_output_dim,
										num_heads=num_heads, conv=conv))

		self.hidden_dim = 2*max(input_dim, output_dim)
		self.mlp = MLP(num_linears, 2*input_dim, self.hidden_dim, output_dim, use_bn=False, activate_func=torch.nn.functional.elu)
		self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
									use_bn=False, activate_func=torch.nn.functional.elu)

	def forward(self, x, adj, flags):
		"""
		:param x:  B x N x F_i
		:param adj: B x C_i x N x N
		:return: x_out: B x N x F_o, adj_out: B x C_o x N x N
		"""
		mask_list = []
		x_list = []
		for _ in range(len(self.attn)):
			_x, mask = self.attn[_](x, adj[:,_,:,:], flags)
			mask_list.append(mask.unsqueeze(-1))
			x_list.append(_x)
		x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)), flags)
		x_out = torch.tanh(x_out)

		mlp_in = torch.cat([torch.cat(mask_list, dim=-1), adj.permute(0,2,3,1)], dim=-1)
		shape = mlp_in.shape
		mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
		_adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
		_adj = _adj + _adj.transpose(-1,-2)
		adj_out = mask_adjs(_adj, flags)

		return x_out, adj_out


class BaselineNetworkLayer(torch.nn.Module):

	def __init__(self, num_linears, conv_input_dim, conv_output_dim, input_dim, output_dim, batch_norm=False):

		super(BaselineNetworkLayer, self).__init__()

		self.convs = torch.nn.ModuleList()
		for _ in range(input_dim):
			self.convs.append(DenseGCNConv(conv_input_dim, conv_output_dim))
		self.hidden_dim = max(input_dim, output_dim)
		self.mlp_in_dim = input_dim + 2*conv_output_dim
		self.mlp = MLP(num_linears, self.mlp_in_dim, self.hidden_dim, output_dim, 
							use_bn=False, activate_func=torch.nn.functional.elu)
		self.multi_channel = MLP(2, input_dim*conv_output_dim, self.hidden_dim, conv_output_dim, 
									use_bn=False, activate_func=torch.nn.functional.elu)
		
	def forward(self, x, adj, flags):
	
		x_list = []
		for _ in range(len(self.convs)):
			_x = self.convs[_](x, adj[:,_,:,:])
			x_list.append(_x)
		x_out = mask_x(self.multi_channel(torch.cat(x_list, dim=-1)) , flags)
		x_out = torch.tanh(x_out)

		x_matrix = node_feature_to_matrix(x_out)
		mlp_in = torch.cat([x_matrix, adj.permute(0,2,3,1)], dim=-1)
		shape = mlp_in.shape
		mlp_out = self.mlp(mlp_in.view(-1, shape[-1]))
		_adj = mlp_out.view(shape[0], shape[1], shape[2], -1).permute(0,3,1,2)
		_adj = _adj + _adj.transpose(-1,-2)
		adj_out = mask_adjs(_adj, flags)

		return x_out, adj_out


class BaselineNetwork(torch.nn.Module):

	def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
					c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

		super(BaselineNetwork, self).__init__()

		self.nfeat = max_feat_num
		self.max_node_num = max_node_num
		self.nhid  = nhid
		self.num_layers = num_layers
		self.num_linears = num_linears
		self.c_init = c_init
		self.c_hid = c_hid
		self.c_final = c_final

		self.layers = torch.nn.ModuleList()
		for _ in range(self.num_layers):
			if _==0:
				self.layers.append(BaselineNetworkLayer(self.num_linears, self.nfeat, self.nhid, self.c_init, self.c_hid))

			elif _==self.num_layers-1:
				self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_final))

			else:
				self.layers.append(BaselineNetworkLayer(self.num_linears, self.nhid, self.nhid, self.c_hid, self.c_hid)) 

		self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
		self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
							use_bn=False, activate_func=torch.nn.functional.elu)
		self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
		self.mask.unsqueeze_(0)   

	def forward(self, x, adj, flags=None):

		adjc = pow_tensor(adj, self.c_init)

		adj_list = [adjc]
		for _ in range(self.num_layers):

			x, adjc = self.layers[_](x, adjc, flags)
			adj_list.append(adjc)
		
		adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
		out_shape = adjs.shape[:-1] # B x N x N
		score = self.final(adjs).view(*out_shape)

		self.mask = self.mask.to(score.device)
		score = score * self.mask

		score = mask_adjs(score, flags)

		return score


class ScoreNetworkA(BaselineNetwork):

	def __init__(self, max_feat_num, max_node_num, nhid, num_layers, num_linears, 
					c_init, c_hid, c_final, adim, num_heads=4, conv='GCN'):

		super(ScoreNetworkA, self).__init__(max_feat_num, max_node_num, nhid, num_layers, num_linears, 
											c_init, c_hid, c_final, adim, num_heads=4, conv='GCN')
		
		self.adim = adim
		self.num_heads = num_heads
		self.conv = conv

		self.layers = torch.nn.ModuleList()
		for _ in range(self.num_layers):
			if _==0:
				self.layers.append(AttentionLayer(self.num_linears, self.nfeat, self.nhid, self.nhid, self.c_init, 
													self.c_hid, self.num_heads, self.conv))
			elif _==self.num_layers-1:
				self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
													self.c_final, self.num_heads, self.conv))
			else:
				self.layers.append(AttentionLayer(self.num_linears, self.nhid, self.adim, self.nhid, self.c_hid, 
													self.c_hid, self.num_heads, self.conv))

		self.fdim = self.c_hid*(self.num_layers-1) + self.c_final + self.c_init
		self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=1, 
							use_bn=False, activate_func=torch.nn.functional.elu)
		self.mask = torch.ones([self.max_node_num, self.max_node_num]) - torch.eye(self.max_node_num)
		self.mask.unsqueeze_(0)  

	def forward(self, x, adj, flags):

		adjc = pow_tensor(adj, self.c_init)

		adj_list = [adjc]
		for _ in range(self.num_layers):

			x, adjc = self.layers[_](x, adjc, flags)
			adj_list.append(adjc)
		
		adjs = torch.cat(adj_list, dim=1).permute(0,2,3,1)
		out_shape = adjs.shape[:-1] # B x N x N
		score = self.final(adjs).view(*out_shape)
		
		self.mask = self.mask.to(score.device)
		score = score * self.mask

		score = mask_adjs(score, flags)

		return score

class ScoreNetworkX(torch.nn.Module):

	def __init__(self, max_feat_num, depth, nhid):

		super(ScoreNetworkX, self).__init__()

		self.nfeat = max_feat_num
		self.depth = depth
		self.nhid = nhid

		self.layers = torch.nn.ModuleList()
		for _ in range(self.depth):
			if _ == 0:
				self.layers.append(DenseGCNConv(self.nfeat, self.nhid))
			else:
				self.layers.append(DenseGCNConv(self.nhid, self.nhid))

		self.fdim = self.nfeat + self.depth * self.nhid
		self.final = MLP(num_layers=3, input_dim=self.fdim, hidden_dim=2*self.fdim, output_dim=self.nfeat, 
							use_bn=False, activate_func=torch.nn.functional.elu)

		self.activation = torch.tanh

	def forward(self, x, adj, flags):

		x_list = [x]
		for _ in range(self.depth):
			x = self.layers[_](x, adj)
			x = self.activation(x)
			x_list.append(x)

		xs = torch.cat(x_list, dim=-1) # B x N x (F + num_layers x H)
		out_shape = (adj.shape[0], adj.shape[1], -1)
		x = self.final(xs).view(*out_shape)

		x = mask_x(x, flags)

		return x


class GraphJointNetwork(torch.nn.Module):
	def __init__(
		self, num_tasks, t_limit, max_feat_num=9, max_node_num=38, depth=2,
		nhid=16, num_layers=6, num_linears=3, c_init=2, c_hid=8, c_final=4,
		adim=16, num_heads=4, conv="GCN", time_embed_std=30, time_embed_size=256
	):
		super().__init__()
		
		self.creation_args = locals()
		del self.creation_args["self"]
		del self.creation_args["__class__"]
		self.creation_args = sanitize_sacred_arguments(self.creation_args)


		self.num_tasks = num_tasks
		self.t_limit = t_limit

		self.x_nets = torch.nn.ModuleList([
			ScoreNetworkX(max_feat_num=max_feat_num, depth=depth, nhid=nhid)
			for _ in range(num_tasks)
		])
		self.a_nets = torch.nn.ModuleList([
			ScoreNetworkA(
				max_feat_num=max_feat_num, max_node_num=max_node_num, nhid=nhid,
				num_layers=num_layers, num_linears=num_linears,
				c_init=c_init, c_hid=c_hid, c_final=c_final, adim=adim,
				num_heads=num_heads, conv=conv
			) for _ in range(num_tasks)
		])
		
		# Random embedding layer for time; the random weights are set at the
		# start and are not trainable
		self.time_embed_rand_weights = torch.nn.Parameter(
			torch.randn(time_embed_size // 2) * time_embed_std,
			requires_grad=False
		)
		
		self.time_embedder = torch.nn.Linear(time_embed_size, 1)
		
		# Activation functions
		self.swish = lambda x: x * torch.sigmoid(x)
		
	def forward(self, xt, t, node_flags, task_inds=None):
		"""
		Forward pass of the network.
		Arguments:
			`xt`: a B x M x (M + D) matrix, where the first B x M x M is the
				adjacency matrices, and the next B x M x D is the node-feature
				matrices
			`t`: B-tensor containing the times to train the network for each
				input
			`node_flags`: a B x M binary tensor denoting which nodes of each
				graph actually are present, and which are just there for padding
			`task_inds`: an iterable of task indices to generate predictions
				for; if specified, the output tensor will be
				B x `len(task_inds)` x D instead of B x T x D
		Returns a B x T x M x (M + D) tensor which consists of the prediction.
		"""
		# Get the time embeddings for `t`
		# We had sampled a vector z from a zero-mean Gaussian of fixed variance
		# We embed the time as cos((t/T) * (pi / 2) * z) and
		# sin((t/T) * (pi / 2) * z)
		time_embed_args = (t[:, None] / self.t_limit) * \
			self.time_embed_rand_weights[None, :] * (np.pi / 2)
		# Shape: B x (E / 2)

		time_embed = self.swish(torch.cat([
			torch.sin(time_embed_args), torch.cos(time_embed_args)
		], dim=1))
		# Shape: B x E
		time_scalar = self.time_embedder(time_embed)[:, None, None]
		# Shape: B x 1 x 1 x 1
		
		adj, x = xt[:, :, :xt.shape[1]], xt[:, :, xt.shape[1]:]
		x_preds = [
			self.x_nets[i](x, adj, node_flags) for i in
			(range(self.num_tasks) if task_inds is None else task_inds)
		]
		a_preds = [
			self.a_nets[i](x, adj, node_flags) for i in
			(range(self.num_tasks) if task_inds is None else task_inds)
		]
		x_preds, a_preds = \
			torch.stack(x_preds, dim=1), torch.stack(a_preds, dim=1)
		x_preds_scaled, a_preds_scaled = \
			x_preds * time_scalar, a_preds * time_scalar

		return torch.cat([a_preds_scaled, x_preds_scaled], dim=3)

	def loss(self, pred_values, true_values, task_inds, weights=None):
		"""
		Computes the loss of the neural network.
		Arguments:
			`pred_values`: a B x T x M x (M + D) tensor of predictions from the
				network
			`true_values`: a B x M x (M + D) tensor of true values to predict
			`task_inds`: a B-tensor of indices (0 through T - 1) that determine
				which predicted values to compare to true values
			`weights`: if provided, a tensor broadcastable with B x M x (M + D)
				to weight the squared error by, prior to summing or averaging
				across dimensions
		Returns a scalar loss of mean-squared-error values, summed across the
		M x (M + D) dimensions and averaged across the batch dimension.
		"""
		pred_values_subset = torch.stack([
			pred_values[i, task_inds[i]] for i in range(len(task_inds))
		])	# Shape: B x D
		
		# Compute loss as MSE
		squared_error = torch.square(true_values - pred_values_subset)
		if weights is not None:
			squared_error = squared_error / weights
			
		return torch.mean(torch.sum(
			squared_error,
			dim=tuple(range(1, len(squared_error.shape)))
		))

if __name__ == "__main__":
	model = GraphJointNetwork(num_tasks=3, t_limit=1)
	x = torch.ones((32, 38, 9))
	adj = torch.ones((32, 38, 38))
	xt = torch.cat([x, adj], dim=2)
	t = torch.zeros(xt.shape[0])
	pred = model(xt, t)
	loss = model.loss(pred, xt, torch.randint(3, size=(xt.shape[0],)))
	loss.backward()
