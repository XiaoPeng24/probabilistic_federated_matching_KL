import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

class FcNet(nn.Module):
	"""
	Fully connected network for MNIST classification
	"""

	def __init__(self, input_dim, hidden_dims, output_dim, dropout_p=0.0):

		super().__init__()

		self.input_dim = input_dim
		self.hidden_dims = hidden_dims
		self.output_dim = output_dim
		self.dropout_p = dropout_p

		self.dims = [self.input_dim]
		self.dims.extend(hidden_dims)
		self.dims.append(self.output_dim)

		self.layers = nn.ModuleList([])

		self.cat = False

		for i in range(len(self.dims)-1):
			ip_dim = self.dims[i]
			op_dim = self.dims[i+1]
			self.layers.append(
				nn.Linear(ip_dim, op_dim, bias=True)
			)

		self.__init_net_weights__()

	def __init_net_weights__(self):

		for m in self.layers:
			m.weight.data.normal_(0.0, 0.1)
			m.bias.data.fill_(0.1)

	def forward(self, x):

		x = x.view(-1, self.input_dim)

		if self.cat:
			x = torch.cat((x, torch.ones(x.shape[0], 1).to(x.device)), 1)

		for i, layer in enumerate(self.layers):
			x = layer(x)

			# Do not apply ReLU on the final layer
			if i < (len(self.layers) - 1):
				x = F.relu(x)

			if i < (len(self.layers) - 1):		# No dropout on output layer
				x = F.dropout(x, p=self.dropout_p, training=self.training)

		return x

def cat_w_b(net, device="cpu"):
	"""
	Concatenate the weight and bias for skip matching
	"""
	#pdb.set_trace()
	for l in range(len(net._modules['layers'])):
		weight = net._modules['layers'][l].weight.data
		bias = net._modules['layers'][l].bias.data.reshape(-1, 1)
		w_b = torch.cat((weight, bias), 1)
		if l < len(net._modules['layers'])-1:
			exp_dim = torch.zeros(w_b.shape[1]).reshape(1, -1)
			exp_dim.scatter_add_(1, torch.tensor([[exp_dim.shape[1]-1]]), torch.ones(1,1))
			exp_dim = exp_dim.to(device)
			w_b = torch.cat((w_b, exp_dim), 0)
		net._modules['layers'][l].weight.data = w_b
		net._modules['layers'][l].bias = None

	net.cat = True

