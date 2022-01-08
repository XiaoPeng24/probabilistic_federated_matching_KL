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

class LinMod(nn.Linear):
    '''Linear modules with or without batchnorm, all in one module
    '''
    def __init__(self, n_inputs, n_outputs, bias=False, batchnorm=False):
        super(LinMod, self).__init__(n_inputs, n_outputs, bias=bias)
        if batchnorm:
            self.bn = nn.BatchNorm1d(n_outputs, affine=True)

        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.batchnorm = batchnorm
        self.bias_flag = bias

    def forward(self, inputs):
        outputs = super(LinMod, self).forward(inputs)
        if hasattr(self, 'bn'):
            outputs = self.bn(outputs)
        return outputs

    def extra_repr(self):
        return '{n_inputs}, {n_outputs}, bias={bias_flag}, batchnorm={batchnorm}'.format(**self.__dict__)

class FFNet(nn.Module):
    '''Feed-forward all-to-all connected network
    '''
    def __init__(self, n_inputs, n_hiddens, n_hidden_layers=2, n_outputs=10, nlin=nn.ReLU, bias=False, batchnorm=False):
        super(FFNet, self).__init__()

        self.features = ()  # Skip convolutional features

        self.classifier = nn.Sequential(LinMod(n_inputs, n_hiddens, bias=bias, batchnorm=batchnorm), nlin())
        for i in range(n_hidden_layers - 1):
            self.classifier.add_module(str(2 * i + 2), LinMod(n_hiddens, n_hiddens, bias=bias, batchnorm=batchnorm))
            self.classifier.add_module(str(2 * i + 3), nlin())
        self.classifier.add_module(str(len(self.classifier)), nn.Linear(n_hiddens, n_outputs))

        self.batchnorm = batchnorm
        self.n_inputs = n_inputs
        self.n_hiddens = n_hiddens
        self.n_outputs = n_outputs

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class SimpleCNN(nn.Module):
	def __init__(self, input_dim, hidden_dims, output_dim=10):
		super(SimpleCNN, self).__init__()
		self.features = nn.Sequential(
            nn.Conv2d(3, 6, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

		# for now, we hard coded this network
		# i.e. we fix the number of hidden layers i.e. 2 layers
		self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, 16 * 5 * 5)
		x = self.classifier(x)

		return x

class SimpleCNNContainer(nn.Module):
	def __init__(self, input_channel, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
		super(SimpleCNNContainer, self).__init__()
		'''
        A testing cnn container, which allows initializing a CNN with given dims
        num_filters (list) :: number of convolution filters
        hidden_dims (list) :: number of neurons in hidden layers
        Assumptions:
        i) we use only two conv layers and three hidden layers (including the output layer)
        ii) kernel size in the two conv layers are identical
        '''
		self.features = nn.Sequential(
            nn.Conv2d(input_channel, num_filters[0], kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(num_filters[0], num_filters[1], kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        # for now, we hard coded this network
        # i.e. we fix the number of hidden layers i.e. 2 layers
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Linear(hidden_dims[1], output_dim),
        )

	def forward(self, x):
		x = self.features(x)
		x = x.view(-1, x.size()[1] * x.size()[2] * x.size()[3])
		x = self.classifier(x)
		return x

### Moderate size of CNN for CIFAR-10 dataset
class ModerateCNN(nn.Module):
    def __init__(self):
        super(ModerateCNN, self).__init__()
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            #nn.Linear(4096, 1024),
            nn.Linear(4096, 512),
            nn.ReLU(inplace=True),
            #nn.Linear(1024, 512),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
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

class ModerateCNNContainer(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, input_dim, hidden_dims, output_dim=10):
        super(ModerateCNNContainer, self).__init__()

        ##
        self.conv_layer = nn.Sequential(
            # Conv Layer block 1
            nn.Conv2d(in_channels=input_channels, out_channels=num_filters[0], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[0], out_channels=num_filters[1], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=num_filters[1], out_channels=num_filters[2], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[2], out_channels=num_filters[3], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),

            # Conv Layer block 3
            nn.Conv2d(in_channels=num_filters[3], out_channels=num_filters[4], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_filters[4], out_channels=num_filters[5], kernel_size=kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(input_dim, hidden_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dims[1], output_dim)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return x

    def forward_conv(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        return x

class LeNet(nn.Module):
    '''Based on https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
    '''
    def __init__(self, num_input_channels=3, num_classes=10, window_size=32, bias=True):
        super(LeNet, self).__init__()
        self.bias = bias
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Conv2d(num_input_channels, 6, 5, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Conv2d(6, 16, 5, bias=bias),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(16 * int((int((window_size - 4) / 2) - 4) / 2)**2, 120, bias=bias),
            nn.ReLU(),
            nn.Linear(120, 84, bias=bias),
            nn.ReLU(),
            nn.Linear(84, num_classes, bias=bias),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x