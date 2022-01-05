import torch.utils.data as data
from PIL import Image
import numpy as np
from torchvision.datasets import MNIST, CIFAR10
import torchvision.transforms as transforms

def load_mnist_data(datadir):

    transform = transforms.Compose([transforms.ToTensor()])

    mnist_train_ds = MNIST_truncated(datadir, train=True, download=True, transform=transform)
    mnist_test_ds = MNIST_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = mnist_train_ds.data, mnist_train_ds.target
    X_test, y_test = mnist_test_ds.data, mnist_test_ds.target

    X_train = X_train.data.numpy()
    y_train = y_train.data.numpy()
    X_test = X_test.data.numpy()
    y_test = y_test.data.numpy()

    return (X_train, y_train, X_test, y_test)

def load_cifar10_data(datadir):

    transform = transforms.Compose([transforms.ToTensor(),
								   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    cifar10_train_ds = CIFAR10_truncated(datadir, train=True, download=True, transform=transform)
    cifar10_test_ds = CIFAR10_truncated(datadir, train=False, download=True, transform=transform)

    X_train, y_train = cifar10_train_ds.data, cifar10_train_ds.target
    X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target

    return (X_train, y_train, X_test, y_test)

class MNIST_truncated(data.Dataset):

	def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

		self.root = root
		self.dataidxs = dataidxs
		self.train = train
		self.transform	 = transform
		self.target_transform = target_transform
		self.download = download

		self.data, self.target = self.__build_truncated_dataset__()

	def __build_truncated_dataset__(self):

		mnist_dataobj = MNIST(self.root, self.train, self.transform, self.target_transform, self.download)

		data = mnist_dataobj.data
		target = mnist_dataobj.targets

		if self.dataidxs is not None:
			data = data[self.dataidxs]
			target = target[self.dataidxs]

		return data, target

	def __getitem__(self, index):
		"""
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
		img, target = self.data[index], self.target[index]

		# doing this so that it is consistent with all other datasets
		# to return a PIL Image
		img = Image.fromarray(img.numpy(), mode='L')

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)


class CIFAR10_truncated(data.Dataset):

	def __init__(self, root, dataidxs=None, train=True, transform=None, target_transform=None, download=False):

		self.root = root
		self.dataidxs = dataidxs
		self.train = train
		self.transform = transform
		self.target_transform = target_transform
		self.download = download

		self.data, self.target = self.__build_truncated_dataset__()

	def __build_truncated_dataset__(self):

		cifar_dataobj = CIFAR10(self.root, self.train, self.transform, self.target_transform, self.download)

		data = np.array(cifar_dataobj.data)
		target = np.array(cifar_dataobj.targets)

		if self.dataidxs is not None:
			data = data[self.dataidxs]
			target = target[self.dataidxs]

		return data, target

	def __getitem__(self, index):
		"""
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
		img, target = self.data[index], self.target[index]

		if self.transform is not None:
			img = self.transform(img)

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return len(self.data)