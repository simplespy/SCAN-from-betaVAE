from torch import nn, optim
import torch
from torch.autograd import Variable


class DAE(nn.Module):
	def __init__(self):
		super(DAE, self).__init__()
		#(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
		#80*80
		self.conv = nn.Sequential(
			nn.Conv2d( 3, 32, 4, 2, 1),
			nn.ELU(),
			nn.Conv2d(32, 32, 4, 2, 1),
			nn.ELU(),
			nn.Conv2d(32, 64, 4, 2, 1),
			nn.ELU(),
			nn.Conv2d(64, 64, 4, 2, 1),
			nn.ELU())

		self.fc1 = nn.Linear(5 * 5 * 64, 100)
		self.fc2 = nn.Linear(100, 5 * 5 * 64)

		self.deconv = nn.Sequential(
			nn.ConvTranspose2d(64, 64, 4, 2, 1),
			nn.ELU(),
			nn.ConvTranspose2d(64, 32, 4, 2, 1),
			nn.ELU(),
			nn.ConvTranspose2d(32, 32, 4, 2, 1),
			nn.ELU(),
			nn.ConvTranspose2d(32, 3, 4, 2, 1),
			nn.Sigmoid())

		self.elu = nn.ELU()
		self.tanh = nn.Tanh()
	def encode(self, x):
		h = self.conv(x)
		z = self.tanh(self.fc1(h.view(-1, 5*5*64)))
		return z
	def decode(self, z):
		h = self.elu(self.fc2(z)).view(-1,64,5,5)
		out = self.deconv(h)
		return out
	def forward(self, x):
		z = self.encode(x)
		out = self.decode(z)
		return out
	def compute_loss(self, x_org, x_out):
		delta = x_org - x_out
		reconstr_loss = 0.5 * torch.sum(torch.mul(delta, delta))
		return reconstr_loss


class VAE(nn.Module):
	def __init__(self):
		super(VAE, self).__init__()

		self.conv = nn.Sequential(
			nn.Conv2d( 3, 32, 4, 2, 1),
			nn.ReLU(),
			nn.Conv2d(32, 32, 4, 2, 1),
			nn.ReLU(),
			nn.Conv2d(32, 64, 4, 2, 1),
			nn.ReLU(),
			nn.Conv2d(64, 64, 4, 2, 1),
			nn.ReLU())

		self.fc1 = nn.Linear(5 * 5 * 64, 256)
		self.fc21 = nn.Linear(256, 32)
		self.fc22 = nn.Linear(256, 32)
		self.fc3 = nn.Linear(32, 256)
		self.fc4 = nn.Linear(256, 5 * 5 * 64)

		self.dconv = nn.Sequential(
			nn.ConvTranspose2d(64, 64, 4, 2, 1),
			nn.ReLU(),
			nn.ConvTranspose2d(64, 32, 4, 2, 1),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 32, 4, 2, 1),
			nn.ReLU(),
			nn.ConvTranspose2d(32, 3, 4, 2, 1),
			nn.Sigmoid())

		self.relu = nn.ReLU()

	def encode(self, x):
		x = self.conv(x).view(-1, 5 * 5 * 64)
		h = self.relu(self.fc1(x))
		return self.fc21(h), self.fc22(h)

	def reparameterize(self, mu, logvar):
		std = logvar.mul(0.5).exp_()
		eps = Variable(std.data.new(std.size()).normal_())
		return eps.mul(std).add_(mu)

	def decode(self, z):
		x = self.relu(self.fc3(z))
		x = self.relu(self.fc4(x)).view(-1,64,5,5)
		x = self.dconv(x)
		return x

	def forward(self, x):
		mu, logvar = self.encode(x)
		z = self.reparameterize(mu, logvar)
		return self.decode(z), mu, logvar

	def compute_loss(self, x, x_out, mu, logvar, dae, beta=53.0):
		z_d = dae.encode(x)
		z_out_d = dae.encode(x_out)
		delta = z_d - z_out_d
		L2_loss = 0.5 * torch.sum(torch.mul(delta, delta))
		KLD_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
		return L2_loss, beta * KLD_loss

class SCAN(nn.Module):
	def __init__(self):
		super(SCAN, self).__init__()
		


