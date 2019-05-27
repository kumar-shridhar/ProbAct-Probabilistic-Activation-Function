import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


device = torch.device("cuda:0")

class TrainableSigma(nn.Module):


	def __init__(self, num_parameters=1, init=0):
		self.num_parameters = num_parameters
		super(TrainableSigma, self).__init__()
		self.weight = Parameter(torch.Tensor(num_parameters).fill_(init))

	def forward(self, input):

		mu = input

		if mu.is_cuda:
			eps = torch.cuda.FloatTensor(mu.size()).normal_(mean = 0, std = 1)
		else:
			eps = torch.FloatTensor(mu.size()).normal_(mean = 0, std = 1)

		return F.relu(mu) + self.weight * eps

	def extra_repr(self):
	    return 'num_parameters={}'.format(self.num_parameters)
