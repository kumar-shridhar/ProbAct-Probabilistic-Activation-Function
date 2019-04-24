import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
import numpy as np
from torch.nn import Parameter
from .backpropProActiv import BackpropProActiv

class LearnableProActiv(nn.Module):
    def __init__(self, in_features):
        super(LearnableProActiv, self).__init__()
        self.in_features = in_features
        # self.N = N
        self.initial_sigma = 0.05
        self.mu_weight = in_features
        # self.mu_weight = Parameter(torch.cuda.FloatTensor(in_features))
        self.sigma_weight = Parameter(torch.cuda.FloatTensor(in_features))
        self.register_buffer('eps_weight', torch.cuda.FloatTensor(in_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mu_weight.size(1))
        # self.mu_weight.data.uniform_(-stdv, stdv)
        self.sigma_weight.data.fill_(self.initial_sigma)
        self.eps_weight.data.zero_()

    def forward(self, input):

    	weight = BackpropProActiv.backpropWeight(self.mu_weight, self.sigma_weight, self.eps_weight)

    	return weight


class FixedSigmaProActiv(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        ctx.save_for_backward(input)
        input = input.clamp(min=0)

        dist = normal.Normal(loc = input, scale = 1)
        return dist.sample((1,))

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
