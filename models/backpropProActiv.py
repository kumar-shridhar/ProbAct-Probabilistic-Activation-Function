import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
import numpy as np
from torch.nn import Parameter

class BackpropProActiv(torch.autograd.Function):

    @staticmethod
    def forward(ctx, mu, sigma, eps):
    	
    	eps.normal_()
    	ctx.save_for_backward(mu, sigma, eps)

    	return mu + torch.exp(sigma) * eps
		
    def backward(ctx, grad_output):
        mu, sigma, eps = ctx.saved_tensors
        grad_mu = grad_sigma = grad_eps = None
        tmp = torch.exp(sigma)
        if ctx.needs_input_grad[0]:
            grad_mu = grad_output + mu
        if ctx.needs_input_grad[1]:
            grad_sigma = grad_output*tmp*eps - 1 + tmp*tmp
        return grad_mu, grad_sigma, grad_eps

backpropWeight = BackpropProActiv.apply
