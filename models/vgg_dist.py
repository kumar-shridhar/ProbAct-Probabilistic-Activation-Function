'''VGG11/13/16/19 in Pytorch.'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
import numpy as np
from torch.nn import Parameter
from .proactiv import FixedSigmaProActiv

# from .proactiv import LearnableProActiv
device = torch.device("cuda:0")

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG_Dist(nn.Module):

    def __init__(self, nclass, img_width=32):
        super(VGG_Dist, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.dropout = nn.Dropout(0.4)

        # self.proactiv = ProActiv.apply

        # ProActiv.LearnableProActLearnableProActiv()

        self.img_width = img_width
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.globalAvgpool = nn.AvgPool2d(kernel_size=1, stride=1)

        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            # nn.ReLU(True),
            nn.Dropout(),
            # nn.Linear(4096, 512),
            # nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, nclass),
        )

        # layers = [self.conv1, self.batch_norm1, self.relu1, self.dist1, self.conv2, self.batch_norm2, self.relu2, self.dist2, self.pool1, 
        # self.conv3, self.batch_norm3, self.relu3, self.dist3, self.conv4, self.batch_norm4, self.relu4, self.dist4, self.pool2, 
        # self.conv5, self.batch_norm5, self.relu5, self.dist5, self.conv6, self.batch_norm6, self.relu6, self.dist6, self.conv7, self.batch_norm7, self.relu7, self.dist7, self.pool3, 
        # self.conv8, self.batch_norm8, self.relu8, self.dist8, self.conv9, self.batch_norm9, self.relu9, self.dist9, self.conv10, self.batch_norm10, self.relu10, self.dist10, self.pool4,
        # self.conv11, self.batch_norm11, self.relu11, self.dist11, self.conv12, self.batch_norm12, self.relu12, self.dist12, self.conv13, self.batch_norm13, self.relu13, self.dist13, self.pool5, 
        # self.globalAvgpool, self.flatten, self.classifier
        # ]

        #self.layers = nn.ModuleList(layers)

    def non_trainable_sigma(self, x):

    	mu = x
    	shape = mu.size()

    	# m = normal.Normal(0, 1)
    	# eps = m.sample((1,))

    	if mu.is_cuda:
    	 	eps = torch.cuda.FloatTensor(shape).normal_(mean = 0, std = 0.05)
    	else:
    	 	eps = torch.FloatTensor(shape).normal_(mean = 0, std = 0.05)

    	return mu + eps

    def trainable_sigma(self,x):

    	mu = x
    	shape = mu.size()
    	alpha = 1
    	beta = 0.05

    	w = torch.randn(x.size(), device=device, dtype=torch.float, requires_grad=True)

    	if mu.is_cuda:
    	 	eps = torch.cuda.FloatTensor(shape).normal_(mean = 0, std = 1)
    	else:
    	 	eps = torch.FloatTensor(shape).normal_(mean = 0, std = 1)

    	return x + (alpha * F.sigmoid(w) + beta) * eps


    def sigmoid_x(self,x):

    	mu = x
    	shape = mu.size()
    	alpha = 1
    	beta = 0.05

    	if mu.is_cuda:
    	 	eps = torch.cuda.FloatTensor(shape).normal_(mean = 0, std = 1)
    	else:
    	 	eps = torch.FloatTensor(shape).normal_(mean = 0, std = 1)

    	return x + (alpha * F.sigmoid(x) + beta) * eps

    def forward(self, x):

        out = self.conv1(x)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        out = self.pool1(out)
        # out = self.dropout(out)

        out = self.conv3(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        # out = self.dropout(out)
        out = self.conv4(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        out = self.pool2(out)
        # out = self.dropout(out)

        out = self.conv5(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        # out = self.dropout(out)
        out = self.conv6(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        # out = self.dropout(out)
        out = self.conv7(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        out = self.pool3(out)
        # out = self.dropout(out)

        out = self.conv8(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        # out = self.dropout(out)
        out = self.conv9(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        # out = self.dropout(out)
        out = self.conv10(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        out = self.pool4(out)
        # out = self.dropout(out)

        out = self.conv11(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        # out = self.dropout(out)
        out = self.conv12(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        # out = self.dropout(out)
        out = self.conv13(out)
        # out = LearnableProActiv(out)
        out = self.trainable_sigma(out)
        out = self.pool5(out)

        # out = self.dropout(out)
        out = self.globalAvgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out



# class LearnableProActiv(torch.autograd.Function):
#     """
#     We can implement our own custom autograd Functions by subclassing
#     torch.autograd.Function and implementing the forward and backward passes
#     which operate on Tensors.
#     """

#     @staticmethod
#     def forward(ctx, input):
#         """
#         In the forward pass we receive a Tensor containing the input and return
#         a Tensor containing the output. ctx is a context object that can be used
#         to stash information for backward computation. You can cache arbitrary
#         objects for use in the backward pass using the ctx.save_for_backward method.
#         """
#         ctx.save_for_backward(input)
#         input = input.clamp(min=0)

#         dist = normal.Normal(loc = input, scale = 1)
#         return dist.sample((25,)).mean(0)

#     @staticmethod
#     def backward(ctx, grad_output):
#         """
#         In the backward pass we receive a Tensor containing the gradient of the loss
#         with respect to the output, and we need to compute the gradient of the loss
#         with respect to the input.
#         """
#         input, = ctx.saved_tensors
#         grad_input = grad_output.clone()
#         grad_input[input < 0] = 0
#         return grad_input

# class LearnableProActiv(nn.Module):
#     def __init__(self, in_features):
     # LearnableLearnableProActiv, self).__init__()
#         self.in_features = in_features
#         self.initial_sigma = 0.05
#         self.mu_weight = Parameter(torch.Tensor(in_features))
#         self.sigma_weight = Parameter(torch.Tensor(in_features))
#         self.register_buffer('eps_weight', torch.Tensor(in_features))
#         self.reset_parameters()

#     def reset_parameters(self):
#         stdv = 1. / math.sqrt(self.mu_weight.size(1))
#         self.mu_weight.data.fill_(self.in_features)
#         self.sigma_weight.data.fill_(self.initial_sigma)
#         self.eps_weight.data.zero_()

#     def forward(self, input):
#         sig_weight = torch.exp(self.sigma_weight)

#         out = self.mu_weight + sig_weight * self.eps_weight.normal_()

#         return out





# class VGG_Dist(nn.Module):
#     def __init__(self, vgg_name, nclass, img_width=32):
#         super(VGG_Dist, self).__init__()
#         self.img_width = img_width
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, nclass)

    # def distributed_activation(self, x):
#         m = normal.Normal(x, 0.05)
#         x = m.sample((5,)).mean(0)
        
#         return x

#     def forward(self, x):
#         out = self.features(x)
#         out = out.view(out.size(0), -1)
#         out = self.classifier(out)
#         return out, None # return None, to make it compatible with VGG_noise

#     def _make_layers(self, cfg):
#         layers = []
#         in_channels = 3
#         width = self.img_width
#         for x in cfg:
#             if x == 'M':
#                 layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
#                 width = width // 2
#             else:
#                 layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
#                            nn.BatchNorm2d(x),
#                            nn.ReLU(inplace=False),
                           # self.distributed_activation(x)]
#                 in_channels = x
#         layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
#         return nn.ModuleList(layers)





# class Normal(Distribution):
#     # scalar version
#     def __init__(self, mu, logvar):
#         self.mu = mu
#         self.logvar = logvar
#         self.shape = mu.size()

#         super(Normal, self).__init__()

#     def logpdf(self, x):
#         c = - float(0.5 * math.log(2 * math.pi))
#         return c - 0.5 * self.logvar - (x - self.mu).pow(2) / (2 * torch.exp(self.logvar))

#     def pdf(self, x):
#         return torch.exp(self.logpdf(x))

#     def sample(self):
#         if self.mu.is_cuda:
#             eps = torch.cuda.FloatTensor(self.shape).normal_()
#         else:
#             eps = torch.FloatTensor(self.shape).normal_()
#         # local reparameterization trick
#         return self.mu + torch.exp(0.5 * self.logvar) * eps

#     def entropy(self):
#         return 0.5 * math.log(2. * math.pi * math.e) + 0.5 * self.logvar

