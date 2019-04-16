'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
import numpy as np

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
            nn.ReLU(inplace=True))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True))

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True))

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True))

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.dropout = nn.Dropout(0.4)

        self.img_width = img_width
        # self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.globalAvgpool = nn.AvgPool2d(kernel_size=1, stride=1)

        self.classifier = nn.Sequential(
            # nn.Linear(512 * 7 * 7, 4096),
            # nn.ReLU(True),
            # nn.Dropout(),
            # nn.Linear(4096, 512),
            # nn.ReLU(True),
            # nn.Dropout(),
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

    def distributed_activation(self, x):

    	mu = x
    	logvar = 0.05
    	shape = mu.size()

    	m = normal.Normal(0, 0.1)
    	eps = m.sample((10,)).mean(0)

    	# if mu.is_cuda:
    	# 	eps = torch.cuda.FloatTensor(shape).normal_(mean = 0, std = 0.1)
    	# else:
    	# 	eps = torch.FloatTensor(shape).normal_(mean = 0, std = 0.1)

    	return mu + np.exp(0.5 * logvar) * eps
        


    def forward(self, x):

        out = self.conv1(x)
        out = self.distributed_activation(out)
        # out = self.dropout(out)
        out = self.conv2(out)
        out = self.distributed_activation(out)
        out = self.pool1(out)

        out = self.conv3(out)
        out = self.distributed_activation(out)
        # out = self.dropout(out)
        out = self.conv4(out)
        out = self.distributed_activation(out)
        out = self.pool2(out)

        out = self.conv5(out)
        out = self.distributed_activation(out)
        # out = self.dropout(out)
        out = self.conv6(out)
        out = self.distributed_activation(out)
        # out = self.dropout(out)
        out = self.conv7(out)
        out = self.distributed_activation(out)
        out = self.pool3(out)

        out = self.conv8(out)
        out = self.distributed_activation(out)
        # out = self.dropout(out)
        out = self.conv9(out)
        out = self.distributed_activation(out)
        # out = self.dropout(out)
        out = self.conv10(out)
        out = self.distributed_activation(out)
        out = self.pool4(out)

        out = self.conv11(out)
        out = self.distributed_activation(out)
        # out = self.dropout(out)
        out = self.conv12(out)
        out = self.distributed_activation(out)
        # out = self.dropout(out)
        out = self.conv13(out)
        out = self.distributed_activation(out)
        out = self.pool5(out)

        # out = self.dropout(out)
        out = self.globalAvgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out



# class VGG_Dist(nn.Module):
#     def __init__(self, vgg_name, nclass, img_width=32):
#         super(VGG_Dist, self).__init__()
#         self.img_width = img_width
#         self.features = self._make_layers(cfg[vgg_name])
#         self.classifier = nn.Linear(512, nclass)

#     def distributed_activation(self, x):
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
#                            self.distributed_activation(x)]
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

