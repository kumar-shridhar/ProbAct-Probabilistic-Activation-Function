'''VGG11/13/16/19 in Pytorch.'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import normal
import numpy as np
from torch.nn import Parameter
from .ProbAct import TrainableSigma

device = torch.device("cuda:0")


class VGG_ProbAct_Trainable(nn.Module):

    def __init__(self, nclass, img_width=32):
        super(VGG_ProbAct_Trainable, self).__init__()


        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64))

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), 
            nn.BatchNorm2d(128))

        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128))

        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.conv6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.conv7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256))

        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512))

        self.conv9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512))

        self.conv10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512))

        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512))

        self.conv12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512))

        self.conv13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512))

        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # self.dropout = nn.Dropout(0.5)

        self.ProbActAF = TrainableSigma()


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


    def forward(self, x):

        out = self.conv1(x)
        out = self.ProbActAF(out)

        out = self.conv2(out)
        out = self.ProbActAF(out)
        out = self.pool1(out)


        out = self.conv3(out)
        out = self.ProbActAF(out)

        out = self.conv4(out)
        out = self.ProbActAF(out)
        out = self.pool2(out)


        out = self.conv5(out)
        out = self.ProbActAF(out)

        out = self.conv6(out)
        out = self.ProbActAF(out)

        out = self.conv7(out)
        out = self.ProbActAF(out)
        out = self.pool3(out)


        out = self.conv8(out)
        out = self.ProbActAF(out)

        out = self.conv9(out)
        out = self.ProbActAF(out)

        out = self.conv10(out)
        out = self.ProbActAF(out)
        out = self.pool4(out)


        out = self.conv11(out)
        out = self.ProbActAF(out)

        out = self.conv12(out)
        out = self.ProbActAF(out)

        out = self.conv13(out)
        out = self.ProbActAF(out)
        out = self.pool5(out)


        out = self.globalAvgpool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
