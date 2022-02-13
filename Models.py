import math
import torch
#import itertools
#import torchvision

import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3.0, inplace=self.inplace) / 6.0


class eSEModule(nn.Module):
    def __init__(self, channel, reduction=3):
        super(eSEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(channel, channel, kernel_size=1, padding=0)
        self.hsigmoid = Hsigmoid()

    def forward(self, x):
        input = x
        x = self.avg_pool(x)
        x = self.fc(x)
        x = self.hsigmoid(x)
        return input * x

class DCSCNet(nn.Module):
    def __init__(self,k,d,in_channels=None,class_num=None):
        super(DCSCNet, self).__init__()

        self.k = k
        self.d = d
        self.in_channels = in_channels
        self.dcsc_ini_channels = 16
        self.ini_channels = 16
        self.class_num = class_num
        
        # convolutional filters
        self.filter0 = nn.Parameter(torch.randn(16,self.in_channels,1700), requires_grad=True)
        self.filters1 = nn.ParameterList(
            [nn.Parameter(torch.randn(self.k,self.dcsc_ini_channels+self.k*i,3), requires_grad=True) for i in
             range(self.d)])

        self.b0 = nn.Parameter(torch.zeros(1,16,1), requires_grad=True)
        self.b1 = nn.ParameterList(
            [nn.Parameter(torch.zeros(1,self.dcsc_ini_channels+self.k+self.k*i,1),requires_grad=True) for i in
             range(self.d)])

        self.bn0 = nn.BatchNorm1d(16, affine=True).cuda()
        self.bn1 = [nn.BatchNorm1d(self.dcsc_ini_channels+(i+1)*self.k, affine=True).cuda() for i in
                    range(self.d)]
        self.c1 = nn.ParameterList(
            [nn.Parameter(torch.ones(1, 1, 1), requires_grad=True) for i in range(self.d)])
        
        # classifier
        self.fc1 = nn.Linear(318*(16+self.k*self.d),128)
        self.fc2 = nn.Linear(128, self.class_num)
        self.ese = eSEModule(16+self.k*self.d)
        self.dropout = nn.Dropout(p=0.25)
        
        # toneburst
        fs=10.e+6
        Cyc=5
        N=1700
        Time = np.arange(0,N-1)/fs
        Toneburst = np.zeros((16,N))
        fc=40.e+3
        LenT = int(np.floor(fs/fc*Cyc))
        tem1 = np.sin(2*math.pi*fc*Time[1:LenT])
        tem2 = 1-np.cos(2*math.pi*fc*Time[1:LenT]/Cyc)
        for i in range(5):
            Toneburst[i,100*i+1:100*i+LenT] = 1/2*np.multiply(tem1,tem2)
        fc=60.e+3
        LenT = int(np.floor(fs/fc*Cyc))
        tem1 = np.sin(2*math.pi*fc*Time[1:LenT])
        tem2 = 1-np.cos(2*math.pi*fc*Time[1:LenT]/Cyc)
        for i in range(5):
            Toneburst[i+5,200*i+1:200*i+LenT] = 1/2*np.multiply(tem1,tem2)
        fc=80.e+3
        LenT = int(np.floor(fs/fc*Cyc))
        tem1 = np.sin(2*math.pi*fc*Time[1:LenT])
        tem2 = 1-np.cos(2*math.pi*fc*Time[1:LenT]/Cyc)
        for i in range(5):
            Toneburst[i+10,250*i+1:250*i+LenT] = 1/2*np.multiply(tem1,tem2)
        Toneburst[15,900+1:900+LenT] = 1/2*np.multiply(tem1,tem2)
        noise = np.random.random([16,1700])/10
        Toneburst += noise
        temp1=np.expand_dims(Toneburst,axis=1)
        W1_i=torch.from_numpy(temp1).float()
        
        # initialization
        for i in range(self.d):
            self.filters1[i].data = .1 / np.sqrt((self.ini_channels + self.k * i) * 9) * self.filters1[i].data * 0.5
        self.filter0.data = .1 / np.sqrt(self.in_channels * 9) * W1_i * 0.5
        
    def ISTA_Block(self, input, k, d, filters, b, bn, c,dcyc,unfolding):
        features = []
        features.append(input)
        
        for i in range(d):
            f1 = F.conv1d(features[-1], filters[i], stride=1, padding=(i % dcyc) + 1,dilation=1)
            f2 = torch.cat((features[-1], f1), dim=1)
            del f1
            f3 = c[i] * f2 + b[i]
            del f2
            features.append(F.relu(bn[i](f3)))
            del f3

        # backward
        for loop in range(unfolding):
            for i in range(d - 1):
                f1 = F.conv_transpose1d(features[-1 - i][:, -k:, :], filters[-1 - i], stride=1,padding=((-1 - i + d) % dcyc) + 1,dilation=1)
                features[-2 - i] = f1 + features[-1 - i][:, 0:-k, :]
            # forward
            for i in range(d):
                f1 = F.conv_transpose1d(features[i + 1][:, -k:, :], filters[i], stride=1,padding=(i % dcyc) + 1, dilation=1)
                f2 = features[i + 1][:, 0:-k, :] + f1
                del f1
                f3 = F.conv1d(f2, filters[i], stride=1, padding=(i % dcyc) + 1,dilation= 1)
                f4 = torch.cat((f2, f3), dim=1)  
                del f2,f3
                f5 = F.conv1d(features[i], filters[i], stride=1, padding=(i % dcyc) + 1,dilation= 1)
                f6 = torch.cat((features[i], f5), dim=1)  
                f7 = features[i + 1] - c[i] * (f4 - f6) + b[i]
                del f4,f6
                features[i + 1] = F.relu(bn[i](f7))

        return features[-1]

    def forward(self, x):
        x = F.conv1d(x,self.filter0,stride=1,padding=850)+self.b0
        x = F.max_pool1d(x,kernel_size=16)
        x = self.bn0(x)
        
        x = self.ISTA_Block(x,32,3,self.filters1,self.b1,self.bn1,self.c1,1,3)
        x = self.ese(x)
        x = x.view(x.shape[0],x.shape[1]*x.shape[2])
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        
        outplot = x
        output = F.log_softmax(x, dim=1)
        
        return output,outplot