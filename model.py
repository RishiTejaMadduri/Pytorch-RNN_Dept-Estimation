#!/usr/bin/env python
# coding: utf-8

# In[21]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import BasicConvLSTMCell

import torchvision
DISP_SCALING_RESNET50 = 10.0
MIN_DISP = 0.01


# In[22]:


def resize_like(inputs,ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    
    return input[:, :, :ref.size(2), :ref.size(3)]
     


# In[23]:


def convLSTM(input, hidden, filters, kernel, scope):
    cell = BasicConvLSTMCell.BasicConvLSTMCell([input.shape[1], input.shape[2]], kernel, filters)
    
    if hidden is None:
        hidden=cell.zero_state(input.shape[0]).float()
    
    y_, hideen=cell(input, hidden)
    
    return y_, hidden


# In[24]:


def pred_depth(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 1, kernel_size = 1, padding = 1),
        nn.Sigmoid()
    )


# In[25]:


def conv(in_planes, out_planes, kernel_size = 3, stride = 2):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, padding=(kernel_size-1)//2, stride = stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace = True)
    )


# In[26]:


def Iconv(in_planes, out_planes):
    return nn.Sequential(
        nn.Conv2d(in_place, out_planes, kernel_size = 3, padding = 1),
        nn.ReLU(inplace=True)
    )


# In[27]:


def downsample_conv(in_planes, out_planes, kernel_size = 3):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size = kernel_size, stride = 2, padding = (kernel_size - 1)//2),
        nn.Relu(inplace = True),
        nn.BatchNorm2d(out_planes),
        nn.Conv2d(out_planes, out_planes, kernel_size = kernel_size, padding = (kernel_size - 1)//2),
        nn.Relu(inplace = True)
    )


# In[28]:


def upconv(in_planes, out_planes):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size = 3, stride = 2, padding = 1, output_padding = 1),
        nn.ReLU(inplace =  True)
    )


# In[29]:


class rnn_depth_net_encoderlstm():
    
    def __init__(self, current_input, hidden_state):
        super(rnn_depth_net_encoderlstm, self).__init__()
        
        H = current_input.shape[1]
        W = current_input.shape[2]
        
        conv_planes = [32, 64, 128, 256, 256, 256, 512]
        
        self.conv1 = downsample_conv(3, conv_planes[0], kernel_size = 7)
        self.conv2 = downsample_conv(conv_planes[0], conv_planes[1], kernel_size = 5)
        self.conv3 = down_sample_conv(conv_planes[1], conv_planes[2])
        self.conv4 = down_sample_conv(conv_planes[2], conv_planes[3])
        self.conv5 = down_sample_conv(conv_planes[3], conv_planes[4])
        self.conv6 = down_sample_conv(conv_planes[4], conv_planes[5])
        self.conv7 = down_sample_conv(conv_planes[5], conv_planes[6])
        
        upconv_planes = [256, 128, 128, 128, 64, 32, 16]
        
        self.upconv7 = upconv(conv_planes[6], upconv_planes[0])
        self.upconv6 = upconv(conv_planes[0], upconv_planes[1])
        self.upconv5 = upconv(conv_planes[1], upconv_planes[2])
        self.upconv4 = upconv(conv_planes[2], upconv_planes[3])
        self.upconv3 = upconv(conv_planes[3], upconv_planes[4])
        self.upconv2 = upconv(conv_planes[4], upconv_planes[5])
        self.upconv1 = upconv(conv_planes[5], upconv_planes[6])
        
        self.iconv7 = Iconv(input,256)
        self.iconv6 = Iconv(input, 128)
        self.iconv5 = Iconv(input, 128)
        self.iconv4 = Iconv(input, 128)
        self.iconv3 = Iconv(input, 64)
        self.iconv2 = Iconv(input, 32)
        self.iconv1 = Iconv(input, 16)
        
        depth = pred_depth(16)
        #try putting input instead of out_conv# incase you get dimension error
        def forward(self, current_input, hidden_state):
            out_conv1 = self.conv1(current_input)
            out_conv1b, hidden1 = convLSTM(out_conv1, hidden_state[0], 32, [3,3])
            out_conv2 = self.conv2(out_conv1b)
            out_conv2b, hidden2 = convLSTM(out_conv2, hidden_state[1], 64, [3,3])
            out_conv3 = self.conv3(out_conv2b)
            out_conv3b, hidden3 = convLSTM(out_conv3, hidden_state[2], 128, [3,3])
            out_conv4 = self.conv4(out_conv3b)
            out_conv4b, hidden4 = convLSTM(out_conv4, hidden_state[3], 256, [3,3])
            out_conv5 = self.conv5(out_conv4b)
            out_conv5b, hidden5 = convLSTM(out_conv5, hidden_state[4], 256, [3,3])
            out_conv6 = self.conv6(out_conv5b)
            out_conv6b, hidden6 = convLSTM(out_conv6, hidden_state[5], 256, [3,3])
            out_conv7 = self.conv7(out_conv6b)
            out_conv7b, hidden7 = convLSTM(out_conv7, hidden_state[6], 512, [3,3])
            
            out_upconv7 = self.upconv7(out_conv7b)
            out_upconv7 = resize_like(out_upconv7, out_conv6b)
            i7_in = torch.cat((out_upconv7, conv6b), dim=3)
            out_iconv7 = self.iconv7(i7_in)
            
            out_upconv6 = self.upconv6(out_iconv7)
            out_upconv6 = resize_like(out_upconv6, out_conv5b)
            i6_in = torch.cat((out_upconv6, conv5b), dim=3)
            out_iconv6 = self.iconv6(i6_in)
            
            out_upconv5 = self.upconv5(out_iconv6)
            out_upconv5 = resize_like(out_upconv5, out_conv4b)
            i5_in = torch.cat((out_upconv5, conv4b), dim=3)
            out_iconv5 = self.iconv5(i5_in)
            
            out_upconv4 = self.upconv6(out_iconv5)
            out_upconv4 = resize_like(out_upconv4, out_conv3b)
            i4_in = torch.cat((out_upconv4, conv3b), dim=3)
            out_iconv4 = self.iconv4(i4_in)
            
            out_upconv3 = self.upconv3(out_iconv4)
            out_upconv3 = resize_like(out_upconv3, out_conv2b)
            i3_in = torch.cat((out_upconv3, conv2b), dim=3)
            out_iconv3 = self.iconv3(i3_in)
            
            out_upconv2 = self.upconv2(out_iconv3)
            out_upconv2 = resize_like(out_upconv2, out_conv1b)
            i2_in = torch.cat((out_upconv2, conv1b), dim=3)
            out_iconv2 = self.iconv2(i2_in)
            
            out_upconv1 = self.upconv1(out_iconv2)
            out_iconv1 = self.iconv1(out_upconv1)
            
            out_depth = self.depth(out_iconv1)*DISP_SCALING_RESNET50+MIN_DISP
            
            return depth, [hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]
            
            
            
            
        
        


# In[31]:


class pose_net(nn.Module):
    
    def __init__(self, posenet_inputs, hidden_state):
        super(pose_net, self).__init__()
        
        conv_planes = [16, 16, 64, 128, 256, 256, 256, 512]
        self.conv1 = conv(3, conv_planes[0], kernel_size = 7)
        self.conv2 = conv(conv_planes[0], conv_planes[1])
        self.conv3 = conv(conv_planes[1], conv_planes[2])
        self.conv4 = conv(conv_planes[2], conv_planes[3])
        self.conv5 = conv(conv_planes[3], conv_planes[4])
        self.conv6 = conv(conv_planes[4], conv_planes[5])
        self.conv7 = conv(conv_planes[6], conv_planes[7])
        self.pose_pred = nn.Conv2d(conv_planes[6], 6, kernel_size = 1)
        
    def forward(self, posnet_inputs, hidden_states):
        out_conv1 = self.conv1(posnet_inputs)
        out_conv1b, hidden1 = convLSTM(out_conv1, hidden_state[0], 16, 3)
        out_conv2 = self.conv2(out_conv1b)
        out_conv2b, hidden2 = convLSTM(out_conv2, hidden_state[1], 64, 3)
        out_conv3 = self.conv3(out_conv2b)
        out_conv3b, hidden3 = convLSTM(out_conv3, hidden_state[2], 128, 3)
        out_conv4 = self.conv4(out_conv3b)
        out_conv4b, hidden4 = convLSTM(out_conv4, hidden_state[3], 256, 3)
        out_conv5 = self.conv5(out_conv4b)
        out_conv5b, hidden5 = convLSTM(out_conv5, hidden_state[4], 256, 3)
        out_conv6 = self.conv6(out_conv5b)
        out_conv6b, hidden6 = convLSTM(out_conv6, hidden_state[5], 256, 3)
        out_conv7 = self.conv7(out_conv6b)
        out_conv7b, hidden7 = convLSTM(out_conv7, hidden_state[6], 512, 3)
        pose = self.pose_pred(out_conv7b)
        pose_avg = pose.mean(3).mean(2)
        pose_final = torch.reshape(pose_avg, [-1, 6])*0.01
        
        return pose_final,[hidden1, hidden2, hidden3, hidden4, hidden5, hidden6, hidden7]  
    
        
        
        


# In[ ]:




