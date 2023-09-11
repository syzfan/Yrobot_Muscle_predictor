# -*- coding: utf-8 -*
import torch
from matplotlib import animation
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
from scipy.interpolate import interpolate
from torch import nn,optim
from torch.autograd import Variable
from  torch.nn import init
import numpy as np
import random
import seaborn as sns
import os
from itertools import cycle
from interpolate import *
import gc

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
device = torch.device('cuda')
torch.backends.cudnn.enabled = False

#####################################################model#####################################################
class lstmModel(nn.Module):
    def __init__(self,in_dim,hidden_dim,out_dim,layer_num):
        super().__init__()
        self.linear1 = torch.nn.Linear(in_dim, hidden_dim)
        self.lstmLayer=nn.LSTM(hidden_dim,hidden_dim,layer_num,batch_first=True,dropout=0.2)
        self.sigmoid=nn.Sigmoid()
        self.fcLayer=nn.Linear(hidden_dim,out_dim)
        self.weightInit=(np.sqrt(1.0/hidden_dim))

    def forward(self, x):
        x1=self.linear1(x)
        out,_=self.lstmLayer(x1)
        s,b,h=out.size()
        out=self.fcLayer(out)
        return out

    def weightInit(self, gain=1):
        for name, param in self.named_parameters():
            if 'lstmLayer.weight' in name:
                init.orthogonal(param, gain)

lstm1 = lstmModel(36, 256, 101, 3).to(device)
criterion=nn.CrossEntropyLoss().to(device)
optimizer = optim.Adam(lstm1.parameters(), lr = 0.0001)
Data1_train = load_pickle_data("Train0609")

if __name__ == '__main__':
        vali_loss = []
        Print_Vali_Constant=25
        for i in range(200):
            for iteration, data in enumerate(Data1_train):
                #   for iteration, data in enumerate(zip((Data1_train), (Data2_train) )):

                inputs = data[0:512, 0:50, 0:36].to(torch.float32).to(device)
                outputs_real = ((data[0:512, 0:50, 37:38])).to(torch.float32).to(device)
                outputs_real = outputs_real.reshape(512, 50).to(torch.long)
                outputs_predict = lstm1(inputs)
                outputs_predict = outputs_predict.permute(0, 2, 1)
                loss = criterion(outputs_predict, outputs_real)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # print training information
                print_loss = loss.item()
                # torch.cuda.empty_cache()
                gc.collect()
                vali_loss.append(print_loss)
                print('Epoch num:{},Origin num:{},Iteration[{}], Loss: {:.5f}'.format(i + 1, "0", iteration * (i + 1),
                                                                                      print_loss))

        lstm1 = lstm1.eval()
        torch.save(lstm1, '20230609_treadmill_muscle2')
        save_to_pickle(vali_loss, "vali_loss_muscle2_treadmill_0609")
        gc.collect()






