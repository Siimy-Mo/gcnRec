import datetime
import math
import numpy as np
import torch
from torch import nn
from torch.nn import Module, Parameter
import torch.nn.functional as F


class GNN(Module):
  def __init__(self, hidden_size, step=1):
    self.step = step
    self.hidden_size = hidden_size
    # 设置一些要用的参数，wb

  def GNNCell(self, A, hidden):
    return 'hy'



  def forward(self, A, hidden):
    for i in range(self.step):
        hidden = self.GNNCell(A, hidden)
    return hidden

class SessionGraph(Module):
  def __init__(self, opt, n_node):
    super(SessionGraph, self).__init__()
    self.optimizer = torch.optim.Adam(self.parameters(), lr=opt.lr, weight_decay=opt.l2)
    self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=opt.lr_dc_step, gamma=opt.lr_dc)
    self.reset_parameters()

  def reset_parameters(self):
    stdv = 1.0 / math.sqrt(self.hidden_size)
    for weight in self.parameters():
        weight.data.uniform_(-stdv, stdv)

  def compute_scores(self, hidden, mask):
    return hidden

  def forward(self, inputs, A):
    return inputs


# 一些其他的def