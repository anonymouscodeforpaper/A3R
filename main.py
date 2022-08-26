#!/usr/bin/env python
# coding: utf-8

# In[2]:

import json
from train import train
import argparse
from data_loader import read,read_attributes


# In[4]:






parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default='music', help='selection of dataset')
parser.add_argument('--n_epochs', type=int, default=100, help='the number of epochs')
parser.add_argument('--dim', type=int, default=8, help='dimension of user and entity embeddings')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--l2_weight', type=float, default=1e-6, help='weight of l2 regularization')
parser.add_argument('--lr', type=float, default=1e-2, help='learning rate')
parser.add_argument('--rate', type=float, default=0.0, help='dropout rate')
parser.add_argument('--verbos', type=bool, default=True, help='Whether to show the results of each epoch')

args = parser.parse_args()



train(args)


# In[ ]:




