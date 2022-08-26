#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import torch
from torch import nn
import time
from model_book import aspect_augumentation_book
from model_movie import aspect_augumentation_movie
from model_music import aspect_augumentation_music
from utilities import RMSE
from data_loader import read
from tqdm import tqdm
import torch.nn.functional as F
loss_func = torch.nn.MSELoss()

# In[4]:


def train(args):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    trainset, testset, n_user, n_attributes, n_attribute_types = read(args)
    rate = args.rate
    if args.name == 'book':
        model = aspect_augumentation_book(
            n_user, n_attributes, n_attribute_types, args.dim).to(DEVICE)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, threshold=0.005, threshold_mode='abs')
        for epoch in range(args.n_epochs):
            model.train()
            t1 = time.time()
            num_example = len(trainset)
            indices = list(range(num_example))
            for i in tqdm(range(0, num_example, args.batch_size)):
                optimizer.zero_grad()
                indexs = indices[i:min(i+args.batch_size, num_example)]
                
                
                users_index = trainset.iloc[:, 0].loc[indexs].values
                users = torch.LongTensor(users_index).to(DEVICE)
           
        
                actors_id = trainset.iloc[:, 2].loc[indexs]
                actors_id.index = range(len(actors_id))
                
                
                directors_id = trainset.iloc[:, 3].loc[indexs]
                directors_id.index = range(len(directors_id))
                
                
                rating = torch.FloatTensor(
                    trainset.iloc[:, 4].loc[indexs].values).to(DEVICE)
                prediction, scores, contribute_actors, contribute_directors, cnm = model(
                    users, actors_id, directors_id,rate)

                err = loss_func(prediction, rating) + args.l2_weight * ((model.user_factors.weight*model.user_factors.weight).sum() + (model.entity_factors.weight*model.entity_factors.weight).sum() +(model.relation_k.weight*model.relation_k.weight).sum())

                err.backward()
                optimizer.step()
            t2 = time.time()
            rmse, mae,cnm = RMSE(
                data = testset, model = model,rate = 0.0, name = 'book')
            scheduler.step(rmse)
            if args.verbos == True:
                print("Epoch: ", epoch, " Loss: ", err,
                      " RMSE in test set:", rmse, "MAE in test set: ", mae)
                
                
                
                
    if args.name == 'music':
        model = aspect_augumentation_music(
            n_user, n_attributes, n_attribute_types, args.dim).to(DEVICE)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, threshold=0.005, threshold_mode='abs')
        for epoch in range(args.n_epochs):
            model.train()
            t1 = time.time()
            num_example = len(trainset)
            indices = list(range(num_example))
            for i in tqdm(range(0, num_example, args.batch_size)):
                optimizer.zero_grad()
                indexs = indices[i:min(i+args.batch_size, num_example)]
                
                users_index = trainset.iloc[:, 0].loc[indexs].values
                users = torch.LongTensor(users_index).to(DEVICE)
                
                
                actors_id = trainset.iloc[:, 2].loc[indexs]
                actors_id.index = range(len(actors_id))
                
                
                directors_id = trainset.iloc[:, 3].loc[indexs]
                directors_id.index = range(len(directors_id))
                
                
                rating = torch.FloatTensor(
                    trainset.iloc[:, 4].loc[indexs].values).to(DEVICE)
                prediction, scores, contribute_actors, contribute_directors, cnm = model(
                    users, actors_id, directors_id,rate)

                err = loss_func(prediction, rating) + args.l2_weight * ((model.user_factors.weight*model.user_factors.weight).sum() + (model.entity_factors.weight*model.entity_factors.weight).sum() +(model.relation_k.weight*model.relation_k.weight).sum())

                err.backward()
                optimizer.step()
            t2 = time.time()
            rmse, mae,cnm = RMSE(
                data = testset, model = model,rate = 0.0, name = 'music')
            scheduler.step(rmse)
            if args.verbos == True:
                print("Epoch: ", epoch, " Loss: ", err,
                      " RMSE in test set:", rmse, "MAE in test set: ", mae)
        
    
    
    
    
    else:
        model = aspect_augumentation_movie(
            n_user, n_attributes, n_attribute_types, args.dim).to(DEVICE)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5, threshold=0.005, threshold_mode='abs')
        for epoch in range(args.n_epochs):
            model.train()
            t1 = time.time()
            num_example = len(trainset)
            indices = list(range(num_example))
            for i in tqdm(range(0, num_example, args.batch_size)):
                optimizer.zero_grad()
                indexs = indices[i:min(i+args.batch_size, num_example)]
                
                users_index = trainset.iloc[:, 0].loc[indexs].values
                users = torch.LongTensor(users_index).to(DEVICE)
                
                actors_id = trainset.iloc[:, 2].loc[indexs]
                actors_id.index = range(len(actors_id))
                
                directors_id = trainset.iloc[:, 3].loc[indexs]
                directors_id.index = range(len(directors_id))
                
                genres_id = trainset.iloc[:, 4].loc[indexs]
                genres_id.index = range(len(genres_id))
                
                rating = torch.FloatTensor(
                trainset.iloc[:, 5].loc[indexs].values).to(DEVICE)
                
                prediction, scores, contribute_actors, contribute_directors, contribute_genres,cnm = model(
                users, actors_id, directors_id, genres_id,rate)
                
                
                err = loss_func(prediction, rating) + args.l2_weight * ((model.user_factors.weight*model.user_factors.weight).sum() + (model.entity_factors.weight*model.entity_factors.weight).sum() +(model.relation_k.weight*model.relation_k.weight).sum())

                err.backward()
                optimizer.step()
            t2 = time.time()
            rmse, mae,cnm = RMSE(
                data = testset, model = model,rate = 0.0, name = 'movie')
            scheduler.step(rmse)
            if args.verbos == True:
                print("Epoch: ", epoch, " Loss: ", err,
                      " RMSE in test set:", rmse, "MAE in test set: ", mae)
                
        
    
#                 print("Accuracy in test set is: ", accuracy, "Precision in test set:",
#                       p, "Recall in test set: ", r, "F1 scores in test set is:", f)
#                 print("Time consumed is:", t2-t1)
#         return rmse, mae, p, r, f, accuracy, model, cnm
    
    
        


# In[ ]:




