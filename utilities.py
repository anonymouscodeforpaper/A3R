#!/usr/bin/env python
# coding: utf-8

# In[ ]:
from sklearn.preprocessing import LabelEncoder
import torch
loss_func = torch.nn.MSELoss()
from sklearn.metrics import f1_score, roc_auc_score,accuracy_score
from concurrent.futures import ProcessPoolExecutor
THREADS = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"




def data_split(df):  # This function randomly splits the whole dataset into training set and test set with the ration train:test = 8:2
    trainset = df.sample(frac=0.9, random_state=0, axis=0)
    trainset.index = range(len(trainset))
    testset = df[~df.index.isin(trainset.index)]
    testset.index = range(len(testset))
    return trainset, testset

def RMSE(data, model,rate,name):  ### this function returns RMSE, MAE, precicion, recall, f1 score, accuracy, and the rating before averaging
    if name == 'book':
        users_index = data.iloc[:, 0].values
        users = torch.LongTensor(users_index).to(DEVICE)
        actors_id = data.iloc[:, 2]
        directors_id = data.iloc[:, 3]
        rating = torch.FloatTensor(
        data.iloc[:, 4].values).to(DEVICE)
        prediction,scores,contribute_actors,contribute_directors,cnm = model(users,actors_id, directors_id,rate)
        rmse = loss_func(prediction, rating)
        mae = torch.nn.L1Loss()(prediction, rating)
        return rmse ** 0.5,mae,cnm
    if name == 'music':
        users_index = data.iloc[:, 0].values
        users = torch.LongTensor(users_index).to(DEVICE)
        actors_id = data.iloc[:, 2]
        directors_id = data.iloc[:, 3]
        rating = torch.FloatTensor(
        data.iloc[:, 4].values).to(DEVICE)
        prediction,scores,contribute_actors,contribute_directors,cnm = model(users,actors_id, directors_id,rate)
        rmse = loss_func(prediction, rating)
        mae = torch.nn.L1Loss()(prediction, rating)
        return rmse ** 0.5,mae,cnm
        
    else:
        users_index = data.iloc[:, 0].values
        users = torch.LongTensor(users_index).to(DEVICE)
        actors_id = data.iloc[:, 2]
        directors_id = data.iloc[:, 3]
        genres_id = data.iloc[:, 4]
        rating = torch.FloatTensor(data.iloc[:, 5].values).to(DEVICE)
        prediction,scores,contribute_actors,contribute_directors,contribute_genres,cnm = model(users,actors_id, directors_id, genres_id)
        rmse = loss_func(prediction, rating)
        mae = torch.nn.L1Loss()(prediction, rating)
        return rmse ** 0.5,mae


# In[3]:


def arg_accuracy_int(ratings, predictions): ###  This is the implementation in reference 13, it compute the accuracy of prediction
    ratings = ratings.cpu().detach().numpy()
    predictions = predictions.cpu().detach().numpy()
    total_nr = len(ratings)
    total_pred = 0
    for i in range(total_nr):
        (true_rating, pred_rating) = ratings[i], predictions[i]
        if round(pred_rating) >= int(true_rating)-1 and round(pred_rating) <= int(true_rating)+1:
            total_pred += 1

    return float(total_pred)/total_nr


def round_of_rating(number):
    return round(number * 2) / 2



def map_aspect_values_to_movies(x): ## This is the implementation in reference 15, it maps the values of attributes to each movies, books, music
    (film, meta), aspect = x
    aspects = dict()
    if aspect == "director" and type(meta[aspect]) is str:
        aspects[meta[aspect]] = 1
    else:
        for g in meta[aspect]:
            aspects[g] = 1
    return film, meta, aspects


def dict_movie_aspect(paper_films, aspect): ##This is the implementation in reference 14, it returns a dictionary of each attribute type for books, movies, music
    paper_films_aspect_prepended = map(
        lambda e: (e, aspect), list(paper_films.items()))
    aspect_dict = dict()
    with ProcessPoolExecutor(max_workers=THREADS) as executor:
        results = executor.map(map_aspect_values_to_movies,
                               paper_films_aspect_prepended)
    for film, meta, aspects in results:
        aspect_dict[film] = aspects

    return aspect_dict




def transform_index(df,col):
    le = LabelEncoder()
    y = le.fit_transform(df[col])
    df[col] = y
    return df
    