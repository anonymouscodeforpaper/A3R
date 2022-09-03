#!/usr/bin/env python
# coding: utf-8

# In[1]:


from utilities import map_aspect_values_to_movies,dict_movie_aspect,data_split,transform_index
import json 
import numpy as np
import pandas as pd
from operator import itemgetter


# In[2]:


def read(args):
    n_attributes  = read_attributes(args.name)
    trainset, testset,n_users,n_attribute_types = read_ratings(args.name)
    return trainset, testset, n_users, n_attributes,n_attribute_types
       


# In[3]:




def read_attributes(name):
    
    
    if name == 'book':
        
        with open("data/book/attribute.json", 'r', encoding='UTF-8') as f:
            info = json.load(f)
        info = {int(k): v for k, v in info.items()}

        # We get a dictionary that contains the genres of each book
        book_genres = dict_movie_aspect(info, "genre")
        # We get a dictionary that contains the authors of each book
        book_authors = dict_movie_aspect(info, 'author')

        authors = pd.DataFrame.from_dict(
            book_authors, dtype='int64', orient='index').T.replace(np.nan, 0).index
        genres = pd.DataFrame.from_dict(
            book_genres, dtype='int64', orient='index').T.replace(np.nan, 0).index

        authors_dict = dict(zip(authors, range(len(authors))))
        genres_dict = dict(zip(genres, range(
            len(authors), len(authors) + len(genres))))
        
        n_attributes = max(genres_dict.values()) + 1 ## This is the number of attributes of all items
        
    
        
        jsonstr_author = json.dumps(authors_dict) ## We save the author information into a json file
        filename_author = open('data/book/authors.json','w')
        filename_author.write(jsonstr_author)
        
        
        jsonstr_genre = json.dumps(genres_dict)  ## We save the genre information into a json file
        filename_genre = open('data/book/genre.json','w')
        filename_genre.write(jsonstr_genre)
        
        
        jsonstr_genre = json.dumps(genres_dict)  ## ## We save the genre information into a json file
        filename_genre = open('data/book/genre.json','w')
        filename_genre.write(jsonstr_genre)
        return n_attributes
     
        
        
        
    if name == 'music':
        music_df = pd.read_excel('data/music/Data_InCarMusic.xlsx', usecols=[0,2,7], sheet_name='Music Track') 
        genre_map = {1:'Blues music',2:'Classical music',3:'Country music',4:'Disco music',5:'Hip Hop music',6:'Jazz music',
            7:'Metal music',8:'Pop music',9:'Reggae music',10:'Rock music'}
        music_df = music_df.rename(columns = {'id':'ItemID',' category_id':'category_id',' artist':'artist'})
        music_df['category_id'] = music_df['category_id'].map(genre_map)
        
        ### Stock the attributes of music in a dictionary
        info = {}
        for ind in music_df.index:
            music = music_df['ItemID'][ind]
            info.setdefault(music,{})
            info[music]['artist'] = [music_df['artist'][ind]]
            info[music]['category'] = [music_df['category_id'][ind]]
            
        music_category = dict_movie_aspect(info, "category")
        music_artist = dict_movie_aspect(info,'artist')
        
        
        
        
        cartegory = pd.DataFrame.from_dict(music_category,dtype = 'int64',orient = 'index').T.replace(np.nan, 0).index
        artist = pd.DataFrame.from_dict(music_artist,dtype = 'int64',orient = 'index').T.replace(np.nan, 0).index
        
        
        category_dict = dict(zip(cartegory, range(len(cartegory))))
        artist_dict = dict(zip(artist, range(len(cartegory), len(cartegory) + len(artist))))
        
        
   
        
        
        jsonstr_category = json.dumps(category_dict)  ## We save the category information into a json file
        filename_category = open('data/music/category.json','w')
        filename_category.write(jsonstr_category)
        
        
        jsonstr_artist = json.dumps(artist_dict)  ## ## We save the artist information into a json file
        filename_artist = open('data/music/artist.json','w')
        filename_artist.write(jsonstr_artist)
        
        n_attributes = max(artist_dict.values()) + 1
        
        
        
        return n_attributes
        
        
        
    
    
    else:
        films = pd.read_pickle('data/' + str(name) + '/movie_metadata.pkl')
        row = []
        for mid,haha in films.items():
            haha['movie'] = mid
            row.append(haha)
        df_movie = pd.DataFrame(row)
            
            
        movie = dict()
        for i in range(df_movie.shape[0]):
            m = df_movie['movie'][i]
            movie.setdefault(m,dict())
            movie[m]['director'] = df_movie['director'][i]
            movie[m]['genre'] = df_movie['genre'][i]
            movie[m]['actors'] = df_movie['actors'][i]
            movie[m]['title'] = df_movie['title'][i]
            
            
        
        movie_genres = dict_movie_aspect(movie, "genre")
        movie_directors = dict_movie_aspect(movie,'director')
        movie_actors = dict_movie_aspect(movie,'actors')
        actors = pd.DataFrame.from_dict(movie_actors,dtype = 'int64',orient = 'index').T.replace(np.nan, 0).index
        directors = pd.DataFrame.from_dict(movie_directors,dtype = 'int64',orient = 'index').T.replace(np.nan, 0).index
        genres = pd.DataFrame.from_dict(movie_genres,dtype = 'int64',orient = 'index').T.replace(np.nan, 0).index
        
        ### This is to get a maping for each attribute
        actors_dict = dict(zip(actors, range(len(actors))))
        directors_dict = dict(zip(directors, range(len(actors), len(actors) + len(directors))))
        genres_dict = dict(zip(genres, range(len(actors) + len(directors), len(actors) + len(directors) + len(genres))))
        
        n_attributes = max(genres_dict.values()) + 1 ## The total number of of attributes 
        
        
        
        jsonstr_movie = json.dumps(movie) ## We save the item information into a json file
        filename_movie = open('data/' + str(name) + '/' + str(name) + '.json','w')
        filename_movie.write(jsonstr_movie)
        
        
        jsonstr_genre = json.dumps(genres_dict)  ## We save the genre information into a json file
        filename_genre = open('data/' + str(name) + '/genre.json','w')
        filename_genre.write(jsonstr_genre)
        
        jsonstr_genre = json.dumps(genres_dict)  ## We save the genre information into a json file
        filename_genre = open('data/' + str(name) + '/genre.json','w')
        filename_genre.write(jsonstr_genre)
        
        
        jsonstr_actor = json.dumps(actors_dict)  ## ## We save the actor information into a json file
        filename_actor = open('data/' + str(name) + '/actor.json','w')
        filename_actor.write(jsonstr_actor)
        
        jsonstr_director = json.dumps(directors_dict)  ## ## We save the director information into a json file
        filename_director = open('data/' + str(name) + '/director.json','w')
        filename_director.write(jsonstr_director)
        
        
        return n_attributes 
        
        
        
        
        
        
        









def read_ratings(name):
    if name == 'book':
        with open("data/book/attribute.json", 'r', encoding='UTF-8') as f:
            info = json.load(f)
        info = {int(k): v for k, v in info.items()}
        
        global authors_dict
        global genres_dict
        
        with open("data/book/authors.json", 'r', encoding='UTF-8') as f:
            authors_dict = json.load(f)
            
        
        with open("data/book/genre.json", 'r', encoding='UTF-8') as f:
            genres_dict = json.load(f)
            
        
        
        df = pd.read_csv('data/book/ratings.csv') ### This file contains the user-item interactions
    
        i_map = pd.read_csv('data/book/i_map.dat',sep = '\t',header = None) # This file contains the mapping of item index
    
        haha = pd.read_csv('i2kg_map.tsv',sep='\t', header=None)# This file contains the index of books, dblink of books
        haha = haha.rename(columns = {0:'item',1:'name',2:'url'})
        haha = haha.sort_values(by='item',ascending=True) 
        haha_final = haha[haha['item'].isin(i_map[1])]
    
    
        haha_final['authors'] = haha_final['item'].map(info) 
        haha_final['genres'] = haha_final['item'].map(info) 
        for i in haha_final.index:
            haha_final['authors'][i] = info[haha_final['item'][i]]['author']## We get the rela value of authors of books
            haha_final['genres'][i] = info[haha_final['item'][i]]['genre']## We get the real values of genres of books
        
        
        haha_final.index = range(len(haha_final))
        del haha_final['url']
        i_map = i_map.rename(columns = {0:'real',1:'item'})
    
    
        final = pd.merge(haha_final,i_map) ## We get the attributes of items with the index corresponding to user-item interactions
        del final['item']
        final = final.rename(columns = {"real":"item"})
    
    
        
        model_df = pd.merge(df,final)
        model_df = model_df[['user', 'item', 'authors', 'genres','rating']]
        
        
        
        model_df['authors'] = model_df['authors'].map(change_author_book)
        model_df['genres'] = model_df['genres'].map(change_genre_book)

        
        
        
     
        model_df['rating'] = model_df['rating'].astype(float)
        model_df['rating'] = (model_df['rating'] - 1) / 2
        trainset, testset = data_split(model_df)
        n_users = max(model_df['user'])+1 ## This is the number of users
        n_attribute_types = 2
    
        return trainset, testset, n_users, n_attribute_types
    
    if name == 'music':
        ratings = pd.read_excel('data/music/Data_InCarMusic.xlsx', usecols=[0,1,2], sheet_name='ContextualRating')
        music_df = pd.read_excel('data/music/Data_InCarMusic.xlsx', usecols=[0,2,7], sheet_name='Music Track')
        genre_map = {1:'Blues music',2:'Classical music',3:'Country music',4:'Disco music',5:'Hip Hop music',6:'Jazz music',
            7:'Metal music',8:'Pop music',9:'Reggae music',10:'Rock music'}
        music_df = music_df.rename(columns = {'id':'ItemID',' category_id':'category_id',' artist':'artist'})
        ratings = ratings.rename(columns = {' Rating':'Rating'})
        music_df['category_id'] = music_df['category_id'].map(genre_map)
        df = pd.merge(ratings,music_df)
        
        with open('data/music/category.json', 'r', encoding='UTF-8') as f:
            category_map = json.load(f)
        
        
        with open('data/music/artist.json', 'r', encoding='UTF-8') as f:
            artist_map = json.load(f)
            
        df['artist'] = df['artist'].map(artist_map)
        df['category_id'] = df['category_id'].map(category_map)
        
        df = transform_index(df,'UserID')
        df = transform_index(df,'ItemID')
        df = df[['UserID', 'ItemID', 'artist', 'category_id', 'Rating']]
        df['Rating'] = (df['Rating'] - 1) / 2
        trainset, testset = data_split(df)
        n_users = len(df['UserID'].value_counts())
        n_attribute_types = 2
        
        return trainset, testset, n_users, n_attribute_types
        
        
        
        
    
    else:
        ratings = pd.read_pickle('data/' + str(name) + '/movie_ratings_500_id.pkl')
        films = pd.read_pickle('data/' + str(name) + '/movie_metadata.pkl')
        
        global actors_movie_dict
        global directors_movie_dict
        global genres_movie_dict
        
        
            
        
        with open('data/' + str(name) + '/actor.json', 'r', encoding='UTF-8') as f:
            actors_movie_dict = json.load(f)
        
        
        with open('data/' + str(name) + '/director.json', 'r', encoding='UTF-8') as f:
            directors_movie_dict = json.load(f)
            
        with open('data/' + str(name) + '/genre.json', 'r', encoding='UTF-8') as f:
            genres_movie_dict = json.load(f)
    

    
    
        row = []
        for mid,haha in ratings.items(): 
            for info in haha:
                info['movie'] = mid
                row.append(info)      
        df_ratings = pd.DataFrame(row) ## We get a dataframe that contains user-item interactions


        row = []
        for mid,haha in films.items():
            haha['movie'] = mid
            row.append(haha)
        df_movie = pd.DataFrame(row)## We get a dataframe that contains attributes of items
        
        
        
        final = pd.merge(df_movie,df_ratings,on='movie') ### We merge df_movie and df_ratings
        a = final['user_id'].value_counts()[final['user_id'].value_counts() >= 10]
        final = final[final['user_id'].isin(list(a.index))]
        final.index = range(len(final))
        hehe = final.copy()
        del hehe['title']
        del hehe['user_rating_date']
        
        
        hehe['director'] = hehe['director'].map(change_director_movie)
        hehe['actors'] = hehe['actors'].map(change_actor_movie)
        hehe['genre'] = hehe['genre'].map(change_genre_movie)
        hehe = hehe[['user_id','movie','actors','director','genre','user_rating']]
        hehe['user_rating'] = hehe['user_rating'].astype('float')
        hehe = transform_index(hehe,'user_id')
        hehe = transform_index(hehe,'movie')
        hehe['user_rating'] = (hehe['user_rating'] - 1) / 2
        
        n_users = max(hehe['user_id']) + 1
        trainset, testset = data_split(hehe)
        n_attribute_types = 3
    
        return trainset, testset, n_users, n_attribute_types

        

        

        
        
        
        
#### Map the keys to values for books
def change_author_book(x):
    qunima = itemgetter(*x)(authors_dict)
    if type(qunima) == int:
        return [qunima]
    else:
        return change_list(qunima)
    
def change_genre_book(x):
    qunima = itemgetter(*x)(genres_dict)
    if type(qunima) == int:
        return [qunima]
    else:
        return change_list(qunima)   
    
    
    
### Map the keys to values for movies
    
def change_director_movie(x):
    return [itemgetter(x)(directors_movie_dict)]


def change_actor_movie(x):
    return change_list((itemgetter(*x)(actors_movie_dict)))


def change_genre_movie(x):
    qunima = itemgetter(*x)(genres_movie_dict)
    if type(qunima) == int:
        return [qunima]
    else:
        return change_list(qunima)
    

    
    


def change_list(x):
    return list(x)








# In[ ]:




