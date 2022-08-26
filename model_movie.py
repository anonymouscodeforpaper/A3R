#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tqdm import tqdm
from torch.nn import LeakyReLU
leaky = LeakyReLU(0.2)
from torch import nn
import torch
import torch.nn.functional as F

# In[3]:


def calculate_score(x, users, aspects):
    niubi = []
    # users: 128 * 8
    first = x.index[0]
    val_base = x[first]
    actor_base = torch.LongTensor(val_base)
    actors_base = aspects(actor_base)  # 3 * 8
    pre_rating = torch.mm(actors_base, users[first].unsqueeze(1))
    niubi.append(pre_rating)
    pre_rating = pre_rating / pre_rating.shape[0]
    pre_rating = pre_rating.sum(0)
    for i in x.index[1:]:
        val = x[i]
        actor = torch.LongTensor(val)
        actors = aspects(actor)
        pre_ra = torch.mm(actors, users[i].unsqueeze(1))
        niubi.append(pre_ra)
        actors_f = pre_ra / pre_ra.shape[0]
        actors_f = actors_f.sum(0)
        pre_rating = torch.cat((pre_rating, actors_f))
    return pre_rating, niubi  ### pre_rating is the rating after averaging on the type; niu is the rating before averaging on the type


class aspect_augumentation_movie(nn.Module):
    def __init__(self, n_users, n_entity, n_rk, n_factors):
        super(aspect_augumentation_movie, self).__init__()
        
        
        '''
        n_users denotes the numeber of users
        n_entity denotes the number of attributes
        n_rk denotes the number of attribute type
        n_factors denotes the embedding size
        '''
        self.n_users = n_users
        self.n_entity = n_entity
        self.n_rk = n_rk
        self.n_factors = n_factors
        self.user_factors = torch.nn.Embedding(n_users, n_factors)
        self.entity_factors = torch.nn.Embedding(n_entity, n_factors)
        self.relation_k = torch.nn.Embedding(n_factors, n_rk)

    def forward(self, user_id, actors_id, directors_id, genres_id,rate):
        '''
        user_factors: n_users * 64
        entity_factors: n_entity * 64
        relation_k: 64 * 3
        artists_id is the index of authors
        categories_id is the the index of genres
        '''

        users = self.user_factors(user_id)  # 128 * 8 We get the representation of users
        aspects = self.entity_factors  # n_entity * 8 We get the representation of aspects
        
        
        users = F.dropout(users, p=rate, training=self.training) ## Here we apply dropout 
        
        '''
        Compute the importance of each aspects
        '''
        scores = torch.matmul(users, F.dropout(
            self.relation_k.weight, p=rate, training=self.training))  # 128 * 3 Here, we compute the score of each attribute type to users, Equation 1 in the paper
        scores = leaky(scores)
        m = torch.nn.Softmax(dim=1)  # 128 * 3 
        scores = m(scores)  # 128 * 3 We normalize the score to get the importance of each attribute type to users, Equation 2 in the paper


        '''
        Compute the importance of each aspects
        '''
        scores_actors=scores[:, 0] # 128,
        scores_directors=scores[:, 1] # 128,
        scores_genres=scores[:, 2] # 128,

        '''
        Compute the contribution of each aspects
        '''
        contribute_actors,niubi_act=calculate_score(
            actors_id, users,aspects)
        contribute_directors, niubi_dir=calculate_score(
            directors_id, users,aspects)
        contribute_genres, niubi_gen=calculate_score(
            genres_id, users, aspects)

        '''
        Compute the final predictions
        '''
        importance_sum=scores_actors + scores_directors + scores_genres
        prediction_sum=contribute_actors * scores_actors + contribute_directors * scores_directors + contribute_genres * scores_genres
        prediction=prediction_sum / importance_sum
        cnm = [niubi_act,niubi_dir,niubi_gen]

        return prediction, scores, contribute_actors, contribute_directors, contribute_genres,cnm
    
        '''
        prediction is the the prediction of ratings; 
        scores represent the importance of each attribute type;
        contribute_actors denote the contribution of actors;
        contribute_directors denote the contribution of directors;
        contribute_genres denote the contribution of genres;
        cnm is score before averaging
        ''' 


# In[ ]:




