import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle

sample_data= pd.read_csv('prepared_data.csv')
sample_duplicates= pd.read_csv('prepared_duplicates.csv')
sample_limit= len(sample_data)
# Taking the first sample_limit number of sample questions
sample_data= sample_data[:sample_limit]

# loading the pre-trained use_model
path= 'https://tfhub.dev/google/universal-sentence-encoder/4'
use_model= hub.load(path)

# importing the use_embedding
with open('use_embedding.data','rb') as filehandle:
    use_embedding= pickle.load(filehandle)

def cosine(u, v):
    ''' this function returns the cosine similarity b/w two vectors'''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# finding the similarity score list for each query and saving all of them together- the output is (440* 18k) 2D numpy array
use_simscores=[]    # list of lists which stores the similarity score list of each query
test_questions= list(sample_duplicates['title'])
for i in range(len(test_questions)):
    query= test_questions[i]
    query_vec = use_model([query])[0]
    simscore=[] # list of similarity scores with all the questions for this current query.
    for s in use_embedding:
        sim= cosine(query_vec, s)
        simscore.append(sim)
    # now we have the similarity scores with all the questions for this current query.
    use_simscores.append(simscore)
use_simscores= np.array(use_simscores)

# saving the use_simscores
with open('use_simscores.data', 'wb') as filehandle:
    pickle.dump(use_simscores, filehandle)