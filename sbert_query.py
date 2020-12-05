import numpy as np
import pandas as pd
import math
import io
import nltk
from nltk.tokenize import word_tokenize
import pickle
from sentence_transformers import SentenceTransformer

sample_data= pd.read_csv('prepared_data.csv')
sample_duplicates= pd.read_csv('prepared_duplicates.csv')
sample_limit= len(sample_data)
# Taking the first sample_limit number of sample questions
sample_data= sample_data[:sample_limit]

# importing the saved sbert_model
filename= 'sbert_model.sav'
sbert_model= pickle.load(open(filename, 'rb'))

# importing the sbert_embedding
with open('sbert_embedding.data','rb') as filehandle:
    sbert_embedding= pickle.load(filehandle)

def cosine(u, v):
    ''' this function returns the cosine similarity b/w two vectors'''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))

# finding the similarity score list for each query and saving all of them together- the output is (440* 18k) 2D numpy array
sbert_simscores=[]    # list of lists which stores the similarity score list of each query
test_questions= list(sample_duplicates['title'])
for i in range(len(test_questions)):
    query= test_questions[i]
    query_vec = sbert_model.encode([query])[0]
    simscore=[] # list of similarity scores with all the questions for this current query.
    for s in sbert_embedding:
        sim= cosine(query_vec, s)
        simscore.append(sim)
    # now we have the similarity scores with all the questions for this current query.
    sbert_simscores.append(simscore)
sbert_simscores= np.array(sbert_simscores)

# saving the use_simscores
with open('sbert_simscores.data', 'wb') as filehandle:
    pickle.dump(sbert_simscores, filehandle)
