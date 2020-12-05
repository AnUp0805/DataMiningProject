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

# importing the sbert model
filename= 'sbert_model.sav'
sbert_model= pickle.load(open(filename, 'rb'))

# importing the sbert_embedding
with open('sbert_embedding.data','rb') as filehandle:
    sbert_embedding= pickle.load(filehandle)

def cosine(u, v):
    ''' this function returns the cosine similarity b/w two vectors'''
    return np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
#--------------------------------------------------------similarity list creation-------------------------------------------------------------------

# taking in the query and embedding it
query_number= 400
query= sample_duplicates.iloc[query_number]['title']

use_query_vec = use_model([query])[0]
sbert_query_vec= sbert_model.encode([query])[0]
# Finding the similiarity scores with all the questions in our sample_data (sample of sample_limit questions)
use_simscore=[]
sbert_simscore=[]
for i in range(len(use_embedding)):
    current_question_use_embedding= use_embedding[i]
    current_question_sbert_embedding= sbert_embedding[i]
    use_sim = cosine(use_query_vec, current_question_use_embedding)
    sbert_sim= cosine(sbert_query_vec, current_question_sbert_embedding)
    use_simscore.append(use_sim)
    sbert_simscore.append(sbert_sim)

use_simscore= np.array(use_simscore)
sbert_simscore= np.array(sbert_simscore)

# computing the final similarity list
final_simscore= np.add(use_simscore*(6/7), sbert_simscore*(1/7))

# Finding the indices of the top N most similar questions
N= 7
indices= np.argpartition(final_simscore, -N)[-N:]
print(indices)
print('\n')

# printing the query question and questions that are present in these indices
print('The query question is:', sample_duplicates.iloc[query_number]['questionid'], sample_duplicates.iloc[query_number]['title'])
print('The duplicate question for this query is:', sample_duplicates.iloc[query_number]['originalquesid'], sample_data[ (sample_data['questionid']== sample_duplicates.iloc[query_number]['originalquesid']) ]['title'])
print('\nN most similar questions (not ordered) to the query question are:\n')
for i in indices:
    print(sample_data.iloc[i]['questionid'], sample_data.iloc[i]['title'])

