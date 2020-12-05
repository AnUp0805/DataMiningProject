import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import pandas as pd
import pickle

sample_data= pd.read_csv('prepared_data.csv')
sample_duplicates= pd.read_csv('prepared_duplicates.csv')
sample_limit= len(sample_data)
# Taking the first sample_limit sample questions
sample_data= sample_data[:sample_limit]

with open('use_simscores.data','rb') as filehandle:
    use_simscores= pickle.load(filehandle)

correctly_predicted=0
for i in range((use_simscores.shape)[0]):
    simscore_for_current_query= use_simscores[i]
    # Finding the indices of the top N most similar questions
    N= 7
    indices= np.argpartition(simscore_for_current_query, -N)[-N:]
    # finding the IDs of the predicted similar questions
    predicted_similar_ID=[]
    for index in indices:
        predicted_similar_ID.append( sample_data.iloc[index]['questionid'] )
    # we now have the IDs of the predicted similar questions
    actual_similar_ID= sample_duplicates.iloc[i]['originalquesid']
    # if the actual similar question ID is present in the predicted similar question IDs then I increment the correct counter
    if (actual_similar_ID in predicted_similar_ID):
        correctly_predicted+=1
accuracy= float(correctly_predicted)/(len(sample_duplicates))*100
accuracy= round(accuracy, 3)
print('The accuracy of the use model is:', str(accuracy)+'%')