from tkinter import *
import tkinter.font as font
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


# loading the pre-trained model
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
#------------------------------------------------------------------------------------------------------------------------------------------------------#
root= Tk()
#myFont = font.Font(family='Helvetica')
myLabel1= Label(root, text= 'Please enter the query question in the box and hit submit!')
myLabel1.configure(font=("Times New Roman", 12, "bold"))
myLabel1.pack()
e= Entry(root, width=50, borderwidth= 2)
e.pack()
query=''
def onClick():
    #myLabel2= Label(root, text= 'The query is submitted. Please wait for the results!')
    #myLabel2.pack()
    query= e.get()
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
    final_simscore= np.add(use_simscore*(5/6), sbert_simscore*(1/6))

    # Finding the indices of the top N most similar questions
    N= 7
    indices= np.argpartition(final_simscore, -N)[-N:]
    #print(indices)
    #print('\n')

    # printing the query question and questions that are present in these indices
    similar_questions=[]
    #print('The query question is:', query)
    #print('\nN most similar questions (not ordered) to the query question are:\n')
    for i in indices:
        similar_questions.append(sample_data.iloc[i]['title'])
        #print(sample_data.iloc[i]['questionid'], sample_data.iloc[i]['title'])
    # displaying the query question and the top N similar questions in the GUI
    #my_text= Text(root)
    my_text.insert(END, 'The query question is: '+str(query)+'\n')
    my_text.insert(END, 'The 7 most similar questions to this query are:\n')
    for q in similar_questions:
        my_text.insert(END, q+'\n')
    #my_text.pack()

def clearClick():
    my_text.delete('1.0', END)
    e.delete(0,END)

myButton= Button(root, text='Submit', command= onClick)
myButton.configure(font=("Times New Roman", 12, "bold"))
myButton.pack()
my_text= Text(root)
my_text.configure(font=("Times New Roman", 12, "bold"))
my_text.pack()
clearButton= Button(root, text='Clear', command= clearClick)
clearButton.configure(font=("Times New Roman", 12, "bold"))
clearButton.pack()
root.mainloop()