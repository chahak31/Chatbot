#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Import the libraries

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import tensorflow
import json
import numpy
import random


# In[2]:


#Loading Dataset

with open('./intents.json') as file:
    data=json.load(file)
#print(data)
#print(data['intents'][0])


# In[3]:


#Processing Data

words=[]
labels=[]
docs_x=[]
docs_y=[]

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds=nltk.word_tokenize(pattern) 
        words.extend(wrds)
        docs_x.append(wrds)
        #print(docs_x)
        docs_y.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])
#print(labels)


# In[4]:


#Remove duplicates from the list words

words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
words = sorted(list(set(words)))
#print(words)

#Sort labels
labels = sorted(labels)


# In[5]:


#Neural network understands only numbers and not strings

training=[]
output=[]

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag=[]
    
    wrds = [stemmer.stem(w.lower()) for w in doc]
    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)
    
    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])]=1
               
    training.append(bag)
    output.append(output_row)


# In[6]:


#Building the model

training=numpy.array(training)
output=numpy.array(output)

#Classifying our data


# In[ ]:




