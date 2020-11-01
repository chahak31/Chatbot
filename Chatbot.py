#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import the libraries

import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer=LancasterStemmer()
import tensorflow
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tflearn
import json
import numpy
import random
import pickle


# In[ ]:


#Loading Dataset

with open('./intents.json') as file:
    data=json.load(file)
#print(data)
#print(data['intents'][0])


# In[ ]:


#Processing Data
#rb -- read bytes
try:
    with open("data.pickle","rb") as f:
        words, labels, training, output = pickle.load(f)
except:
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

    #Remove duplicates from the list words

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    #print(words)

    #Sort labels
    labels = sorted(labels)

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

    #Building the model

    training=numpy.array(training)
    #print(training)
    output=numpy.array(output)
    
    with open("data.pickle","wb") as f:
        pickle.dump((words, labels, training, output), f)


# In[ ]:


#Classifying our data

#input layer
net = tflearn.input_data(shape=[None,len(training[0])])
print(len(training[0]))
print(len(output[0]))
#two hidden layers
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)

#output layer
net = tflearn.fully_connected(net,len(output[0]),activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

#Training the model
try: 
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000,batch_size=8, show_metric=True)
    model.save("model.tflearn")


# In[ ]:


#Predictions 

def bag_of_words(s,words):
    bag=[0 for _ in range(len(words))]
    
    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]
    
    for se in s_words:
        for i, w in enumerate(words):
            if w==se:
                bag[i]=1
    return numpy.array(bag)


# In[ ]:


def chat():
    print("Start talking with the bot(enter quit to exit)")
    while True:
        inp = input("You: ")
        if inp.lower() =="quit":
            break
        
        results=model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]
        
        for tg in data['intents']:
            if tg['tag'] == tag:
                responses = tg['responses']
        
        print(random.choice(responses))

chat()


# In[ ]:




