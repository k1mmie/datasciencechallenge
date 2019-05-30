# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 20:24:59 2017

@author: Kim.Vuong
"""

import pandas as pd
import numpy as np
import spacy 
from sklearn.preprocessing import MultiLabelBinarizer

# flatten topics to one-hot encoding (dummy variables)


f = 'data/topicDictionary.txt'# list of the 160 interesting topics
topics_df = pd.read_csv(f, header=None)
topics_df.columns = ['topics']
desired_topics = list(topics_df['topics'])
    
    
df['Filtered_Topics'] = df['Topics'].apply(lambda topics: [t for t in topics if t in desired_topics])# remove other topics not in topics_df e.g. sports, lifestyle.
    
    

mlb = MultiLabelBinarizer(classes=list(topics_df['topics'])) #Keep class in the order of the topic dictionary
Y = mlb.fit_transform(df['Filtered_Topics']) #create a dataframe of Y encoded labels [0,1,0,0,1,0]
print(mlb.classes_) 


