# -*- coding: utf-8 -*-
"""
@author: Kim.Vuong
"""
#import libraries

## infer topics from words in articles and add new topics

f = 'articles.csv'
df = pd.read_csv(f, encoding='latin')
df['strTopics'] = df['Topics']
df['Topics'] = df['Topics'].apply(eval)
df['BodyText'] = df['BodyText'].fillna('')


df['strTopics'] = df['strTopics'].astype(str)
   
def check_contains_label(df):
    for index, row in df.iterrows():
        if 'australiaguncontrol' in row['strTopics']:
                row['Topics'].append('australianguncontrol')

check_contains_label(df)
