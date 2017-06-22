
import pandas as pd
import spacy 
from spacy import attrs
from spacy.tokens.doc import Doc

#load spacy
nlp = spacy.load('en_core_web_md') 
#add stop words from en_default
nlp.vocab.add_flag(lambda s: s in spacy.en.word_sets.STOP_WORDS, spacy.attrs.IS_STOP)

#load training data
f = 'trainingdata.csv'
df = pd.read_csv(f, encoding='latin')

print('using {0} samples'.format(len(df)))

#formatting the data  
df['BodyText'] = df['BodyText'].fillna('') #fill empty string
df['Topics'] = df['Topics'].apply(eval) # evaluate "strings" into real python lists
  
#run each doc through spacy for properties. do the same for test file
df['docs'] = [doc for doc in nlp.pipe(df['BodyText'], batch_size=1000, n_threads=16)]
