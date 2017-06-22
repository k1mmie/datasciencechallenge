from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS

#filter if stop or punctuation ,"SYM", "NUM"
def filter_token(tok):
    return tok.is_stop or tok.is_punct or tok.pos_ in ["PUNCT","SYM", "NUM"]\
            or tok.tag_ in ["POS", "VBZ", "CD", "VB", "RB"]\
            or tok.like_email or tok.is_space or tok.like_num or tok.text in ['+']\
            or tok.lower_ in ENGLISH_STOP_WORDS\
            #or len(tok.text) < 3
#, "CD","VB","RB"            

#replace tokens with either Lemma or named entity
def preprocess(doc):
    # merge entities
    for ent in doc.ents:
        if len(ent) > 1:
            ent.merge(ent.root.tag_, ent.text, ent.label_)
    
    # return our tokens
    return [entity_or_lemma(tok) for tok in doc if not filter_token(tok)]
	

def preprocess_tokens(df):
    df['tokens'] = df['docs'].apply(preprocess)
    return df