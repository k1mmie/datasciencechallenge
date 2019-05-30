
#need to run 'labels.py' and 'naively inferring data.py' first on the dataset to ensure we get a sample of all topics
Y_flat = pd.DataFrame(Y, columns=mlb.classes_)
df = pd.concat([df, Y_flat], axis=1)

# (optional) cycles through the Y labels picking up the max number of topics until reaches 50k. prioritize topics with lower number of documents first.
# We do this because laptop couldn't deal with the huge amount of data.  Not needed if memory isn't an issue. 
indexes = []
for col in itertools.cycle(mlb.classes_):
    if len(indexes) >= 50000:
        break
    
    potential = df[df[col] == 1].sample(1).index[0]
    
    if potential in indexes:
        continue
    
    indexes.append(potential)
    
df_sample = df.ix[indexes]
