
##running model

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV

vectorizer = TfidfVectorizer(tokenizer=tok, preprocessor=prep,
							ngram_range=(1,3), min_df=2)
							
#pipeline model							
model = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier',OneVsRestClassifier(LinearSVC(C=100)))
        ])

docs_train, docs_test, labels_train, labels_test = train_test_split(
            X_train, Y_train, test_size=0.1, random_state=42)
    
# train
model.fit(docs_train, labels_train)
    
# test  
labels_predict = model.predict(docs_test)

#print results
print(classification_report(labels_test, labels_predict))
print(accuracy_score(labels_test, labels_predict))

#train on whole data
model.fit(X_train, Y_train) # train whole data

#predict unseen test set
Y_test = model.predict(X_test)#test data

# save file - upload file
np.savetxt("predicted_labels.csv", Y_test, delimiter=",")


#Grid search - optimizing parameters

def grid_search(X_train, Y_train):
    model = create_model()

    #get param names
    print(sorted(model.get_params()))
    
    # gridsearch params
    #look for the best hyperparameters for SVC
    #make one list and not multiples dict
    parameters = [
            {'ovr__estimator__classification__C':[10,50,100],
            ]
    
    gs = GridSearchCV(estimator=model, param_grid=parameters, cv=2, scoring='f1_micro')     
    gs.fit(X_train, Y_train) #mod to use grid search or "model" for the pipeline only
    
    # distance of your data points from the hyperplane that separates the data
    #print(mod.decision_function(docs_test))
    
    print(gs.grid_scores_)
    print(gs.cv_results_)
    print(gs.best_params_)    
