# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
from data_preprocessor import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score  
import pandas as pd
import pickle        

# =============================================================================
# Classifiers
# =============================================================================
def svm_classifier(X_train, y_train):
    from sklearn import svm
    clf = svm.SVC(kernel='linear', gamma='scale')
    clf.fit(X_train, y_train)
    return clf

def naive_bayes_classifier(X_train, y_train,
                           nb_type="complemet_nb", verbose=False):
    clf = None
    if nb_type.lower() == "complement_nb":
        from sklearn.naive_bayes import ComplementNB
        clf = ComplementNB(alpha=1.0e-10, fit_prior=True, norm=True)
        clf.fit(X_train, y_train)
    elif nb_type.lower() == "gaussian_nb":
        from sklearn.naive_bayes import GaussianNB
        clf = GaussianNB()
        clf.fit(X_train, y_train)
    return clf

def mlp_classifier(X_train, y_train, verbose=False):
    from sklearn.neural_network import MLPClassifier
    
    clf = MLPClassifier(solver='adam', max_iter=1000, learning_rate_init=0.005,
                        alpha=1e-3, activation='relu', batch_size=4,
                        hidden_layer_sizes=(10, 20, 2), random_state=1,
                        verbose=verbose)
    clf.fit(X_train, y_train)
    return clf

# =============================================================================
# Dimensionality Reduction
# =============================================================================
def perform_dimensionality_reduction(X_train, y_train, X_test=None, rtype="lda"):
    from sklearn.preprocessing import StandardScaler
    
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    if rtype == "pca":
        from sklearn import decomposition
        pca = decomposition.PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        if X_test is not None:
            X_test = pca.transform(X_test)
    elif "lda":       
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
        lda = LDA(n_components=1)
        X_train = lda.fit_transform(X_train, y_train)
        if X_test is not None:
            X_test = lda.transform(X_test)
        # lda_var_ratios = lda.explained_variance_ratio_
        # select_n_components(lda_var_ratios, 0.95)
    
    if X_test is not None:
        return X_train, y_train, X_test
    return X_train, y_train

# =============================================================================
# Training and Testing
# =============================================================================
def print_cross_val_score(clf, X, y, cv=5):    
    import numpy as np
    from sklearn.model_selection import cross_val_score
    #train model with cv of 5 
    cv_scores = cross_val_score(clf, X, y, cv=cv)
    #print each cv score (accuracy) and average them
    print(cv_scores)
    print("cv_scores mean:{}".format(np.mean(cv_scores)))  
    
def train_model(X_train, y_train, X_test, y_test,
                model_filename="model.pkl", save=True):        
    # clf = svm_classifier(X_train, y_train)    
    clf = naive_bayes_classifier(X_train, y_train, nb_type="complement_nb",verbose=True)
    # clf = naive_bayes_classifier(X_train, y_train, nb_type="gaussian_nb",verbose=True)
    # clf = mlp_classifier(X_train, y_train, verbose=True)
    if save:
        with open(model_filename, 'wb') as f:
           pickle.dump(clf, f)
        
    return clf
        

def predict(clf, X_test, y_test=None, df=None, verbose=False):
    y_pred = clf.predict(X_test)
    
    if verbose:
        print(confusion_matrix(y_test, y_pred))
        print('Accuracy ' + str(accuracy_score(y_test, y_pred)))
        if df is not None:
            incorrects = df.loc[y_test[y_test != y_pred].index]
            return y_pred, incorrects
    return y_pred

def load_pickle(pkl_filename):
    with open(pkl_filename, 'rb') as f:
        clf = pickle.load(f)
    return clf




# =============================================================================
# Training the model
# =============================================================================
PRODUCTION = True

if not PRODUCTION:
    df = pd.read_csv("Dataset_with_text.csv") 
    
    df, count_vectorizer, tf_idf_vectors = preprocessing(df, save_vocab=True)    
    feature_names = count_vectorizer.get_feature_names() 
    X = pd.DataFrame(tf_idf_vectors.todense(), columns=feature_names) 
    # tfidf_df = tfidf_df.sort_values(by=["tfidf"],ascending=False)
    y = df['class']
    X_train, X_test, y_train, y_test =\
        train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)
       
    # X_train, y_train, X_test =\
    #     perform_dimensionality_reduction(X_train, y_train, X_test=X_test, rtype="lda")
    
    clf = train_model(X_train, y_train, X_test, y_test,
                      model_filename="model.pkl", save=True)   
    print_cross_val_score(clf, X, y, cv=5)
    
    y_pred, incorrects = predict(clf, X_test, y_test=y_test, df=df, verbose=True)


