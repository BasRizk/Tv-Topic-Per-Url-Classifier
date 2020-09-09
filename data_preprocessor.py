# -*- coding: utf-8 -*-

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer

df = pd.read_csv("Dataset_with_text.csv")
df = df.dropna()
df['text'] = df['text'].str.lower()

def tfidf_vectorize(df):
    from sklearn.feature_extraction.text import TfidfVectorizer 
 
    # settings that you use for count vectorizer will go here 
    tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True) 
    # just send in all your docs here 
    return tfidf_vectorizer.fit_transform(df)


def preprocessing(df, save_vocab = False):
    # TODO Removing stopwords
    # new_text = ""
    # for word in words:
    #     if word not in stop_words:
    #         new_text = new_text + " " + word
        

    # Removing symbols 
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        df['text'] = df['text'].str.replace(i, ' ')
        
    cv = CountVectorizer(stop_words='english') 
     
    # generate_word_counts
    word_count_vector = cv.fit_transform(df["text"])
    
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(word_count_vector)
    
    # print idf values 
    df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])  
    # sort ascending 
    df_idf.sort_values(by=['idf_weights'])

    if save_vocab:
        pickle.dump(cv.vocabulary_,open("feature.pkl", "wb"))
        
    # tf-idf scores 
    return cv, tfidf_transformer.transform(word_count_vector)
    
def count_vector(df):
    tfidf_transformer = TfidfTransformer()
    loaded_cv = CountVectorizer(decode_error="replace",
                                 vocabulary=pickle.load(open("feature.pkl", "rb")))
    count_vector = loaded_cv.transform(df['text'])  
    # tf-idf scores 
    tf_idf_vector = tfidf_transformer.transform(count_vector)
    return tf_idf_vector


count_vectorizer, tf_idf_vectors = preprocessing(df, save_vocab=True)
feature_names = count_vectorizer.get_feature_names() 
#get tfidf vector for first document 
first_document_vector=tf_idf_vectors[0] 
#print the scores 
df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=["tfidf"]) 
df.sort_values(by=["tfidf"],ascending=False)


# def preprocess(data):
#     data = convert_lower_case(data)
#     data = remove_punctuation(data)
#     data = remove_apostrophe(data)
#     data = remove_single_characters(data)
#     data = convert_numbers(data)
#     data = remove_stop_words(data)
#     data = stemming(data)
#     data = remove_punctuation(data)
#     data = convert_numbers(data)








