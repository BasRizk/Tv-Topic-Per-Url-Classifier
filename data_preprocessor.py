# -*- coding: utf-8 -*-

import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from num2words import num2words
from tqdm import tqdm
tqdm.pandas()

def tfidf_vectorize(df):
    from sklearn.feature_extraction.text import TfidfVectorizer 
 
    # settings that you use for count vectorizer will go here 
    tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True) 
    # just send in all your docs here 
    return tfidf_vectorizer.fit_transform(df)

def remove_puncuations(df):
    # Removing symbols 
    symbols = "!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        df['text'] = df['text'].str.replace(i, ' ')
    return df

def turn_num_to_words(text):
    def is_integer(n):
        try:
            float(n)
        except ValueError:
            return False
        else:
            return float(n).is_integer()
        
    processed_text = ""
    for t in text.split():
        if is_integer(t):
            processed_text += num2words(t) + " "
        else:
            processed_text += t + " "
    return processed_text

def preprocessing(df, save_vocab = True):
    df = df.dropna()
    df = df.reset_index(drop=True)
    df["text"] = df["text"].str.lower()
    
    df = remove_puncuations(df)
    
    # Turn numbers to words
    # df["text"] = df.progress_apply(lambda row: turn_num_to_words(row.text), axis=1)
    
    
    cv = CountVectorizer(stop_words='english') 
     
    # generate_word_counts
    word_count_vector = cv.fit_transform(df["text"])
    
    tfidf_transformer = TfidfTransformer(smooth_idf=True,use_idf=True) 
    tfidf_transformer.fit(word_count_vector)
    
    # # print idf values 
    # df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(),columns=["idf_weights"])  
    # # sort ascending 
    # df_idf.sort_values(by=['idf_weights'])

    if save_vocab:
        pickle.dump(cv.vocabulary_,open("feature.pkl", "wb"))
        pickle.dump(tfidf_transformer)
    # tf-idf scores 
    return df, cv, tfidf_transformer.transform(word_count_vector)

def count_vector(df, vocab_pickle):
    # TODO not complete, df preprocessing might be missing

    tfidf_transformer = TfidfTransformer()
    loaded_cv = CountVectorizer(decode_error="replace",
                                 vocabulary=vocab_pickle)
    count_vector = loaded_cv.fit_transform(df['text'])  
    # tf-idf scores 
    tf_idf_vectors = tfidf_transformer.fit_transform(count_vector)
    return loaded_cv, tf_idf_vectors

def get_X_y(filename="Dataset_with_text.csv"):  
    df = pd.read_csv(filename)
    
    df, count_vectorizer, tf_idf_vectors = preprocessing(df, save_vocab=True)
        
    feature_names = count_vectorizer.get_feature_names() 
    X = pd.DataFrame(tf_idf_vectors.todense(), columns=feature_names) 
    # tfidf_df = tfidf_df.sort_values(by=["tfidf"],ascending=False)
    y = df['class']
    return X, y

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





