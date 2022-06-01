#Import packages
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics 
from sklearn.feature_extraction.text import TfidfVectorizer



#import train
train = pd.read_json('train-1.json')

#import test and rearange columns
test = pd.read_json('test.json')
test = test[['doi',
 'title',
 'abstract',
 'authors',
 'venue',
 'year',
 'references',
 'topics',
 'is_open_access',
 'fields_of_study']]




#cleaning data
def handle_na(df):
    if df['year'].isnull().any() == True:
        df['year'].fillna((df['year'].median()), inplace=True)
        #https://stackoverflow.com/questions/18689823/pandas-dataframe-replace-nan-values-with-average-of-columns
        
    #replacing missing fields_of_study with most likely
    #source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
    df['fields_of_study'].fillna(method = 'ffill', inplace=True)
    
    #This was unfortanetly not possible for the abstract, so we left it empty
    df['abstract'].fillna('', inplace=True)


def remove_brackets(df):
    df['fields_of_study'] = df['fields_of_study'].apply(lambda x: ','.join(set(x)))
    df['topics'] = df['topics'].apply(lambda x: ','.join(set(x)))
    df['authors'] = df['authors'].apply(lambda x: ','.join(set(x)))
        
handle_na(train)
handle_na(test)    
remove_brackets(train)
remove_brackets(test)



def add_count_features(df):
    #Inspiration from: https://predictivehacks.com/?all-tips=word-counts-in-pandas-data-frames
    df['authors_count'] = df['authors'].str.split(',').str.len()
    df['abstract_count'] = df['abstract'].str.split().str.len()
    df['topics_count'] = df['topics'].str.split().str.len()
    df['venue_count'] = df['venue'].str.split().str.len()
    df['title_count'] = df['title'].str.split().str.len()
    
    
add_count_features(train)
add_count_features(test)




def study_categories(df):
    df['study_categories'] = pd.factorize(df.fields_of_study)[0]
    df.drop(['fields_of_study'], axis = 1, inplace=True)
    
study_categories(train)
study_categories(test)




#removing outliers in train dataset
q_low = train["references"].quantile(0.00)
q_hi  = train["references"].quantile(0.95)    
train = train[(train["references"] < q_hi) & (train["references"] > q_low)]  




def log(df, column):
    df[column] += 1 #smoothing
    df[column] = np.log(df[column])

    
log(train, 'citations') 



#TF IDF
#Source TF IDF: https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html
#https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 150, stop_words = 'english')
tf = vectorizer.fit_transform(train['abstract'])
df_bow = pd.DataFrame(tf.toarray(), columns=vectorizer.get_feature_names_out())
train = pd.concat([train.reset_index(drop=True), df_bow], axis=1)

tf = vectorizer.transform(test['abstract'])
df_bow = pd.DataFrame(tf.toarray(), columns=vectorizer.get_feature_names_out())
test = pd.concat([test.reset_index(drop=True), df_bow], axis=1)

vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 200, stop_words = 'english')
tf = vectorizer.fit_transform(train['title'])
df_bow = pd.DataFrame(tf.toarray(), columns=vectorizer.get_feature_names_out())
train = pd.concat([train.reset_index(drop=True), df_bow], axis=1)

tf = vectorizer.transform(test['title'])
df_bow = pd.DataFrame(tf.toarray(), columns=vectorizer.get_feature_names_out())
test = pd.concat([test.reset_index(drop=True), df_bow], axis=1)

vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 100)
tf = vectorizer.fit_transform(train['topics'])
df_bow = pd.DataFrame(tf.toarray(), columns=vectorizer.get_feature_names_out())
train = pd.concat([train.reset_index(drop=True), df_bow], axis=1)

tf = vectorizer.transform(test['topics'])
df_bow = pd.DataFrame(tf.toarray(), columns=vectorizer.get_feature_names_out())
test = pd.concat([test.reset_index(drop=True), df_bow], axis=1)

vectorizer = TfidfVectorizer(analyzer = 'word', max_features = 100)
tf = vectorizer.fit_transform(train['venue'])
df_bow = pd.DataFrame(tf.toarray(), columns=vectorizer.get_feature_names_out())
train = pd.concat([train.reset_index(drop=True), df_bow], axis=1)

tf = vectorizer.transform(test['venue'])
df_bow = pd.DataFrame(tf.toarray(), columns=vectorizer.get_feature_names_out())
test = pd.concat([test.reset_index(drop=True), df_bow], axis=1)




drop = ['citations', 'doi', 'authors', 'title', 'venue', 'topics', 'abstract']
citations = train['citations']
y_train = citations
X_train = train.drop((drop), axis = 1)


#Model: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
#Best parameters where found using RandomizedSearchCV
model = GradientBoostingRegressor(loss = 'squared_error',
                                  n_estimators = 225,
                                  learning_rate =0.15,
                                  subsample = 0.7,
                                  min_samples_leaf = 10,
                                  max_depth = 2,
                                  random_state = 123)

model.fit(X_train, y_train)





#Doing predictions on test json file
test_for_pred = test.drop((drop[1:]), axis = 1)
test_pred = model.predict(test_for_pred)

pred = pd.DataFrame()
pred['doi'] = test['doi']

#https://numpy.org/doc/stable/reference/generated/numpy.exp.html
test_pred = np.exp(test_pred)
pred['citations'] = test_pred



#Writing to predictions file
#Source used: https://stackabuse.com/reading-and-writing-json-to-a-file-in-python/
out = pred.to_json(orient='records', indent = 4, double_precision = 4).replace("\/", '/')
with open('predictions.json', 'w') as f:
    f.write(out)

