#!/usr/bin/env python
# coding: utf-8

# In[47]:


#!pip install nltk
get_ipython().system('pip install GoogleNews')


# In[2]:


import pandas as pd
import numpy as np
import time


# In[4]:


#Scraping articles on gun control from Google News

from GoogleNews import GoogleNews
import pandas as pd


# Create a GoogleNews object
googlenews = GoogleNews()

# Set search parameters
search = 'gun control'
googlenews.search(search)

# Get the first 10 pages of results
articles = []
for i in range(1, 50):
    googlenews.getpage(i)
    articles += googlenews.result()
    time.sleep(3)

# Convert articles to a pandas dataframe
df_long = pd.DataFrame(articles)

# Display the dataframe
print(df_long.head())



# In[5]:


df_long = df_long.drop_duplicates()
df_long = df_long.reset_index()
df_long


# In[6]:


df_long = df_long.drop(['index', 'datetime','img'], axis = 1)
df_long.head()


# In[41]:


len(df_long.media.unique())


# In[9]:


Liberal = ['ABC News','The Washington Post','Daily Hampshire Gazette', 'The Guardian', 'Axios', \
          "The Arkansas Democrat-Gazette",'Globe Echo','New York Daily News', 'CBS News', \
          "Denver7", "The Advocate",'Washington Blade', 'West Hawaii Today', \
          'NPR','USA Today', 'ABC7 Chicago', 'Yahoo News', 'CNN', 'The New York Times', 
          'Baltimore Sun','Colorado Newsline', 'The Denver Post', 'WPLG Local 10', 'The Colorado Sun', \
          'ProPublica', 'AP News', 'Atlanta Journal-Constitution', 'Insider', 'Michigan Advance', 'Crosscut', \
          'Tennessee Lookout', 'TIME', 'Democracy Now!', 'Vox', 'KARE 11', 'New Jersey Monitor', 'ABC7', \
          'NBC News', 'Reuters', 'Los Angeles Times']

x = df_long.media.unique()
x.astype(list)
x
y = []
z = []
for i in x:
    if i in Liberal: 
        z.append(i)
    else:
        y.append(i)
print(y)
print(z)


# In[11]:


df_long.head()


# In[10]:


update = ['Shockya','CGTN','National World','MSN','Euronews','WFDD','DW','Morning Star','Marietta Daily Journal','The Public''s Radio',\
         'Newsweek','Goshen News','Atlanta Daily World','HuffPost','News Channel 5','Slate Magazine','San Antonio Express-News', \
         'Portland Mercury','The Texas Tribune','Austin American-Statesman','MPR News','Salon.com','The Tennessean','ABC27', \
         'Lincoln Journal Star','WHYY','Pioneer Press','Politico','Star Tribune','The New Republic','Maryland Daily Record',\
         'The Seattle Times','PBS','WLKY','CBS Austin','WPRI.com','Cap Times','NBC15','The Detroit News','Action News 5',\
         'Cincinnati Enquirer','Security Magazine','ESPN','NAACP','Chalkbeat Tennessee','Everytown Research & Policy',\
         'The Portland Press Herald','Maine Public','PolitiFact','Minnesota Reformer','The Harvard Crimson','The Fresno Bee',\
         'Wcvb-tv','Bridge Michigan','The Spokesman-Review','The Marshall Project','Courier Post','MSNBC']
Liberal.extend(update)
Liberal


# In[12]:


df_long['Left-Leaning'] = ''


# In[13]:


for i in range(0,len(df_long)):
    if df_long.iloc[i,1] in Liberal:
        df_long.iloc[i,5] = 1
    else:
        df_long.iloc[i,5] = 0
df_long.head()


# In[286]:


#Sources that have political leanings

#Liberal = ['ABC News','The Washington Post','Daily Hampshire Gazette', 'The Guardian', 'Axios', \
#          "The Arkansas Democrat-Gazette",'Globe Echo','New York Daily News', 'CBS News', \
#          "Denver7", "The Advocate",'Washington Blade', 'West Hawaii Today', \
#          'NPR','USA Today', 'ABC7 Chicago', 'Yahoo News', 'CNN', 'The New York Times', 
#          'Baltimore Sun','Colorado Newsline', 'The Denver Post', 'WPLG Local 10', 'The Colorado Sun', \
#          'ProPublica', 'AP News', 'Atlanta Journal-Constitution', 'Insider', 'Michigan Advance', 'Crosscut', \
#          'Tennessee Lookout', 'TIME', 'Democracy Now!', 'Vox', 'KARE 11', 'New Jersey Monitor', 'ABC7', \
#          'NBC News', 'Reuters', 'Los Angeles Times']
for i in range(0,len(df)):
    if df.iloc[i,1] in Liberal:
        df.iloc[i,4] = 1
    else:
        df.iloc[i,4] = 0

df = df[['title','media','date','desc','Liberal']]
df


# In[14]:


df_long['Left-Leaning'].value_counts()


# In[199]:


import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

df_copy = df2


# Remove non-alphanumeric characters
df_copy['clean_title'] = df['title'].apply(lambda x: re.sub(r'\W+', ' ', x).strip().lower())

# Tokenize the text
df_copy['tokenized_text'] = df_copy['clean_title'].apply(lambda x: word_tokenize(x))

# Remove stop words and stem the remaining words
stop_words = stopwords.words('english')
stemmer = PorterStemmer()

df_copy['processed_text'] = df_copy['tokenized_text'].apply(lambda x: [stemmer.stem(word) for word in x if word not in stop_words])


# In[84]:


### Using Naive Bayes Classifiers with specific list of words and extra factors

import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

df = df_long

# Clean and preprocess the data


df["clean_title"] = df["title"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))
df['description'] = df["desc"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))
df['title_length'] = df['title'].apply(lambda x: len(x.split()))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df[["clean_title","title_length",'description']], df["Left-Leaning"], test_size=0.2, random_state=38)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# define list of additional words to include in vocabulary
additional_words = ["background checks", "firearm regulation", "gun control",\
                   "gun safety", "safety","ban"\
                   "worry",'gun bills', 'violence','gop']

vectorizer = CountVectorizer(vocabulary= additional_words)

#Turn the data into a vectorized format
# Convert the title length to a numpy array and concatenate it with the vectorized training data
title_length_train = X_train['title_length'].to_numpy().reshape(-1, 1)

X_train_vectorized = pd.DataFrame(X_train_vectorized.toarray())
X_train_vectorized = pd.concat([X_train_vectorized, pd.DataFrame(title_length_train)], axis=1)
title_length_test = X_test['title_length'].to_numpy().reshape(-1, 1)
X_test_vectorized = pd.DataFrame(X_test_vectorized.toarray())
X_test_vectorized = pd.concat([X_test_vectorized, pd.DataFrame(title_length_test)], axis=1)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test data
predictions = classifier.predict(X_test_vectorized)

# Evaluate the performance of the classifier
accuracy = np.mean(predictions == y_test)
print('Accuracy:', accuracy)
display('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred, target_names = None))


# In[83]:


### Using Naive Bayes Classifiers on the processed title

# Clean and preprocess the data

df = df_long

df["clean_title"] = df["title"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df["clean_title"], df["Left-Leaning"], test_size=0.2, random_state=42)

y_train = y_train.astype(int)
y_test = y_test.astype(int)
# Convert the text data into numerical features using a bag-of-words approach


vectorizer = CountVectorizer()

#Turn the data into a vectorized format

X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train a Naive Bayes classifier
classifier = MultinomialNB()
classifier.fit(X_train_vectorized, y_train)

# Make predictions on the test data
predictions = classifier.predict(X_test_vectorized)

# Evaluate the performance of the classifier
accuracy = np.mean(predictions == y_test)

print('Accuracy:', accuracy_score(y_test, predictions))
display('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred, target_names = None))


# In[111]:


### Naive Bag of Words Approach with extra features

df = df_long

#preprocess and clean data

df["clean_title"] = df["title"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))
df['description'] = df["desc"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))
df['title_length'] = df['title'].apply(lambda x: len(x.split()))

### Split data

X_train, X_test, y_train, y_test = train_test_split(df[['clean_title', 'title_length', 'description']], df['Left-Leaning'], test_size=0.2, random_state=40)


y_train = y_train.astype(int)
y_test = y_test.astype(int)

### Vectorizing Data

vectorizer_title = CountVectorizer()
vectorizer_desc = CountVectorizer()

X_train_title = vectorizer_title.fit_transform(X_train['clean_title'])
X_train_desc = vectorizer_desc.fit_transform(X_train['description'])

### Combine the trained data

X_train_combined = pd.concat([pd.DataFrame(X_train_title.toarray()), pd.DataFrame(X_train_desc.toarray()), X_train['title_length'].reset_index(drop=True)], axis=1)

X_train_combined.columns = X_train_combined.columns.astype(str)

### Fit the model

classifier = MultinomialNB()
classifier.fit(X_train_combined, y_train)

X_test_title = vectorizer_title.transform(X_test['clean_title'])
X_test_desc = vectorizer_desc.transform(X_test['description'])
X_test_combined = pd.concat([pd.DataFrame(X_test_title.toarray()), pd.DataFrame(X_test_desc.toarray()), 
                             X_test['title_length'].reset_index(drop=True)], axis=1)

X_test_combined.columns = X_test_combined.columns.astype(str)

predictions = classifier.predict(X_test_combined)


print('Accuracy:', accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))


# In[205]:


### Run this code after the cell above for correct results
### Using random forrest technique 

df = df_long

#preprocess and clean data

df["clean_title"] = df["title"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))
df['description'] = df["desc"].apply(lambda x: re.sub(r"[^a-zA-Z0-9\s]", "", x.lower()))
df['title_length'] = df['title'].apply(lambda x: len(x.split()))

X_train, X_test, y_train, y_test = train_test_split(df[['clean_title', 'title_length', 'description']], df['Left-Leaning'], test_size=0.2, random_state=40)


y_train = y_train.astype(int)
y_test = y_test.astype(int)

### Vectorizing Data

vectorizer = CountVectorizer(stop_words = 'english')

X_title_vectorized =  pd.DataFrame(vectorizer.fit_transform(X_train['clean_title']))
X_description_vectorized =  pd.DataFrame(vectorizer.fit_transform(X_train['description']))
X_title_length = pd.DataFrame(X_train['title_length'])
X_train_vectorized = pd.concat([X_title_vectorized, X_title_length, X_description_vectorized], axis=1)

X_train_vectorized.columns = X_train_vectorized.columns.astype(str)


#X_train = pd.concat([X_train_title, X_train_title_length, X_train_desc], axis=1)



classifier = RandomForestClassifier(n_estimators=100, max_depth = 5, random_state=42)

# Fit the classifier to the training data
classifier.fit(X_train_combined, y_train)


#X_test_title_vectorized = title_vectorizer.transform(X_test['title']).toarray()
#X_test_title_length = X_test['title'].apply(len).values.reshape(-1,1)
#X_test_desc_vectorized = desc_vectorizer.transform(X_test['description']).toarray()

#X_test = pd.concat([pd.DataFrame(X_test_title_vectorized), pd.DataFrame(X_test_title_length),
                    #pd.DataFrame(X_test_desc_vectorized)], axis=1)

# Convert titles to numerical features using count vectorizer
#vectorizer = CountVectorizer(stop_words='english', max_features=1000)
#X = vectorizer.fit_transform(df['title'].values.astype('U')).toarray()

y_pred = classifier.predict(X_test)





# Set "Liberal" as target variable
#y = df['Left-Leaning']

# Split data into training and testing sets
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

#y_train = y_train.astype(int)
#y_test = y_test.astype(int)

# Create and train random forest classifier
#rf = RandomForestClassifier(n_estimators=100, random_state=42)
#rf.fit(X_train, y_train)

# Predict test data
#y_pred = rf.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
display('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred, target_names = None))


# In[100]:


### Using random forrest technique 

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

df = df_long

# Convert titles to numerical features using count vectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['title'].values.astype('U')).toarray()



# Set "Left-Leaning" as target variable
y = df['Left-Leaning']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=59)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Create and train random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict test data
y_pred = rf.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
display('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred, target_names = None))


# In[46]:


#random forest with hyperparamter tuning

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfTransformer

df = df_long
# Convert titles to numerical features using count vectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['title'].values.astype('U')).toarray()



# Set "Liberal" as target variable
y = df['Left-Leaning']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=17)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Define the parameter grid to search over
param_grid = {
    'n_estimators': [5, 10, 15, 20],
    'max_depth': [5, 10, 20, 30, None],
    'min_samples_split': [2, 5, 10,15],
    'min_samples_leaf': [1, 2, 3, 4],
    'max_features': ['sqrt', 'log2', None]
}

# Create a random forest classifier
rf = RandomForestClassifier(random_state=42)

# Use RandomizedSearchCV to find the best hyperparameters
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=param_grid, n_iter=100, cv=5, random_state=59)
rf_random.fit(X_train, y_train)

# Print the best hyperparameters
print("Best hyperparameters:", rf_random.best_params_)

# Create a new random forest classifier with the best hyperparameters
rf = RandomForestClassifier(**rf_random.best_params_, random_state=59)

# Fit the classifier to the training data
rf.fit(X_train, y_train)

# Predict on the test data
y_pred = rf.predict(X_test)

# Print the accuracy score
print("Accuracy:", accuracy_score(y_test, y_pred))
confusion_mat = confusion_matrix(y_test, y_pred)
display('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred, target_names = None))


# In[47]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# Assume "df" is your pandas dataframe with columns "Title" and "Favoring Gun Control"

# Convert titles to numerical features using count vectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['title'].values.astype('U')).toarray()

# Set "Liberal" as target variable
y = df['Left-Leaning']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=59)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Create and train random forest classifier
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict test data
y_pred = rf.predict(X_test)

# Calculate accuracy and confusion matrix
accuracy = accuracy_score(y_test, y_pred)
confusion_mat = confusion_matrix(y_test, y_pred)
print('Accuracy:', accuracy)
display('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred, target_names = None))


# In[48]:


### Using K Nearest Niehbor with K = 5

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df = df_long
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['Left-Leaning'], test_size=0.2, random_state=59)

y_train = y_train.astype(int)
y_test = y_test.astype(int)

# Create a TF-IDF vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)
# Transform the testing data
X_test_vec = vectorizer.transform(X_test)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train_vec, y_train)

# Predict on the test data and calculate accuracy
y_pred = knn.predict(X_test_vec)
accuracy = (y_pred == y_test).sum() / len(y_test)

confusion_mat = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy}")
display('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred, target_names = None))


# In[51]:


### Using K Nearest Neighbor with K= 3

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


df = df_long
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['title'], df['Left-Leaning'], test_size=0.25, random_state=59)

y_train = y_train.astype(int)
y_test = y_test.astype(int)
# Create a TF-IDF (term frequency-inverse document frequency) vectorizer
vectorizer = TfidfVectorizer(stop_words='english')

# Fit and transform the training data
X_train_vec = vectorizer.fit_transform(X_train)


# Transform the testing data
X_test_vec = vectorizer.transform(X_test)

# Create and train the KNN classifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train_vec, y_train)

# Predict on the test data and calculate accuracy
y_pred = knn.predict(X_test_vec)
accuracy = (y_pred == y_test).sum() / len(y_test)
print(f"Accuracy: {accuracy}")
display('Confusion Matrix:', confusion_mat)
print(classification_report(y_test, y_pred, target_names = None))

