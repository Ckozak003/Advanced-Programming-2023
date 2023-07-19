#!/usr/bin/env python
# coding: utf-8

# # HW#4

# In[101]:


import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from datetime import date, datetime


# 1. Open "Credit Card Info.xlsx" and load table.
# <br>1. Use Regular expression to extract credit card number 
# <br>2. provide type of card "Master Card", "Visa", "Invalid Card Number"
# <br>Master Card - Always 16 digit long, begings with a 5
# <br>Visa - Always 16 digits long, beginning with a 4
# 

# In[97]:


Credits = pd.read_excel('/Users/chrisk/Desktop/Adv Program/Credit Card Info.xlsx')
Credits = Credits.rename(columns={'What is your Credit Card':'Credit_Card_Num' })
Credits['Card_type'] = '' 
numbers = re.compile("[^0-9]")
display(Credits)
for i in range(len(Credits)):
    raw = Credits.iloc[i,1]
    raw = numbers.sub('',raw)
    Credits.iloc[i,1] = raw
    if len(Credits.iloc[i,1]) != 16:
        Credits.iloc[i,2] = 'Invalid Card Number'
    elif Credits.iloc[i,1][0] == '4':
        Credits.iloc[i,2] = 'Visa'
    elif Credits.iloc[i,1][0] == '5':
        Credits.iloc[i,2] = 'Master Card'
    else:
        Credits.iloc[i,2] = 'Invalid Card Number'
    

Credits


# 2. Pull Lebron James' Tweets from Pickle file

# In[291]:


import pickle
LBJ = pickle.load( open( "KingJames_tweets.sav", "rb" ) )
LBJ


# 3. Plot out LBJ's tweets by year, month, durng what month do we see high tweet counts?

# In[414]:


import pickle
LBJ = pickle.load( open( "KingJames_tweets.sav", "rb" ) )
LBJ

copy = LBJ
copy = pd.DataFrame(copy['date'])    
copy['date'] =  pd.to_datetime(copy['date'])
copy['month_year'] = copy['date'].apply(lambda x: f'{x.year}-{x.month}')

counts = copy.groupby(copy['month_year']).count()
display(counts)
counts = counts.reset_index()


plt.figure().set_figwidth(15)
plt.bar(counts['month_year'], counts['date'], )
plt.xticks(rotation=90)
plt.xlabel('Month-Year')
plt.ylabel('Count')
plt.title('# of Tweets by Month and Year')
plt.show()


# Lebron tweeting the most in July of 2021!

# 4. Which Hashtags did LBJ use mostly?

# In[136]:


htags = LBJ['hashtags']
htags.value_counts()
#JamesGang


# The most frequent hashtag used by LBJ is JamesGang.

# 5. In tweets, we see a lot of tweets with web addresses, write a regex term that identify web address

# In[259]:


pattern = re.compile('^(https).*$\s | ^(https).*$\..{,3}')


# The idea behind this expression is to capture anything that starts with the generic website address characters followed by a space OR anything that starts with the generic website starter and is at the end of the tweet. I thought about hyperlinks that may not have this format, but was unsure of how (if it is even possible) to capture this kind of web address.

# In[231]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import metrics
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


# In[232]:


### See https://en.wikipedia.org/wiki/Iris_flower_data_set for description of dataset

from sklearn import datasets
iris = datasets.load_iris()


# In[247]:


### for X, there are 4 variables, sepal length, sepal width, petal length, petal width
### for target = 0: setosa, target = 1: versicolor, target = 2: virginica
X = iris.data
Y = iris.target
Description = iris.target_names


# In[248]:


Description


# In[249]:


X.shape


# 6. let's build a dataframe with the first 2 variables from X(Sepal length, sepal width) and target(Y), and plot it out. Use Sepal length/sepal width as axis, and target as color. Do same colors "Closely grouped together"?

# In[251]:


X_new = pd.DataFrame(X)
X_new = X_new[[0,1]]
display(X_new)
Y_new = pd.DataFrame(Y)
Y_new = Y_new.rename({0 : 'Target'}, axis = 1)
comb = X_new.join(Y_new, how = 'left')
comb
for i in range(0,len(comb)):
    if comb.iloc[i,2] == 0:
        plt.plot(X_new.iloc[i,0],X_new.iloc[i,1], 'ro')
        
        continue
    elif comb.iloc[i,2] == 1:
        plt.plot(X_new.iloc[i,0],X_new.iloc[i,1], 'bo')
        continue
    else:
        plt.plot(X_new.iloc[i,0],X_new.iloc[i,1], 'yo')
        continue


# Each color is generally grouped together with versicolor and virginica having similar sepal lengths and widths in general.

# 7. Only use the first 2 variable from X, randomize data, split dataset into 120 as training and 30 as testing. Build KNN model around it. Then print out accuracy and confusion matrix. How good is our model?

# In[254]:


shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]
X2 = pd.DataFrame(X)
X2 = X2[[0,1]]
train_x = X2[:120]
train_y = Y[:120]

test_x = X2[120:]
test_y = Y[120:]



k=3
knn_model = KNeighborsClassifier(n_neighbors = k)

knn_model.fit(X = train_x, y = train_y)

y_pred = knn_model.predict(test_x)

        
print('Accuracy for k = {0:d} is: {1:.3f}'.format(k, accuracy_score(test_y, y_pred)))
        
print(classification_report(test_y, y_pred, target_names = None))


# Not shown here, but increasing k decreased the accuracy (k = 5 resulted in an accuracy rating of .7).

# 8. rebuild model with all variables from X, do we see any improvement?

# In[258]:


X = iris.data
Y = iris.target
shuffle = np.random.permutation(np.arange(X.shape[0]))
X, Y = X[shuffle], Y[shuffle]
train_x = X[:120]
train_y = Y[:120]

test_x = X[120:]
test_y = Y[120:]

k=3
knn_model = KNeighborsClassifier(n_neighbors = k)

knn_model.fit(X = train_x, y = train_y)

y_pred = knn_model.predict(test_x)

        
print('Accuracy for k = {0:d} is: {1:.3f}'.format(k, accuracy_score(test_y, y_pred)))
        
print(classification_report(test_y, y_pred, target_names = None))


# The accuracy shows a marked improvement!
