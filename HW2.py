#!/usr/bin/env python
# coding: utf-8

# # Advanced Python HW#2

# In[1]:


#imports and such
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# 1. Please load titanic data titanic.csv

# In[76]:


data = pd.read_csv('/Users/chrisk/Desktop/Jupyter/titanic.csv')
data.head(14)


# 2. Let's find some basic information about the dataset
# a. How many unique passengers do we have?
# b. What is average age of all passengers
# c. How many male vs. female
# d. How many people survived in the end?

# In[26]:


display(data.describe())
data.sex.value_counts()
data.survived.value_counts()


# a.) There were 1309 unique passengers
# b.) The average age of the 1309 passengers was 29.88 (about 30)
# c.) There were 843 men and 466 women
# d.) Out of the 1309 passengers, 500 survived and 809 did not.

# 3. What's the average cost of tickets by class (pclass)? Which ticket class has biggest price range difference?

# In[9]:


cost_by_class = data['fare'].groupby(data.pclass)
cost_by_class.describe()


# The average cost for 1st class was 87.51, 2nd class was 21.18, and 3rd class was 11.49. The class with the biggest price range was 1st class.

# 4. How many passengers have "home.dest" in the USA? <br>
# <b>Hint:</b> US has state code. US state code is different from Canada

# In[111]:


data = pd.read_csv('/Users/chrisk/Desktop/Jupyter/titanic.csv')
pd.set_option('display.max_rows', None)
data['US_home'] = np.where(data['home.dest'].str.contains(', *\w') & (data['home.dest'].isna()==False), '1', '0')
data['US_home'].value_counts()


# There were 681 US bound passnegers on the trip.

# 5. Which ticket class (pclass) has more percentage of passengers with home.dest in USA?

# In[138]:


pivot = pd.pivot_table(data, values = 'sex', index = 'pclass',columns = 'US_home', aggfunc=('count'))
pivot


# 1st class had the largest percentage of people with home.dest in the USA.

# 6. Create a column called "Age Group" and define Children as anyone less than 18 years old, Adult as anyone between 18 and 60, and elderly as anyone who is greater than 60.

# - Break data down by Age Group and Gender, <br>
# - Which group is more likely to have ticket pclass = 1? <br>
# - Which segment has higher chance of survival? <br>
# 

# In[186]:


data['Age_Group'] = pd.cut(data.age, [0,18,60,999], labels = ('Child','Adult','Elderly'))
group2 = data['sex'].groupby([data.Age_Group,data.pclass,data.survived])
sumdf2 = group2.agg('count')
sumdf2


# In[208]:


pivot2 = pd.pivot_table(data, values = 'survived', index = 'Age_Group',columns = ['sex','pclass'], aggfunc = ['sum'])
display(pivot2)
pivot2.describe()


# Adults are the most likely to have a first class ticket. The group with the highest survival rate is adult women in first class.

# 7. Do higher pclass customers have higher of survival? Among Class 1, do femal or male more likely to survive? How about class 3?

# Generally speaking the higher the class the greater the survival rate. This is not true for 3rd class men however where on average more 3rd class men survived than 2nd and 1st. Among class 1, females are more likely to survive than males. Among class 3 the same is true.

# 8. Any other interesting insight you can find?

# In[206]:


pivot2 = pd.pivot_table(data, values = 'fare', index = 'survived',columns = ['sex','pclass'], aggfunc = ['mean'])
pivot2


# I find it interesting how for females in class 1 and 3 there is a deviation from the general trend of "pay more survive more." For every other group of passenger the higher average fare was (loosley) connected to a better survival rate. In theory, this makes sense since a higher price most likely would have rooms further from the interior or potentially closer to life-rafts/other emergency vessels. I am not sure why there is such a staunch deviation from this trend when looking at the 1st and 3rd class women.

# In[ ]:




