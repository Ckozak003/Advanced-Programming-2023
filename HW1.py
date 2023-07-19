#!/usr/bin/env python
# coding: utf-8

# ## Lecture 2 In Class Exercise

# # 1. Generating Random Data and Filtering That Data
# 

# In[3]:


import numpy as np
np.random.seed(25)
ar = np.random.randn(1000)


# **1.** Multiply all element in AR by 100, display the first 3 elements of ar

# In[4]:


ar = ar*100
ar[:3]


# **2.** Now convert that matrix into a matrix of 8-bit integers. Display first 3 elements

# In[5]:


ar = ar.astype(int)
ar[:3]


# *the first 3 values should be*
# 
# ```
#  22,  102,  -83,
# ```
# 

# **3.** Reshape the array into a set of five columns with 200 rows. Replace the original `ar` array with this new matrix.

# In[7]:


ar = ar.reshape(200,5)


# **4.** Now that `ar` is a matrix, let's get the maximum and minimum values from the matrix both row wise and column wise. Be sure to remember how many total values you should be outputting - one for every row, then one for each column.

# In[16]:


ar.max(axis=1), ar.min(axis=1) #rows


# In[17]:


ar.max(axis=0), ar.min(axis=0) #columns


# **5.** Now that we have gotten the minimum and maximum values for the columns, get the mean of the entire matrix. (This should be a specific value.)

# In[12]:


ar.mean()


# **6.** What is the total number of values that are less than the mean?

# In[35]:


ar[ar <3.561].size


# **7.** What is the total number of values in *each row* that are greater than this mean value? (Remember, if we are doing a row-oriented comparison, how many numbers should be in our output array?)

# In[57]:


count = np.count_nonzero(ar > 3.561, axis = 1)
count


# **8.** How many zeros are in the matrix that we have?

# In[36]:


ar[ar==0].size


# **9.** Lastly, sum all the values that are either greater than 100 or are less than 5

# In[46]:


x = ar[(ar > 100) | (ar < 5)]
sum(x)


# # 2. More NumPy Basics

# Using NumPy to do the following and display result for each step
# 1. Create a dataset of 50 random integers ranging from 0 and 100.
# 2. Reshape the data to be 10x5.
# 3. Create a dataset of 50 numbers between 0 and 100 using linspace()
# 4. Reshape the data to be 10x5.
# 5. Use vstack to stack the data for the two arrays call it "vertstk"
# 6. Use hstack to stack the data for the two arrays call it "hozstk"
# 7. for each array divide each element by 2:<br>
#     a. Use a functional approach (such as map or filter) for hozstk and a vectorized approach for vertstk<br>
#     b. for an extra challenge replace all values greater than 25 with 25
# 8. get column and row means for hozstk and vertstk

# In[64]:


#1
np.random.seed(25)
x = np.random.randint(0,100,size = 50)
x


# In[90]:


#2
x = x.reshape(10,5)
x


# In[91]:


#3
y = np.linspace(0,100,50)
y


# In[94]:


#4
y = y.reshape(10,5)
y


# In[95]:


#5
vertstck = np.vstack([x,y])
vertstck


# In[96]:


#6
hozstck = np.hstack([x,y])
hozstck


# In[103]:


#7
def div(val):
    return val/2
print(div(hozstck), '\n'*5)
print(vertstck/2)


# In[105]:


#8
vertstck.mean(axis = 0), vertstck.mean(axis = 1)


# 

# 

# In[107]:


#8 part 2
hozstck.mean(axis = 0), hozstck.mean(axis = 1)


# In[177]:


from statistics import mean
s = [0,0,0,0,0,0,0,0,0,0,0,0,0,0]
f = [30.2, 32.0, 31.1, 30.1, 30.2, 30.3, 30.6, 30.9, 30.5, 31.1, 31.3, 30.8, 30.3, 29.9, 29.8]
for i in range(0,len(f)-1):
    s[i] = (f[i+1] - f[i])/f[i]
m = 0
s
u = mean(s)
#for i in s:
#   m = m + (i-u)**2
for i in s:
    m = m + i**2
print(math.sqrt(m/15))


# In[ ]:





# In[ ]:




