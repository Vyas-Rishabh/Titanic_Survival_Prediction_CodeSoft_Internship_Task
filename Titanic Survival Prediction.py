#!/usr/bin/env python
# coding: utf-8

# # Titanic Survival Prediction Using Logistic Regression

# In[1]:


#import Libraries


# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# ### Load the Data 

# In[3]:


titanic_data = pd.read_csv("Titanic Dataset.csv")


# In[4]:


titanic_data


# In[5]:


# Length of Data
len(titanic_data)


# ### View the data using head function which return top 5 rows

# In[6]:


titanic_data.head()


# In[7]:


titanic_data.index


# In[8]:


titanic_data.columns


# ### summary of the DataFrame

# In[9]:


titanic_data.info()


# In[10]:


titanic_data.dtypes


# In[11]:


titanic_data.describe()


# ### Explaining Datasets

# survival : Survival 0 = No, 1 = Yes <br>
# pclass : Ticket class 1 = 1st, 2 = 2nd, 3 = 3rd <br>
# sex : Sex <br>
# Age : Age in years <br>
# sibsp : Number of siblings / spouses aboard the Titanic 
# <br>parch # of parents / children aboard the Titanic <br>
# ticket : Ticket number fare Passenger fare cabin Cabin number <br>
# embarked : Port of Embarkation C = Cherbourg, Q = Queenstown, S = Southampton <br>

# # Data Analysis

# ### import Seaborn for visually analysing the data

# ### Find out how many survived vs Died using countplot method of seaborn

# In[12]:


#countplot of survived vs not survived


# In[13]:


sns.countplot(x='Survived', data=titanic_data)


# ### Male vs Female Survival

# In[14]:


#Male vs Female Survival


# In[15]:


sns.countplot(x='Survived', data=titanic_data, hue='Sex')


# Only Females are Survived and Male are not Survived as per the countplot

# **See age group of passengeres travelled **<br>
# Note: We will use displot method to see the histogram. However some records does not have age hence the method will throw an error. In order to avoid that we will use dropna method to eliminate null values from graph

# ### Check for null

# In[16]:


titanic_data.isna()


# ### Check how many values are null

# In[17]:


titanic_data.isna().sum()


# ### Visualize null values help of Heatmap

# In[18]:


sns.heatmap(titanic_data.isna())


# ### find the % of null values in age column

# In[19]:


(titanic_data['Age'].isna().sum()/len(titanic_data['Age']))*100


# ### find the % of null values in cabin column

# In[20]:


(titanic_data['Cabin'].isna().sum()/len(titanic_data['Cabin']))*100


# ### find the distribution for the age column

# In[21]:


sns.displot(x='Age', data=titanic_data)


# # Data Cleaning

# **Fill the missing values**</BR>
# we will find the missing values for age. In order to fill missing values we use fillna method.</BR>
# For now we will fill the missing age by taking average of all age.

# ### fill age column

# In[22]:


titanic_data['Age'].fillna(titanic_data['Age'].mean(), inplace=True)


# ### verify null values

# In[23]:


titanic_data['Age'].isna().sum()


# **Alternatively we will visualise the null value using heatmap.**</BR>
# we will use heatmap method by passing only records which are null.

# In[24]:


#visualize null values again


# In[25]:


sns.heatmap(titanic_data.isna())


# **we can see the cabin column has a number of null values, as such we can not use it for prediction. Hence we will drop it.**

# In[26]:


#drop cabin column


# In[27]:


titanic_data.drop('Cabin', axis=1, inplace=True)


# In[28]:


#see the content of data


# In[29]:


titanic_data.head()


# **Preaparing Data for Model**</BR>
# No we will require to convert all non-numerical columns to numeric. Please note this is required for feeding data into model. Lets see which columns are non numeric info describe method

# In[30]:


#Check for the non-numeric column


# In[31]:


titanic_data.info()


# In[32]:


titanic_data.dtypes


# **We can see, Name, Sex, Ticket and Embarked are non-numerical.It seems Name,Embarked and Ticket number are not useful for Machine Learning Prediction hence we will eventually drop it. For Now we would convert Sex Column to dummies numerical values****

# In[33]:


#convert sex column to numerical values


# In[34]:


gender=pd.get_dummies(titanic_data['Sex'],drop_first=True)


# In[35]:


titanic_data['Gender']=gender


# In[36]:


titanic_data.head()
#Here, in Gender male=1 & female=0


# In[37]:


#drop the column which are not require


# In[38]:


titanic_data.drop(['Name','Sex','Ticket','Embarked'], axis=1, inplace=True)


# In[39]:


titanic_data.head()


# In[40]:


titanic_data.isna().sum()


# In[43]:


# Fill null value with average in Fare Column
titanic_data['Fare'].fillna(titanic_data['Fare'].mean(), inplace=True)


# In[44]:


#Seperate Dependent and Independent variables


# In[45]:


x=titanic_data[['PassengerId','Pclass','Age','SibSp','Parch','Fare','Gender']]
y=titanic_data['Survived']


# In[46]:


x


# In[47]:


y


# # Data Modeling

# **Building Model using Logistic Regression**</BR>
# </BR>
# **Build the model**

# In[48]:


#import train test split method


# In[49]:


from sklearn.model_selection import train_test_split


# In[50]:


#train test split


# In[51]:


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)


# In[52]:


#import logistic Regression


# In[53]:


from sklearn.linear_model import LogisticRegression


# In[54]:


#fit Logistic Regression


# In[55]:


lr = LogisticRegression()


# In[57]:


lr.fit(X_train,y_train)

# Ignoring errors
import warnings
warnings.filterwarnings('ignore')


# In[58]:


#predict


# In[59]:


predict=lr.predict(X_test)


# # Testing

# **See how our model is performing**

# In[60]:


#print confusion matrix


# In[61]:


from sklearn.metrics import confusion_matrix


# In[62]:


pd.DataFrame(confusion_matrix(y_test,predict),columns=['Predicted No','Predicted Yes'], 
             index=['Actual No','Actual Yes'])


# In[63]:


#import classification report


# In[64]:


from sklearn.metrics import classification_report


# In[65]:


print(classification_report(y_test, predict))


# **Precision is fine considering Model Selected and Available Data. Accuracy can be increased by further using more features (which we dropped earlier) and/or by using other model**
# 
# Note:</BR>
# Precision : Precision is the ratio of correctly predicted positive observations to the total predicted positive observations
# Recall : Recall is the ratio of correctly predicted positive observations to the all observations in actual class F1 score - F1 Score is the weighted average of Precision and Recall.
