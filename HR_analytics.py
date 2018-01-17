
# coding: utf-8

# In[263]:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')
train_df = pd.read_csv("C:/Course/Certification/Dec17/HR_comma_sep.csv")


# In[264]:

train_df.shape


# In[265]:

sns.boxplot(x="left", y= "satisfaction_level", data=train_df)
plt.show()


# In[266]:

sns.boxplot(x="left", y= "last_evaluation", data=train_df)
plt.show()


# In[267]:

sns.boxplot(x="left", y= "average_montly_hours", data=train_df)
plt.show()


# In[268]:

sns.boxplot(x="left", y= "number_project", data=train_df)
plt.show()


# In[269]:

train_df.dtypes


# In[270]:

train_df["salary"] = train_df["salary"].apply(lambda salary: 0 if salary == 'low' else 1)
train_df


#     Split the data set into train and test

# In[271]:

y = train_df["left"]
#drop department & left
columns = ['Department', 'left']
train_df = train_df.drop(columns, axis=1)
col = train_df.columns
X = train_df[col]
X


# In[272]:

X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=42)


# In[273]:

my_d_tree = tree.DecisionTreeClassifier(max_depth = 10, min_samples_split = 5, random_state = 1)
my_d_tree = my_tree_one.fit(X_train, y_train)


# In[274]:

print(my_d_tree.feature_importances_) 
print(my_d_tree.score(X, y))


# In[275]:

pred = my_d_tree.predict(X_test)
pred


# In[276]:

pred = my_d_tree.predict(X_test)
df_confusion = metrics.confusion_matrix(y_test, pred)
df_confusion


# In[277]:

print(my_d_tree.score(X,y))


# In[278]:

def plot_confusion_matrix(df_confusion, title='Confusion matrix', cmap=plt.cm.gray_r):
    plt.matshow(df_confusion, cmap=cmap) # imshow
    plt.title('Confusion Matrix')
    plt.colorbar()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

plot_confusion_matrix(df_confusion)


# In[ ]:




# In[279]:

from sklearn.ensemble import RandomForestClassifier

# Building and fitting my_forest
forest = RandomForestClassifier(max_depth = 10, min_samples_split=2, n_estimators = 100, random_state = 1)
my_forest = forest.fit(X_train, y_train)

# Print the score of the fitted random forest
print(my_forest.score(X, y))
print(my_d_tree.feature_importances_) 


# In[280]:

pred = my_forest.predict(X_test)
df_confusion = metrics.confusion_matrix(y_test, pred)
df_confusion


# In[ ]:




# In[ ]:



