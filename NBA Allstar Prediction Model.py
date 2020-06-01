#!/usr/bin/env python
# coding: utf-8

# In[117]:


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


df = pd.read_csv('NBA.csv')
df = df.drop(columns = ['Unnamed: 0','Age','Tm','G','GS','Pos', "Rk"])
df = df.dropna()

feature_cols = ['MP','FG','FGA','FG%','3P','3PA','3P%','2P','2PA','2P%','eFG%','FT','FTA','FT%','ORB','DRB','TRB','AST','STL','BLK','TOV','PF','PTS']
label = ['Allstar']
X = df[feature_cols]
y = df.Allstar
df.head()


# In[118]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.30,random_state=0)

logReg = LogisticRegression()
logReg.fit(X_train,y_train)
y_pred=logReg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix


# In[119]:


print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))


# In[130]:


y_pred_prob = logReg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob)
auc = metrics.roc_auc_score(y_test, y_pred_prob)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr,tpr,label="AUC="+str(auc))
plt.legend(loc=4)
plt.show()


# In[ ]:




