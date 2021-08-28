#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


data= pd.read_csv("Mall_Customers.csv")


# In[3]:


data.head()


# In[4]:


data.rename(columns={'Genre':'Gender'},inplace=True)


# In[5]:


data.head()


# In[6]:


data.describe()


# In[7]:


data.dtypes


# In[8]:


data.isna().sum()


# In[9]:


data.drop(["CustomerID"],axis=1,inplace=True)


# In[10]:


data.head()


# In[11]:


plt.figure(1,figsize=(15,6))
n=0
for x in['Age','Annual Income (k$)','Spending Score (1-100)' ]:
    n=n+1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    sns.distplot(data[x],bins=20)
    plt.title('Distplot of {}'.format(x))
    
plt.show()


# In[12]:


plt.figure(figsize=(15,5))
sns.countplot(y='Gender',data=data)
plt.show()


# In[13]:


plt.figure(1,figsize=(15,7))
n=0
for cols in ['Age','Annual Income (k$)','Spending Score (1-100)' ]: 
    n=n+1
    plt.subplot(1,3,n)
    plt.subplots_adjust(hspace=0.5,wspace=0.5)
    sns.violinplot(x=cols, y= 'Gender', data=data)
    plt.ylabel('Gender' if n==1 else '')
    plt.title('violin plot')
plt.show()


# In[14]:


age_18_25 = data.Age[(data.Age>=18) & (data.Age <= 25)]
age_26_35 = data.Age[(data.Age>=26) & (data.Age <= 35)]
age_36_45 = data.Age[(data.Age>=36) & (data.Age <= 45)]
age_46_55 = data.Age[(data.Age>=46) & (data.Age <= 55)]
age_55above = data.Age[data.Age >=56]

agex=["18-25","26-35","36-45","46-55","55+"]
agey=[len(age_18_25.values),len(age_26_35.values),len(age_36_45.values),len(age_46_55.values),len(age_55above.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=agex,y=agey,palette="mako")
plt.title("Age wise distribution of the customers")
plt.xlabel("Age")
plt.ylabel("Number of customers")
plt.show()


# In[16]:


sns.relplot(x="Annual Income (k$)",y="Spending Score (1-100)",data=data)


# In[17]:


ss_1_20=data["Spending Score (1-100)"][(data["Spending Score (1-100)"]>=1)& (data["Spending Score (1-100)"]<=20)]
ss_21_40=data["Spending Score (1-100)"][(data["Spending Score (1-100)"]>=21)& (data["Spending Score (1-100)"]<=40)]
ss_41_60=data["Spending Score (1-100)"][(data["Spending Score (1-100)"]>=41)& (data["Spending Score (1-100)"]<=60)]
ss_61_80=data["Spending Score (1-100)"][(data["Spending Score (1-100)"]>=61)& (data["Spending Score (1-100)"]<=80)]
ss_81_100=data["Spending Score (1-100)"][(data["Spending Score (1-100)"]>=81)& (data["Spending Score (1-100)"]<=100)]


# In[18]:


ssx =["1-20","21-40","41-60","61-80","81-100"]
ssy= [len(ss_1_20.values),len(ss_21_40.values),len(ss_41_60.values),len(ss_61_80.values),len(ss_81_100.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=ssx, y=ssy, palette="rocket")
plt.title("Spending score")
plt.xlabel("Score")
plt.ylabel("Number of customers having the score")
plt.show()


# In[19]:


ai0_30=data["Annual Income (k$)"][(data["Annual Income (k$)"]>=0)&(data["Annual Income (k$)"]<=30)]
ai31_60=data["Annual Income (k$)"][(data["Annual Income (k$)"]>=31)&(data["Annual Income (k$)"]<=60)]
ai61_90=data["Annual Income (k$)"][(data["Annual Income (k$)"]>=61)&(data["Annual Income (k$)"]<=90)]
ai91_120=data["Annual Income (k$)"][(data["Annual Income (k$)"]>=91)&(data["Annual Income (k$)"]<=120)]
ai121_150=data["Annual Income (k$)"][(data["Annual Income (k$)"]>=121)&(data["Annual Income (k$)"]<=150)]

aix=["$ 0 - 30,000","$ 30,001 - 60,000","$ 60,001 - 90,000","$ 91,001 - 120,000","$ 120,001 - 150,000"]
aiy= [len(ai0_30.values),len(ai31_60.values),len(ai61_90.values),len(ai91_120.values),len(ai121_150.values)]

plt.figure(figsize=(15,6))
sns.barplot(x=aix, y=aiy, palette="Spectral")
plt.title("Annual Incomes")
plt.xlabel("Income")
plt.ylabel("Number of customer")
plt.show()


# In[27]:


X1=data.loc[:,["Age","Spending Score (1-100)"]].values

from sklearn.cluster import KMeans
wcss = []


# In[28]:




for k in range(1,11):
    kmeans=KMeans(n_clusters=k, init = "k-means++")
    kmeans.fit(X1)
    wcss.append(kmeans.inertia_)    
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2, color="red", marker="8")
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.show()
    


# In[30]:


kmeans= KMeans(n_clusters=4)
label= kmeans.fit_predict(X1)
print(label)


# In[31]:


print(kmeans.cluster_centers_)


# In[34]:


plt.scatter(X1[:,0],X1[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')
plt.title('clusters of customers ')
plt.xlabel("Age")
plt.ylabel("Spending score(1-100)")
plt.show()


# In[35]:


X2= data.loc[:,['Annual Income (k$)','Spending Score (1-100)']].values

from sklearn.cluster import KMeans
wcss=[]
for k in range(1,11):
    kmeans= KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X2)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2, color="red", marker="8")
plt.xlabel("K value")
plt.ylabel("wcss")
plt.show()


# In[36]:


kmeans= KMeans(n_clusters=5)

label=kmeans.fit_predict(X2)
print(label)


# In[37]:


print(kmeans.cluster_centers_)


# In[38]:


plt.scatter(X2[:,0],X1[:,1],c=kmeans.labels_,cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],color='black')
plt.title('clusters pf computers')
plt.xlabel('Annual Income(k$)')
plt.ylabel('spending score(1-100)')
plt.show()


# In[39]:


X3=data.iloc[:,1:]

wcss=[]
for k in range(1,11):
    kmeans= KMeans(n_clusters=k, init="k-means++")
    kmeans.fit(X3)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(12,6))
plt.grid()
plt.plot(range(1,11),wcss,linewidth=2, color="red", marker="8")
plt.xlabel("K value")
plt.ylabel("WCSS")
plt.show()


# In[40]:


kmeans= KMeans(n_clusters=5)

label=kmeans.fit_predict(X3)
print(label)


# In[41]:


print(kmeans.cluster_centers_)


# In[44]:


clusters= kmeans.fit_predict(X3)
data["label"]=clusters
from mpl_toolkits.mplot3d import Axes3D

fig= plt.figure(figsize=(20,10))
ax= fig.add_subplot(111, projection='3d')
ax.scatter(data.Age[data.label == 0], data["Annual Income (k$)"][data.label == 0 ], data["Spending Score (1-100)"][data.label == 0], c='blue',s=60)
ax.scatter(data.Age[data.label == 1], data["Annual Income (k$)"][data.label == 1 ], data["Spending Score (1-100)"][data.label == 1], c='red',s=60)
ax.scatter(data.Age[data.label == 2], data["Annual Income (k$)"][data.label == 2 ], data["Spending Score (1-100)"][data.label == 2], c='green',s=60)
ax.scatter(data.Age[data.label == 3], data["Annual Income (k$)"][data.label == 3 ], data["Spending Score (1-100)"][data.label == 3], c='orange',s=60)
ax.scatter(data.Age[data.label == 4], data["Annual Income (k$)"][data.label == 4 ], data["Spending Score (1-100)"][data.label == 4], c='purple',s=60)
ax.view_init(30,185)

plt.xlabel("Age")
plt.ylabel("Annual Income(k$)")
ax.set_zlabel("Spending Score (1-100)")

plt.show()

