#!/usr/bin/env python
# coding: utf-8

# In[196]:


import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[197]:


dftrain = pd.read_csv(r"Downloads\titanictrain.csv")
dftest = pd.read_csv(r'C:Downloads\titanictest.csv')
pd.set_option('display.max_columns',None)
dftrain.head()


# In[198]:


dftest.isnull().sum()


# In[199]:


dftrain.isnull().sum()


# In[200]:


df = pd.concat([dftrain,dftest])
df.shape


# In[201]:


sns.boxplot('Age',data = df)


# In[202]:


sns.boxplot('Pclass',data = df)


# In[203]:


sns.boxplot('SibSp',data = df)


# In[204]:


sns.boxplot('Fare',data = df)


# In[205]:


sns.boxplot('Parch',data = df)


# In[206]:


sns.boxplot('Fare',data = df)


# In[207]:


sns.boxplot('SibSp',data = df)


# In[208]:


df.Age.hist(bins = 20)


# In[209]:


df.Fare.hist(bins = 20)


# In[210]:


q1,q2 = np.percentile(df['SibSp'],[25,75])
print(q1,q2)
iqr = q2-q1
print(iqr)
lower = q1-(1.5*iqr)
upper = q2+(1.5*iqr)
print(lower)
print(upper)


# In[211]:


df.loc[df['SibSp']>=2.5,'SibSp']=2.5


# In[212]:


sns.boxplot('SibSp',data = df)


# In[213]:


q1,q2 = np.percentile(df['Parch'],[25,75])
print(q1,q2)
iqr = q2-q1
print(iqr)
lower = q1-(1.5*iqr)
upper = q2+(1.5*iqr)
print(lower)
print(upper)


# In[214]:


#dftrain.loc[dftrain['Parch']>=2.5,'Parch']=2.5


# In[215]:


sns.boxplot('Parch',data = df)


# In[216]:


#dftrain['Cabin1'] = np.where(dftrain['Cabin'].isnull(),1,0)
#dftrain.head()


# In[217]:


#dftrain['Age1'] = np.where(dftrain['Age'].isnull(),1,0)
#dftrain.head()


# In[218]:


#dftrain['Embarked1'] = np.where(dftrain['Embarked'].isnull(),1,0)
#dftrain.head()


# In[219]:


#dftrain.groupby('Survived')['Cabin1'].mean()


# In[220]:


#dftrain.groupby('Cabin')['Survived'].mean()


# In[221]:


#dftrain.groupby('Survived')['Age1'].mean()


# In[222]:


#dftrain.groupby('Survived')['Embarked1'].mean()


# In[223]:


#highcorr = dftrain.corr()
#highcorrfeature = highcorr.index[abs(highcorr['Survived'])<=.5]
#highcorrfeature


# In[224]:


#xtrain = dftrain.drop(columns = 'Survived')
#ytrain= dftrain['Survived']
#xtrain.shape


# In[225]:


#df = pd.concat([dftrain,dftest])
#df.shape


# In[226]:


df.corr()


# In[227]:


def core(dataset,threshold):
    col = set()
    cormatrix = dataset.corr()
    for i in range(len(cormatrix.columns)):
     for j in range(i):
        if abs(cormatrix.iloc[i,j])>threshold:
             colname = cormatrix.columns[i]
             col.add(colname)
    return col


# In[228]:


corrfeature = core(df,.1)
corrfeature


# In[229]:


#removearr = ['PassengerId', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
#df = df1.drop(columns = removearr)
#df


# In[230]:


miss = df.isnull().mean()
miss


# In[231]:


miss1 = dftest.isnull().mean()
miss1


# In[232]:


arr = miss[miss>21].keys()
arr


# In[233]:


catvar = df.select_dtypes(include = 'object')
catvar


# In[234]:


catvar.isnull().sum()


# In[235]:


df['Cabin'].fillna('missing',inplace= True)
df.head()


# In[236]:


df['Cabin']= df['Cabin'].astype(str).str[0]
df.head()


# In[237]:


ordi11= df.groupby('Cabin')['Survived'].mean().sort_values().index
ordi11


# In[238]:


eu = ['T', 'm', 'A', 'G', 'C', 'F', 'B', 'E', 'D']
from sklearn.preprocessing import OrdinalEncoder 
od = OrdinalEncoder(categories =[eu])
labels = od.fit_transform(df[['Cabin']])
df['Cabin'] = labels


# In[239]:


#dftest['Cabin'].fillna('missing',inplace= True)
#dftest.head(4)


# In[240]:


#dftest['Cabin']= dftest['Cabin'].astype(str).str[0]
#dftest.head()


# In[241]:


ordi = df.groupby('Sex')['Survived'].mean().sort_values()
ordi


# In[242]:


df['Sex'].unique()


# In[243]:


eu6= ['male','female']
from sklearn.preprocessing import OrdinalEncoder 
od = OrdinalEncoder(categories =[eu6])
labels = od.fit_transform(df[['Sex']])
df['Sex'] = labels


# In[244]:


df.head()


# In[245]:


df['Embarked'].value_counts


# In[246]:


mode = df.Embarked.mode()[0]
df['Embarked'].fillna(df['Embarked'].mode()[0],inplace = True)
df.isnull().sum()


# In[247]:



df.Embarked.hist(bins = 20)


# In[248]:


df['Embarked'].unique()


# In[249]:


eu3 = ['S', 'C', 'Q']
from sklearn.preprocessing import OrdinalEncoder 
od = OrdinalEncoder(categories =[eu3])
labels = od.fit_transform(df[['Embarked']])
df['Embarked'] = labels


# In[250]:


sns.boxplot('Embarked',data = df)


# In[251]:


labelvar = ['Name','Ticket']
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for var in labelvar:
    labels = le.fit_transform(df[var])
    df[var] = labels
df.head(5)


# In[252]:


numer = df.select_dtypes(include =['int64','float64'])
numer


# In[253]:


missnumer = [var for var in numer if numer[var] .isnull().sum()>0]
missnumer 


# In[254]:


#median = df1['Fare'].median()
#df1['Fare'] = df1['Fare'].fillna(median)


# In[255]:


extreme = df['Fare'].mean()+3*df['Fare'].std()
extreme


# In[256]:


df['Fare'] = df['Fare'].fillna(extreme)


# In[257]:


extreme = df['Age'].mean()+3*df['Age'].std()
extreme


# In[258]:


df['Age'] = df['Age'].fillna(extreme)


# In[259]:


#randsample = df1['Age'].dropna().sample(df1['Age'].isnull().sum(),random_state= 0)
#randsample


# In[260]:


#df1['Age1'] = df1['Age']
#randsample.index = df1[df1['Age'].isnull()].index
#df1.loc[df1['Age'].isnull(),'Age1']= randsample
#df1['Age1'].head(50)


# In[261]:


sns.boxplot('Fare',data = df)


# In[262]:


q1,q2 = np.percentile(df['Fare'],[25,75])
print(q1,q2)
iqr = q2-q1
print(iqr)
lower = q1-(1.5*iqr)
upper = q2+(1.5*iqr)
print(lower)
print(upper)


# In[263]:


df.loc[df['Fare']>=66,'Fare']=66
df.Fare.isnull().sum()


# In[264]:


sns.boxplot('Fare',data = df)


# In[265]:


df.head()


# In[266]:


#df1 = df1.drop(columns = ['Age'])


# In[267]:


trainlen = len(dftrain)
trainlen


# In[268]:


Train = df[:trainlen]
Train.shape


# In[269]:


Test = df[trainlen:]
Test.shape


# In[270]:


xtest = Test.drop(columns = ['Survived'])
ytest = Test['Survived']


# In[271]:


X = Train.drop(columns = ['Survived'])
Y = Train['Survived']

print(X.shape)
print(Y.shape)


# In[272]:


Train.head()


# In[273]:


Test.head()


# In[274]:


Y = Y.astype(int)


# In[275]:


xtest.isnull().sum()


# In[276]:


X.isnull().sum()


# In[278]:


from sklearn.linear_model import LogisticRegression
lg = LogisticRegression()
lg.fit(xt,yt)
ypred = lg.predict(xtes)
ypred


# In[279]:


from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(xt,yt)
ypred1=dt.predict(xtes)
ypred1


# In[280]:


from sklearn.ensemble import RandomForestClassifier
rancl = RandomForestClassifier( )
rancl.fit(xt,yt)
ypred2 = rancl.predict(xtest)
ypred2


# In[ ]:


from sklearn.model_selection import KFold
kfold_validation = KFold(10)


# In[ ]:


from sklearn. model_selection import cross_val_score
result = cross_val_score(ls,xsc,y,cv = kfold_validation)
print(result)
print(np.mean(result))


# In[281]:


from sklearn.metrics import accuracy_score
print(accuracy_score(ypred,ytes))
print(accuracy_score(ypred1,ytes))
print(accuracy_score(ypred2,ytes))


# In[282]:


Ycal = rancl.predict(xtest)
Ycal


# In[283]:


submittest = pd.concat([dftest['PassengerId'],pd.DataFrame(Ycal)],axis=1)
submittest.columns = ['PassengerId','Survived']
submittest.to_csv('gender_submission.csv',index = False)
pd.set_option('display.max_rows',None)
submittest


# In[ ]:




