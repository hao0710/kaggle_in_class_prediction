import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.linear_model import LogisticRegressionCV
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
#0.7891201742708459





from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import re
from sklearn.impute import SimpleImputer
from sklearn import preprocessing


def help_func(i):
    try:
        a,b=i.split("?")


        try:

            a=str(w2n.word_to_num(a))

        except:

            pass
        try:

            b=str(w2n.word_to_num(b))
        except:

            pass
        a=a.replace(' ',"")
        b=b.replace(' ',"")
        if b.isdigit():
            return a+" "+b
        else:
            return b+" "+a
    except:
        return ""

def age_help_func(l):
    if len(l)>0:
        return int(l[0])
    else:
        return np.nan
def sex_help_func(l):
    if len(l)>0:
        if l[0]=="F":
            return 0
        else:
            return 1
    else:
        return np.nan
def stroke_help_func(i):
    if i=="0":
        return 0
    elif i=="1":
        return 1
    else:
        return np.nan

def smok_help_func(i):
    c=['non-smoker', 'quit', 'active_smoker', np.nan]
    if i not in c:
        return np.nan
    else:
        if i=="non-smoker":
            return 0
        elif i=="quit":
            return 1
        elif i=="active_smoker":
            return 2
        else:
            return np.nan
def zero_1_clean(i):
    c=["0","1",1,0,np.nan]
    if i not in c:
        return np.nan
    else:
        return i
def area_help_func(l):
    if len(l)>0:
        if l[0]=="city":
            return 0
        else:
            return 1
    else:
        return np.nan
def job_help_func(l):
    Jobs={"unemployed":0,"private":1,"government":2,"business":3,"parental":4}
    if len(l)>0:
        return Jobs[l[0]]
    else:
        return 0



file_name=["train.csv","test.csv","sample.csv"]
#
train_x = pd.read_csv(file_name[0])
test_x = pd.read_csv(file_name[1])

testId=test_x["id"]
DataSet =  pd.concat(objs=[train_x, test_x], axis=0,sort=False).reset_index(drop=True)

train_len=train_x.shape[0]
test_len=test_x.shape[0]
# test_x.shape
#
# train_x["sex and age"].unique()
#
DataSet["sex and age"]=DataSet["sex and age"].fillna("")
DataSet["sex and age"]=DataSet["sex and age"].str.replace("Male","M")
DataSet["sex and age"]=DataSet["sex and age"].str.replace("MALE","M")
DataSet["sex and age"]=DataSet["sex and age"].str.replace("male","M")
DataSet["sex and age"]=DataSet["sex and age"].str.replace("female","F")
DataSet["sex and age"]=DataSet["sex and age"].str.replace("Female","F")

DataSet["sex and age"][0].split(",")
DataSet["sex and age"]=DataSet["sex and age"].fillna("")
DataSet["sex"]=[sex_help_func(re.findall('F|f|M|m',i)) for i in DataSet["sex and age"]]
DataSet["age"]=[age_help_func(re.findall('\d+',i)) if i else np.nan for i in DataSet["sex and age"]]

DataSet

DataSet["job_status and living_area"]=DataSet["job_status and living_area"].fillna("")
DataSet["job_status and living_area"]=DataSet["job_status and living_area"].str.lower()

DataSet["living_area"]=[area_help_func(re.findall('remote|city',i)) for i in DataSet["job_status and living_area"]]
DataSet["job"]=[job_help_func(re.findall('private|government|business|unemployed|parental',i))  for i in DataSet["job_status and living_area"]]

DataSet=DataSet.drop(["job_status and living_area","sex and age",'id'], axis=1)
DataSet=DataSet.drop(["TreatmentA","TreatmentB","TreatmentC","TreatmentD"], axis=1)


DataSet["stroke_in_2018"]=[stroke_help_func(i) for i in DataSet["stroke_in_2018"]]

# DataSet["age"].dropna().astype(int)
# DataSet["stroke_in_2018"].isna()[DataSet["stroke_in_2018"].isna()==True]
DataSet["high_BP"]=DataSet["high_BP"].replace('.,',np.nan).astype(float)
DataSet
DataSet["heart_condition_detected_2017"]=[zero_1_clean(i) for i in DataSet["heart_condition_detected_2017"]]
DataSet["heart_condition_detected_2017"]=DataSet["heart_condition_detected_2017"].astype(float)
DataSet["married"]=[zero_1_clean(i) for i in DataSet["married"]]
DataSet["married"]=DataSet["married"].astype(float)


DataSet["BMI"]=DataSet["BMI"].replace("?",np.nan)
DataSet["BMI"]=DataSet["BMI"].replace(".",np.nan)
DataSet["BMI"]=DataSet["BMI"].astype(float)

DataSet["smoker_status"]=[smok_help_func(i) for i in DataSet["smoker_status"]]


DataSet["smoker_status"].isna().sum()




for co in DataSet.columns:
    DataSet[co].fillna(DataSet[co].dropna().median(), inplace=True)















DataSet
# print(df.shape)
# print(data.columns)
data1=DataSet.iloc[:train_len,:]
# print(data1.shape)
# print(data1['stroke_in_2018'].value_counts())


t1=data1.loc[data1['stroke_in_2018']==1.0]#选择小数据
print(t1.shape)

t1=t1.append([t1]*50,ignore_index=True)
# print(t1.shape)
# x2=df[-8718:].drop(['stroke_in_2018'],axis=1)

data_train=pd.concat([data1,t1])
# x2.to_csv('mytest.csv')
print(data_train['stroke_in_2018'].value_counts())


# print(x1)


# x2=data[-8718:,:]
# y2=data[-8718:,:]

x=data_train.drop(["stroke_in_2018"], axis=1)
y=data_train['stroke_in_2018']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=5)
clf=DecisionTreeClassifier(random_state=0, max_depth=5).fit(x_train,y_train)
y_pred=clf.predict(x_test)
print(accuracy_score(y_test,y_pred))

# df_test=pd.read_csv('mytest.csv',index_col=0)
df_test=DataSet.iloc[train_len:,:]
df_test=df_test.drop(["stroke_in_2018"], axis=1)


test_Survived = pd.Series(clf.predict(df_test), name="stroke_in_2018")
test_Survived.value_counts()
results = pd.concat([testId,test_Survived],axis=1)
results
results["stroke_in_2018"]=results["stroke_in_2018"].astype('int')
results.to_csv("./voted_dt.csv",index=False)
