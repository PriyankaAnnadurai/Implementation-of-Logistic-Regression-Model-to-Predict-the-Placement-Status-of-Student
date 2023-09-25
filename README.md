# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

  1. Import the standard libraries.

  2.Upload the dataset and check for any null or duplicated values using .isnull() and   .duplicated() function respectively.

  3.Import LabelEncoder and encode the dataset.

  4.Import LogisticRegression from sklearn and apply the model on the dataset.

  5.Predict the values of array.

  6.Calculate the accuracy, confusion and classification report by importing the required modules from sklearn.

  7.Apply new unknown values

## Program:
```
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
# Developed by: PRIYANKA.A
# RegisterNumber:  212222230113
# Import Library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read The File
dataset=pd.read_csv('Placement_Data_Full_Class.csv')
dataset
dataset.head(10)
dataset.tail(10)
# Dropping the serial number and salary column
dataset=dataset.drop(['sl_no','ssc_p','workex','ssc_b'],axis=1)
dataset
dataset.shape
dataset.info()
dataset["gender"]=dataset["gender"].astype('category')
dataset["hsc_b"]=dataset["hsc_b"].astype('category')
dataset["hsc_s"]=dataset["hsc_s"].astype('category')
dataset["degree_t"]=dataset["degree_t"].astype('category')
dataset["specialisation"]=dataset["specialisation"].astype('category')
dataset["status"]=dataset["status"].astype('category')
dataset.info()
dataset["gender"]=dataset["gender"].cat.codes
dataset["hsc_b"]=dataset["hsc_b"].cat.codes
dataset["hsc_s"]=dataset["hsc_s"].cat.codes
dataset["degree_t"]=dataset["degree_t"].cat.codes
dataset["specialisation"]=dataset["specialisation"].cat.codes
dataset["status"]=dataset["status"].cat.codes
dataset.info()
dataset
# selecting the features and labels
x=dataset.iloc[:, :-1].values
y=dataset.iloc[: ,-1].values
y
# dividing the data into train and test
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
dataset.head()
y_train.shape
x_train.shape
# Creating a Classifier using Sklearn
from sklearn.linear_model import LogisticRegression
clf=LogisticRegression(random_state=0,solver='lbfgs',max_iter=1000).fit(x_train,y_train)
# Printing the acc
clf=LogisticRegression()
clf.fit(x_train,y_train)
clf.score(x_test,y_test)
# Predicting for random value
clf.predict([[1	,78.33,	1,	2,	77.48,	2,	86.5,	0,	66.28]])
```

## Output:

### Read CSV File:

![270174007-06f03811-8b27-46a9-9ee1-e92a82526e4e](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/cf017214-1d26-4287-a6f4-8129ca2b62cf)



### To read 1st ten Data(Head):

![270174020-15a2d9d0-9f53-4136-9160-c97ce7d8698d](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/6dd1deb3-c209-4dca-b59f-9934e1deb862)



### To read last ten Data(Tail):

![270174033-085ec7ef-63fd-40f1-8113-d945026096e5](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/9fa733cf-c75c-4e0f-babf-d9530a5e898f)



### Dropping the serial number and salary column:

![270174057-209fe561-064f-48bd-90e4-d0cfdfb29aef](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/2ffe4201-f449-4cfd-b0ad-34890c9cd551)


### Dataset Information:

![270174079-580bcfbd-afc0-4bb1-a44d-d4623687e515](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/23521bcf-eb9a-421e-bd8b-7ba98ae40b79)



### Dataset after changing object into category:

![270174093-4185c6fe-016b-4630-a896-561257b97234](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/34008710-b581-46e8-bb73-de4acafd3944)




### Dataset after changing category into integer:

![270174117-eef76c9d-70a2-4dad-838f-8cfd5c3e6b8d](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/b943392b-ff4d-4b14-8244-d6a534265eaa)




### Displaying the Dataset:

![270174130-52d99f9d-85c0-4cce-91b2-3ac0f7dc8e81](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/b3e1575b-8bbf-45be-ba27-45a133dc6168)




### Selecting the features and labels:

![270174158-5961a94b-f854-4486-9649-5f7a46d93529](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/5bcd2b83-efa9-40ff-80ef-e685994bd9e4)



### Dividing the data into train and test:

![270174180-b2c41d22-80eb-4bea-9f01-f35bcd05aad1](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/7e4c21e1-f60a-4363-ac05-a615514dfe14)



### Creating a Classifier using Sklearn:

![270174198-59ef639a-219d-45d4-8c97-dae07b9c5b80](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/f0eb9d68-3066-47ad-a398-91ad76e68e0b)



### Predicting for random value:

![270174217-8411a5e1-3606-48c8-bc3c-2bbbc08f6cc6](https://github.com/PriyankaAnnadurai/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/118351569/45e49f8c-90ab-428c-9109-598f6141699e)



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
