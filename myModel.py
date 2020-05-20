#importing pandas and numpy

import pandas as pd
import numpy as np

#read data

data=pd.read_csv('./cpdata.csv')
# print(data.head(1))

#preprocessing data

from sklearn import preprocessing
#labelEncoder=preprocessing.LabelEncoder()
# data['label']=labelEncoder.fit_transform(data['label'])

target=data['label']

# print(data.iloc[:,4].value_counts())
# print(target)

# Scaled feature 

# min_max_scaler = preprocessing.MinMaxScaler(feature_range =(0, 1)) 
# data = min_max_scaler.fit_transform(data.iloc[:,0:4]) 

standardisation=preprocessing.StandardScaler()
standardisation.fit(data.iloc[:,0:4])
# print(standardisation)
data=standardisation.transform(data.iloc[:,0:4])
# print("mean is ",standardisation.mean_)

# print("scale is ",standardisation.scale_)
# print(data)
#creating dataframe from the array returned by min_max_scaler

data=pd.DataFrame(data,columns=['temperature','humidity','ph','rainfall'])
x=data.iloc[:,:].values
y=target[:].values
 
#splitting data into train and test data

from sklearn.model_selection import train_test_split
#from sklearn import utils
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,shuffle=True)

#importing neural_network from sklearn

from sklearn.neural_network import MLPClassifier
m_clf=MLPClassifier(solver='lbfgs')
m_clf.fit(x_train,y_train)

# print(x_test)
y_predict=m_clf.predict(x_test)
# print(y_predict)

from sklearn.metrics import accuracy_score
# Finding the accuracy of the model
a=accuracy_score(y_test,y_predict)
# print("The accuracy of this model is: ", a*100)

#importing pickle
import pickle
pickle.dump(m_clf,open('models/final_prediction.pickle','wb'))

