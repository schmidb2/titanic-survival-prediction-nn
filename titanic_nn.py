import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tabulate import tabulate

#read in data
data = pd.read_csv('passenger list.csv')

#drop irrelevant data
data= data.drop(columns=['home.dest','body','boat','cabin','name','ticket'],axis=1)

#replace features that are strings with equivalents int
data.replace({'sex':{'male':0,'female':1},'embarked':{'S':0,'C':1,'Q':2}}, inplace=True)

#replace nan values with mean
data['age'].fillna(data['age'].mean(),inplace=True)
data['fare'].fillna(data['fare'].mean(),inplace=True)
data['embarked'].fillna(data['embarked'].mean(),inplace=True)

#remove label from data into separate variable 
label = data['survived']
train = data.drop(columns = ['survived'])

#split data into train and test (20% holdout)
x_train,x_test,y_train,y_test = train_test_split(train,label, test_size=0.2,random_state=2)

x = np.array(x_train)
y = np.array(y_train)


results_table = [['Neurons in Hidden Layer','Number of Epochs','Accuracy Score']]

model = Sequential()
model.add(Dense(7,input_shape=(7,),activation='relu'))
model.add(Dense(4,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

model.fit(x,y,epochs=50,batch_size=10)

_, accuracy = model.evaluate(x_test,y_test)
print('accuracy: %2f' %(accuracy*100))
