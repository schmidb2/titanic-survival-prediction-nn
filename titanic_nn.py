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

##UNCOMMENT BELOW CODE TO RUN FOR LOOP OF VARIOUS CONFIGURATIONS OF BATCH NUMBER
##AND EPOCHS
        
##for batch in [32,64,128,256]:
##
##    for epoch in [10,25,50,150,300]:
##
##        model = Sequential()
##        model.add(Dense(7,input_shape=(7,),activation='relu'))
##        model.add(Dense(4,activation='relu'))
##        model.add(Dense(1,activation='sigmoid'))
##
##        model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
##
##        model.fit(x,y,epochs=epoch,batch_size=batch)
##
##        _, accuracy = model.evaluate(x_test,y_test    )
##        results_table.append([batch,epoch,accuracy*100])
##print(tabulate(results_table,headers='firstrow',tablefmt='fancy_grid'))

###plotting data
##results_table = np.array(results_table)
##x_axis = [10,25,50,150,300]
####batch_10 = results_table[1:5,1]
##batch_32 = results_table[1:6,2].astype(float)
##batch_64 = results_table[6:11,2].astype(float)
##batch_128 = results_table[11:16,2].astype(float) 
##batch_256 = results_table[16:21,2].astype(float) 
##
##line1, = plt.plot(x_axis,batch_32,color='blue')
##line1.set_label('Batch = 32')
##line2, = plt.plot(x_axis,batch_64,color='pink')
##line2.set_label('Batch = 64')
##line3, = plt. plot(x_axis,batch_128,color='green')
##line3.set_label('Batch = 128')
##line4, = plt.plot(x_axis,batch_256,color='orange')
##line4.set_label('Batch = 256')
##
##plt.xlabel('Number of Epochs')
##plt.ylabel('Accuracy')
##plt.legend()
##plt.show()
