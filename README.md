# titanic-survival-prediction-nn
Using passenger list from Titanic, creates model to predict survival using neural networks

## The Dataset

#### The passenger list from the Titanic was downloaded from the link below. This dataset contains the information of 1309 passengers including their glass, age, gender, and whether they survived. 

https://raw.githubusercontent.com/tpradeep8/tableau-data-visualization/master/titanic%20passenger%20list.csv

## Training

#### The following blog was used for reference when creating the model: 
https://machinelearningmastery.com/tutorial-first-neural-network-python-keras/

#### Keras was used to train the model. A Sequential model was created, then layers were added to the model. 
#### The first layer, the input layer, contained 7 neurons as there are 7 features in the data. The output layer contained 1 neuron as there is only 1 output values from the model. Between the input and output layers there is a hidden layer that has 4 neurons. Other resources suggested that the number of neurons in a hidden layer should be the mean of the number of neurons in the input and output layers. 
#### The model was compiled usingthe Adam Optimizer.
#### When the model was fit to the data, a batch size of 10 and epochs of 50 was chosen. 
## Results
#### Using the above strategy, the accuracy of the model was 78.63%.