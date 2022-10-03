# Neural_Network_Charity_Analysis
# Module19_Neural_Netwroks_and_Deep_Learning

## Overview of the analysis: Explain the purpose of this analysis.
Beks has come a long way since she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With the knowledge of machine learning and neural networks, use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

- EIN and NAME—Identification columns
- APPLICATION_TYPE—Alphabet Soup application type
- AFFILIATION—Affiliated sector of industry
- CLASSIFICATION—Government organization classification
- USE_CASE—Use case for funding
- ORGANIZATION—Organization type
- STATUS—Active status
- INCOME_AMT—Income classification
- SPECIAL_CONSIDERATIONS—Special consideration for application
- ASK_AMT—Funding amount requested
- IS_SUCCESSFUL—Was the money used effectively

Following are the steps involved in the analysis:
1. Preprocess the data for neural network
2. Compile, train and evaluate the model
3. Optimize the model

### Deliverable 1: Preprocessing Data for a Neural Network Model
The following preprocessing steps have been performed:
- The EIN and NAME columns have been dropped
- The columns with more than 10 unique values have been grouped together
- The categorical variables have been encoded using one-hot encoding
- The preprocessed data is split into features and target arrays
- The preprocessed data is split into training and testing datasets
- The numerical values have been standardized using the StandardScaler() module 

### Deliverable 2: Compile, Train, and Evaluate the Model
The neural network model using Tensorflow Keras contains working code that performs the following steps:
- The number of layers, the number of neurons per layer, and activation function are defined
- An output layer with an activation function is created
- There is an output for the structure of the model
- There is an output of the model’s loss and accuracy
- The model's weights are saved every 5 epochs
- The results are saved to an HDF5 file

### Deliverable 3: Optimize the Model
The model is optimized, and the predictive accuracy is increased to over 75%, or there is working code that makes three attempts to increase model performance using the following steps:
- Noisy variables are removed from features
- Additional neurons are added to hidden layers
- Additional hidden layers are added
- The activation function of hidden layers or output layers is changed for optimization
- The model's weights are saved every 5 epochs
- The results are saved to an HDF5 file

## Results: Using bulleted lists and images to support your answers, address the following questions.

### Data Preprocessing
What variable(s) are considered the target(s) for your model?
IS_SUCCESSFUL - considered target for the model

What variable(s) are considered to be the features for your model?
Each and every column - considered features for the model

What variable(s) are neither targets nor features, and should be removed from the input data?
Dropped columns - EIN and NAME as they had little or no impact to our outcome

### Compiling, Training, and Evaluating the Model
How many neurons, layers, and activation functions did you select for your neural network model, and why?
The model has 2 hidden layers for the neural network. The first layer has 80 neurons and the second layer has 30 neurons, with an output layer to the model. 

The first and the second hidden layers have the 'relu' activation function while the output layer has 'sigmoid.

![Fig1](https://github.com/veenapu/Neural_Network_Charity_Analysis/blob/main/Images/fig1.PNG)

Were you able to achieve the target model performance?
The model was not able to reach the 75% target and the accuracy of the model was 68%

![Fig2](https://github.com/veenapu/Neural_Network_Charity_Analysis/blob/main/Images/fig2.PNG)

What steps did you take to try and increase model performance?
Attempt 1: 
-  Additional feature 'USE_CASE' column was removed and rest of the features stayed the same but the accuracy dropped to 67%.

![Fig3](https://github.com/veenapu/Neural_Network_Charity_Analysis/blob/main/Images/fig3_attempt1.PNG)

Attempt 2: 
-  Additional neurons added to the hidden layers and the accuracy to 53%.

![Fig4](https://github.com/veenapu/Neural_Network_Charity_Analysis/blob/main/Images/fig4_attempt2.PNG)

Attempt 3: 
-  Additional neurons added to the hidden layers and accuracy went up to 59%.

![Fig5](https://github.com/veenapu/Neural_Network_Charity_Analysis/blob/main/Images/fig5_attempt3.PNG)

## Summary: 
### Summarize the overall results of the deep learning model. 
### Include a recommendation for how a different model could solve this classification problem, and explain your recommendation.

The model started with an accuracy score of 68% and ended up with an accuracy score of 59%. The loss in accuracy explains the overfitting of the model.

This can be optimized either by removing features or by adding more data to the existing dataset to increase accuracy.  

As indicated in our various attempts, the accuracy went down.  An option could have been to use the Random Forest Classifiers as these are robust and accurate due to their insufficient number of estimators and tree depth. These also have a faster performance and also could have avided the overfitting of the model.
