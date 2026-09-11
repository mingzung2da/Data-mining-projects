#We build a set of detectors predicting students' on-task behavior in classrooms
#by examining whether specific instructional strategies are associated with the incidence of off-task behavior in elementary school children

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
import statsmodels.api as sm
from sklearn.metrics import roc_curve, auc
from pycaret.classification import *

# load data
dat = pd.read_csv('#5 Behavior Detection/data/assignment_4_training.csv')
print(dat.head())
print(dat.shape)

# convert all variables names to lower case
dat.columns = [x.lower() for x in dat.columns]
print(dat.columns)


# histgorams of the variables, stacked/grouped by the ontask value
for var in dat.columns:
    data = dat[var] #x-axis=the values of the variable (y-axis=the frequency/count of each value--how many times each value appears)
    plt.hist([data[dat.ontask == 'Y'], data[dat.ontask == 'N']], #create histogram with two separate groups (values where ontask is Y, values where ontask is N)
             stacked=True, edgecolor='k', alpha=0.7) #stack the Y and N bars on top of each other

    plt.title(var)
    plt.legend(['Y', 'N'])
    plt.show()
    print(len(data.unique()))

#From looking at the data shape, I can do data cleaning but nothing looks too bad rn. 
#I'll keep the data


#Data preprocessing
# remove variables that are not useful
dat = dat.drop(['uniqueid','studentid'], axis=1)

# convert ontask to 0/1
dat['ontask'] = np.where(dat['ontask'] == 'Y', 0, 1)

# convert categorical variables to dummy variables
categorical_variables = ['school','class','gender','coder','activity']
dat = pd.get_dummies(dat, columns=categorical_variables, drop_first=True)
print(dat.head())
print(dat.shape)


# Create a decision tree model using all variables to predict ontask
model = DecisionTreeClassifier()
model.fit(dat.drop('ontask', axis=1), pd.Series(dat['ontask'])) #train the model: dat.drop (features except ontask), pd.Series (what to predict)
print(model.feature_importances_)  #show how important each feature is (higher=more important)
print('\n')

# plot variable importance (why importance? It tells which variables matter most for predictions)
importances = model.feature_importances_ 
indices = np.argsort(importances)[::-1] #sort features by importance in descending order
plt.figure(figsize=(12,6))
plt.title("#1 Feature importances by Decision Tree Classifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.xticks(range(len(indices)), dat.drop('ontask', axis=1).columns[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show() 

# get the predicted values
yhat = model.predict(dat.drop('ontask', axis=1))
yhat = pd.Series(yhat.flatten())

# create confusion matrix
confusion_matrix = pd.crosstab(dat['ontask'], yhat, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print('\n')

#Result looks like
#Predicted      0     1
#Actual                
#0          14938     0
#1              0  7246

#14938 correct predictions for class 0
#7246 correct predictions for class 1
#0 wrong predictions overall
#current model is perfect (but suspiciously perfect. the model might have memorized detail of the training data, and won't perform well on new unseen data)

#After seeing the feature imporatance result, I'm going to drop "class" because there are so many which probably makes the model hard to learn
#and non of them look super important. Also it will be not be a useful variable to generalizing the model to data
#from class that we haven't yet seen. I'll drop "school" and "coder" as well, which doesn't seem important.

# drop all variables that start with 'class'
dat = dat.drop(dat.columns[dat.columns.str.startswith('class')], axis=1)

# drop all variables that start with 'school'
dat = dat.drop(dat.columns[dat.columns.str.startswith('school')], axis=1)

# drop coder_Z
dat = dat.drop('coder_Z', axis=1)


# create a decision tree model using all variables to predict ontask (do the same thing like we've done, after we dropped unimportant variables)
model = DecisionTreeClassifier()
model.fit(dat.drop('ontask', axis=1), pd.Series(dat['ontask']))
print(model.feature_importances_)
print('\n')

# plot variable importance
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.figure(figsize=(12,6))
plt.title("#2 Feature importances by Decision Tree Classifier")
plt.bar(range(len(indices)), importances[indices], color='lightblue',  align="center")
plt.xticks(range(len(indices)), dat.drop('ontask', axis=1).columns[indices], rotation='vertical',fontsize=14)
plt.xlim([-1, len(indices)])
plt.show()

# get the predicted values
yhat = model.predict(dat.drop('ontask', axis=1))
yhat = pd.Series(yhat.flatten())

# create confusion matrix
confusion_matrix = pd.crosstab(dat['ontask'], yhat, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print('\n')



# Create a logistic regression model using all variables to predict ontask
# Convert all columns except 'ontask' to numeric, coercing errors to NaN
dat_numeric = dat.drop('ontask', axis=1).apply(pd.to_numeric, errors='coerce')

# Convert boolean columns(True/False) to integers explicitly
for col in dat_numeric.columns:
    if dat_numeric[col].dtype == 'bool':
        dat_numeric[col] = dat_numeric[col].astype(int)

# Drop rows with NaN values that resulted from coercion (optional, depending on how you want to handle non-numeric data)
dat_numeric = dat_numeric.dropna()
dat_ontask = dat.loc[dat_numeric.index, 'ontask']

# Add a constant to the independent variables for the intercept (logistic regression needs an intercept term in the model)
dat_numeric = sm.add_constant(dat_numeric, prepend=False)


# creates and trains a logistic regression model. (see model coefficients and their significance (p-values))
logit_model = sm.Logit(dat_ontask, dat_numeric)
result = logit_model.fit()
print(result.summary2())

# get confusion matrix
yhat = result.predict(dat_numeric) #get prediction scores for each sample
# get optimal threshold using ROC curve (Receiver Operating Characteristic curve. Finds best threshold for classification)
fpr, tpr, thresholds = roc_curve(dat_ontask, yhat)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(optimal_threshold)
print('\n')

# get confusion matrix (evaluate model performance)
yhat = np.where(yhat > optimal_threshold, 1, 0)
yhat = pd.Series(yhat.flatten())
confusion_matrix = pd.crosstab(dat_ontask, yhat, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print('\n')

#Results show that Logistic Regresison is doing very poorly than Decision Tree. 
# My guess is that that Decision Tree is overfitting(like i wrote earlier, the model memorizes the training data instead of learning general patterns) the data. 
#so maybe logistic regression, which performed worse, generalize better to new data.
# Let's test the performance on the validation set


# load data
datval = pd.read_csv('#5 Behavior Detection/data/assignment_4_validation.csv')

# preprocess data (same process what we've done before for preprocessing)
datval.columns = [x.lower() for x in datval.columns]
datval = datval.drop(['uniqueid','studentid'], axis=1)
datval['ontask'] = np.where(datval['ontask'] == 'Y', 0, 1)
datval = pd.get_dummies(datval, columns=categorical_variables, drop_first=True)
datval = datval.drop(datval.columns[datval.columns.str.startswith('class')], axis=1)
datval = datval.drop(datval.columns[datval.columns.str.startswith('school')], axis=1)
datval = datval.drop('coder_Z', axis=1)

# get predictions from decision tree model
yhat = model.predict(datval.drop('ontask', axis=1))
yhat = pd.Series(yhat.flatten())

# get confusion matrix
confusion_matrix = pd.crosstab(datval['ontask'], yhat, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print('\n')

# get accuracy
accuracy = (confusion_matrix.iloc[0,0] + confusion_matrix.iloc[1,1]) / confusion_matrix.sum().sum()
print('accuracy')
print(accuracy)
print('\n')

# get balanced accuracy (for imbalanced data)
# If the data is imbalanced(e.g., 950 class 0, 50 class 1) accuracy predicts correct/total (simple average), but balanced accuracy predicts (class0_correct + class1_correct)/2. 
balanced_accuracy = (confusion_matrix.iloc[0,0] / confusion_matrix.iloc[0,:].sum() + confusion_matrix.iloc[1,1] / confusion_matrix.iloc[1,:].sum()) / 2
print('balanced accuracy')
print(balanced_accuracy)
print('\n')



# get predictions from logistic regression model
# Convert boolean columns to integers explicitly
for col in datval.columns:
    if datval[col].dtype == 'bool':
        datval[col] = datval[col].astype(int)

# Add a constant to the independent variables for the intercept
datval_numeric = sm.add_constant(datval.drop('ontask', axis=1), prepend=False)

yhat = result.predict(datval_numeric)
yhat = np.where(yhat > optimal_threshold, 1, 0)
yhat = pd.Series(yhat.flatten())

# get confusion matrix
confusion_matrix = pd.crosstab(datval['ontask'], yhat, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print('\n')

# get accuracy
accuracy = (confusion_matrix.iloc[0,0] + confusion_matrix.iloc[1,1]) / confusion_matrix.sum().sum()
print('accuracy')
print(accuracy)
print('\n')

# get balanced accuracy
balanced_accuracy = (confusion_matrix.iloc[0,0] / confusion_matrix.iloc[0,:].sum() + confusion_matrix.iloc[1,1] / confusion_matrix.iloc[1,:].sum()) / 2
print('balanced accuracy')
print(balanced_accuracy)
print('\n')


#Results show that both Decision Tree and Logistic Regression perform pretty bad on the validation set.
#So, we'll make our model more robust through model tuning.

# prepare data for automated model tuning
setup(dat,target='ontask') 

# compare all models automatically (logistic regression, decision tree, random forest etc.) and turns best one
mymodel = compare_models()

# train the best model with optimized hyperparameters
mymodel = create_model(mymodel)

# finalizes the model for deployment (trains on full dataset)
mymodel = finalize_model(mymodel)

# make preictions on validation data using the tuned model
predictions = predict_model(mymodel, datval)

# get confusion matrix (with the final chosen model)
confusion_matrix = pd.crosstab(predictions.ontask, predictions.prediction_label, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print('\n')

# get accuracy
accuracy = (confusion_matrix.iloc[0,0] + confusion_matrix.iloc[1,1]) / confusion_matrix.sum().sum()
print('accuracy')
print(accuracy)
print('\n')

# get balanced accuracy (for imbalanced data)
balanced_accuracy = (confusion_matrix.iloc[0,0] / confusion_matrix.iloc[0,:].sum() + confusion_matrix.iloc[1,1] / confusion_matrix.iloc[1,:].sum()) / 2
print('balanced accuracy')
print(balanced_accuracy)
print('\n')


# how imbalanced is ontask? (to detect class imbalance bias. What % of data is class 1 in training and validation)
#Actual distribution
print('percent of ontask=1 in data')
print(100*np.mean(dat.ontask)) # actual distribution (e.g., 42% class 1)
print(100*np.mean(datval.ontask)) # actual distribution (e.g., 45% class 1)
print('\n')

#Model's prediction
print('percent of ontask=1 in predictions') 
print(100*np.mean(predictions.prediction_label)) # Model's predictions (e.g., 20% class 1) -> then the model underestimates class 1
# so the two values should be similar idealy (e.g., actual distribution 40% class 1, predicted distribution 38% class 1)


#The pycaret package was extremely simple and found me a decent model pretty quickly. Overall, seems like a great package. 
#The final model was a boosted tree model(Gradient Boosting Classifier) and they were good in general.
#It had 68% accuracy on the validation set, however, it had 52% balanced accuracy.
#We can see that the ontask variable is imbalanced, so the model probably learn this pattern and starts predicting class 0(correct 93%, which is biased without class 1) almost all the time. 
# If the model only predicts class 0, the model gets high accuracy just by predicting the majority class, not because it's good.
#Future model building should address imbalanced data.


# Rerunning with fix_imbalance=true 
# Why? It automatically balances the classes before training.
# (common methods: Oversample(duplicate minority class, class 1, sample), Undersample(remove majority class, class 0, samples), Synthetic data(create fake class 1 samples))
# e.g. from (93% class 0, 7% class 1) to (50% class 0, 50% class 1). Model learns "Both class matters equally".
setup(dat,target='ontask',fix_imbalance=True)
mymodel = compare_models()
mymodel = create_model(mymodel)
mymodel = finalize_model(mymodel)
predictions = predict_model(mymodel, datval)

# get confusion matrix
confusion_matrix = pd.crosstab(predictions.ontask, predictions.prediction_label, rownames=['Actual'], colnames=['Predicted'])
print(confusion_matrix)
print('\n')

'''
# get accuracy
accuracy = (confusion_matrix.iloc[0,0] + confusion_matrix.iloc[1,1]) / confusion_matrix.sum().sum()
print('accuracy')
print(accuracy)
print('\n')

# get balanced accuracy
balanced_accuracy = (confusion_matrix.iloc[0,0] / confusion_matrix.iloc[0,:].sum() + confusion_matrix.iloc[1,1] / confusion_matrix.iloc[1,:].sum()) / 2
print('balanced accuracy')
print(balanced_accuracy)
print('\n')
'''

#check if both classes exist
if confusion_matrix.shape[1] == 2:
    accuracy = (confusion_matrix.iloc[0,0] + confusion_matrix.iloc[1,1]) / confusion_matrix.sum().sum()
    balanced_accuracy = (confusion_matrix.iloc[0,0] / confusion_matrix.iloc[0,:].sum() + confusion_matrix.iloc[1,1] / confusion_matrix.iloc[1,:].sum()) / 2
    print('accuracy')
    print(accuracy)
    print('\n')
    print('balanced accuracy')
    print(balanced_accuracy)
    print('\n')
else:
    print("Model only predicts one class. fix_imbalance is not working")
    print('\n')


print('percent of ontask=1 in predictions')
print(100*np.mean(predictions.prediction_label))

#Result shows that even with fix_imbalance=True, the model still only predicts class 0. 

