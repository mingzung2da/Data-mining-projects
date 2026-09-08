import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


#Regression
# load data
dat1 = pd.read_csv('data/assignment_1_data.csv')

# look at the data
print(dat1.head())
print(dat1.shape)

# histograms
for var in ['watch.time','confusion.points','key.points']:
    data = dat1[var]
    min_data = np.floor(data.min()).astype(int)
    max_data = np.ceil(data.max()).astype(int)
    bins = range(min_data, max_data + 2)  # +2 to include the rightmost edge
    plt.hist(data, bins=bins, edgecolor='k', alpha=0.7)
    plt.title(var)
    plt.show()

# bar plots
for var in ['year','video','participation']:
    ax = dat1[var].value_counts().plot(kind='bar', alpha=0.7)
    plt.title(var)
    plt.xticks(rotation=0)

    #add exact number on each bar
    for i, v in enumerate(dat1[var].value_counts()):
        ax.text(i, v + 2, str(v), ha='center', va='bottom', fontsize=10)
    plt.show()


# bar plot of watch.time by video
dat1.groupby('video')['watch.time'].mean().plot(kind='bar', alpha=0.7)
plt.ylabel('Average watch time')
plt.xticks(rotation=0)
plt.title('Average watch time by video')
plt.show()

#Simple linear regression
# Q. Does confusion predict watch time?
# predict watch.time from confusion.points
# create a linear regression model
model1 = LinearRegression()
model1.fit(dat1[['confusion.points']], dat1[['watch.time']])
print("Simple linear regression")
print(model1.coef_)
print(model1.intercept_)
print('\n')

# get the predicted values
yhat1 = model1.predict(dat1[['confusion.points']])
yhat1 = pd.Series(yhat1.flatten())

# get correlation between actual and predicted values
print('R^2 = ' + str(round(dat1['watch.time'].corr(yhat1)**2,2)))

# plot yhat vs. y
# plt.scatter(dat1['watch.time'], yhat1)
# add jitter
plt.scatter(dat1['watch.time'], yhat1 + np.random.normal(0, 0.5, len(dat1['watch.time'])), alpha=0.2)
# plot unity line for reference
ymax = max(dat1['watch.time'].max(), yhat1.max())
plt.plot([0, ymax], [0, ymax], color='red', linestyle='-', linewidth=2)
# add labels
plt.xlabel('Actual watch time')
plt.ylabel('Predicted watch time')
plt.title('Predicted vs. actual watch time')
plt.show()


#Multiple linear regression
#Q. What combination of factors predicts watch time?
# predict watch.time from confusion.points, key.points interacted with participation
dat1['key.points.participation'] = dat1['key.points'] * dat1['participation']
dat1['confusion.points.participation'] = dat1['confusion.points'] * dat1['participation']
#create a new column 'key.points.participant' first, 
#multiply every value in key.points by the corresponding value in participation
#this tests if key.points and participation work together. same thing for confusion.points
#why multiply? we ask here 'Does the effect of key.points on watch.time depend on participation?'
#so, High key.points + High participation -> maybe watch.time increases a lot
#or, High key.points + Low participation -> maybe watch.time increases less

#eg, [watch.time = 5 + 2×key.points + 3×participation] without interaction
#Student A: key.points=10, participation=0 → watch.time = 5 + 2(10) + 3(0) = 25
#Student B: key.points=10, participation=1 → watch.time = 5 + 2(10) + 3(1) = 28

#eg, [watch.time = 5 + 2×key.points + 3×participation + 0.5×(key.points × participation)] with interaction
#Student A: key.points=10, participation=0 → watch.time = 5 + 2(10) + 3(0) + 0.5(10×0) = 25
#Student B: key.points=10, participation=1 → watch.time = 5 + 2(10) + 3(1) + 0.5(10×1) = 28.5
#=> When participation=1, the effect of key.points is stronger. 
#Thus, Student who participates AND gets key.points -> watch.time boost is amplified
#but Student who doesn't participate: key.points alone has less effect.

# create a linear regression model
model2 = LinearRegression()
model2.fit(dat1[['confusion.points.participation','key.points.participation']], dat1[['watch.time']])
print("Multiple linear regression")
print(model2.coef_)
print(model2.intercept_)
print('\n')

# get the predicted values
yhat2 = model2.predict(dat1[['confusion.points.participation','key.points.participation']])
yhat2 = pd.Series(yhat2.flatten())

# get correlation between actual and predicted values
print('R^2 = ' + str(round(dat1['watch.time'].corr(yhat2)**2,2)))
#if R^2 = 0.82, it means your model explains 82% of watch.time

# plot yhat vs. y
# plt.scatter(dat1['watch.time'], yhat2)
# add jitter
plt.scatter(dat1['watch.time'], yhat2 + np.random.normal(0, 0.5, len(dat1['watch.time'])), alpha=0.2)
# plot unity line for reference
ymax = max(dat1['watch.time'].max(), yhat2.max())
plt.plot([0, ymax], [0, ymax], color='red', linestyle='-', linewidth=2)
# add labels
plt.xlabel('Actual watch time')
plt.ylabel('Predicted watch time')
plt.title('Predicted vs. actual watch time')