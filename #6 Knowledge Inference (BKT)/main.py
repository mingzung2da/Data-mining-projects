#In this project, the purpose is to predict student performance on problems from 
# logs of students interaction with Intelligent Tutoring Systems.
#BKT =a model that estimates whether a student has learned a skill based on their performance on problems

#ID: A unique id for each row
#Lesson: Learning module
#Student: Student ID
#KC: Knowledge component of each question item. One learning module can have multiple knowledge components.
#item: Item id. One knowledge component may correspond to multiple items.
#right: Whether the student answered the item correctly?
#firstattempt: Is this the first time the student answered the question?
#time: How long does it take for the student to answer the question (in seconds)?


# Import all required packages including pyBKT.models.Model!
import numpy as np
import pandas as pd
from pyBKT.models import Model
import matplotlib.pyplot as plt

#Basic model creation and evaluation

# before training, we need to set up the model with parameters (initializing the model)
model = Model(seed = 42, num_fits = 1, parallel = True) #seed -> consistently replicate the results 

#Data: Fetch Assistments and CognitiveTutor data. (data format for pyBKT are comma separated and tab separated files)
#Note: correctness is given by -1(no response), 0(incorrect), or 1(correct).
model.fetch_dataset('https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/as.csv', '.')
model.fetch_dataset('https://raw.githubusercontent.com/CAHLR/pyBKT-examples/master/data/ct.csv', '.')

#open the dataset and explore before using them.
#Note: the column names all differ between the two datasets
#data1
ct_df = pd.read_csv('ct.csv', encoding = 'latin')
print(ct_df.columns)
ct_df.head(5)
print(ct_df)

#data2
as_df = pd.read_csv('as.csv', encoding = 'latin', low_memory = False)
print(as_df.columns)
as_df.head(5)
print(as_df)

#Now, we'll fit BKT model to every skill in the CognitiveTutor dataset separately.
#Note: when skills aren't specified, it trains a separate model on all skills by default.
model.fit(data_path = 'ct.csv')
training_rmse = model.evaluate(data = ct_df)
training_auc = model.evaluate(data_path = "ct.csv", metric = 'auc')
print("Training RMSE: %f" % training_rmse)
print("Training AUC: %f" % training_auc)

#Model Prediction
#skipped

#Question 1
#Fit my own BKT model below on the 'Calculate unit rate' skill from the CognitiveTutor dataset on the provided training data.
#Evaluate the mean absolute error and AUC(a performance score used to eval how well a classification model distinguishes between two diff groups) of predicting on the given test set and training set.
#Note: we have already performed a train/test split below

#Test data proportion
test_prop = 0.2

model = Model(seed = 42, num_fits = 1)
ct_data = ct_df[ct_df['KC(Default)'] ==
                'Calculate unit rate'].set_index('Anon Student Id') #extract only 'Calculate unit rate', using stud ID as index for easier grouping
idx_split = np.array(ct_data.index.unique()) #gets unique stud ID, converts to array
np.random.seed(42)
np.random.shuffle(idx_split)

#create train/test sets
training_data = ct_data.loc[idx_split[int(test_prop * len(idx_split)):]] #split studs (20% to test, 80% to train)
test_data = ct_data.loc[idx_split[:int(test_prop * len(idx_split))]]

#reset indexs (converts stud ID from index back to regular column so the model can access it properly)
training_data.reset_index(inplace = True)
test_data.reset_index(inplace = True)

#initialize metrics
training_mae, training_auc, test_mae, test_auc = 0, 0, 0, 0

#fit model and evaluate
skill = 'Calculate unit rate'
model.fit(data = training_data, skills = skill)

#train the model on training data, and compute Mean absolute error & AUC
#this shows how well it learned(training metrics) & how it generalizes(test metrics)
training_mae = model.evaluate(data = training_data, metric = 'mae')
training_auc = model.evaluate(data = training_data, metric = 'auc')
test_mae = model.evaluate(data = test_data, metric = 'mae')
test_auc = model.evaluate(data = test_data, metric = 'auc')

print("Training Mean Absolute Error: %f" % training_mae)
print("Training AUC: %f" % training_auc)
print("Test Mean Absolute Error: %f" % test_mae)
print("Test AUC: %f" % test_auc)


training_data.head(5)


#Model Cross-Validation and Varients
#the code below runs the default BKT model on all skills in my dataset using 5-fold cross-validation
#Note: RMSE tells prediction quality (lower RSME means the model predicts performance more accurately)
#Note: the result of this code just shows which skills are easy/hard for the model to predict. 
#-so I pick one skill and test variants diagnostically to find the best model for it.
model.crossvalidate(data_path = 'ct.csv', folds = 5)

#So now, we test variants focusing on only 'Calculations with similar figures' skill to find best model for it.
skill = 'Calculations with Similar Figures'
metric = 'auc'

#Note: by default, standard BKT has 4 parameters: learn rate, guess rate, slip rate, forget rate
#Standard BKT (one learn, guess, slip, forget rate for all)
simple_cv = model.crossvalidate(data = as_df, skills = skill,
                                metric = metric)
simple_cv

#Multigs BKT (different guess/slip rates per problem type)
multigs_cv = model.crossvalidate(data_path = 'as.csv', skills = skill,
                                 multigs = True, metric = metric)
multigs_cv

#Multilearn BKT (different learn rates per problem type)
multilearn_cv = model.crossvalidate(data_path = 'as.csv', skills = skill,
                                    multilearn = True, forgets = True,
                                    metric = metric)
multilearn_cv

#Multiprior BKT (looks at each student's first response to infer their actual starting level of knowledge) 
multiprior_cv = model.crossvalidate(data_path = 'as.csv', skills = skill,
                                    multiprior = True, metric = metric,
                                    folds = 3)
multipair_cv = model.crossvalidate(data_path = 'as.csv', skills = skill,
                                   multipair = True, metric = metric,
                                   folds = 3)
pd.concat([multiprior_cv, multipair_cv], axis = 0)

#Combo BKT (combines multigs + multilearn + forgets all combined) 
#E.g., instead of [Learn = 0.2, Guess = 0.1, Slip = 0.05, Forget = 0.01 (for ALL problem types)]
#combo estimates...
# Fractions(skill):        Learn = 0.25, Guess = 0.15, Slip = 0.08, Forget = 0.03
# Decimalss(skill):         Learn = 0.18, Guess = 0.08, Slip = 0.04, Forget = 0.01
# Percentages(skill):      Learn = 0.30, Guess = 0.12, Slip = 0.06, Forget = 0.02
#so combo does more accurate prediction because it captures how different KC are actually learned. 
combo_cv = model.crossvalidate(data_path = 'as.csv', skills = skill,
                               forgets = True, multilearn = True,
                               multigs = True, metric = metric)
combo_cv

#Question 2
#which combination of model variants has the smallest crossvalidated loss for "Venn Diagram"?
#which model variant is the best performant with a minimal number of parameters?

models = {}
model = Model(seed = 42, num_fits = 1)

models['simple'] = model.crossvalidate(data_path = 'as.csv',
                                       skills = 'Venn Diagram')
models['multilearn'] = model.crossvalidate(data_path = 'as.csv',
                                           skills = 'Venn Diagram',
                                           multilearn = True)
models['multipair'] = model.crossvalidate(data_path = 'as.csv',
                                          skills = 'Venn Diagram',
                                          multipair = True)
models['multigs'] = model.crossvalidate(data_path = 'as.csv',
                                        skills = 'Venn Diagram',
                                        multigs = True)
models['forgets'] = model.crossvalidate(data_path = 'as.csv',
                                        skills = 'Venn Diagram',
                                        forgets = True)
models['multigs + forgets'] = model.crossvalidate(data_path = 'as.csv',
                                                  skills = 'Venn Diagram',
                                                  multigs = True,
                                                  forgets = True)
models['multilearn + forgets'] = model.crossvalidate(data_path = 'as.csv',
                                                     skills = 'Venn Diagram',
                                                     multilearn = True,
                                                     forgets = True)

df = pd.concat(models.values())
df['model type'] = models.keys()
plt.figure(figsize = (14, 6))
plt.plot(df['model type'], df['rmse'])
plt.title('RMSE of Model Variants for Venn Diagram')
plt.ylabel('RMSE')
plt.xlabel('Model Variants')
df.set_index('model type')


#Model Parameter Initialization and Visualization
#initialize any of the model parameters(initial probability student knows the skill) for a particular skill's BKT model before training.
#similar to SciKit learn, it's a way to initialize model parameters and view them after they're fitted.

#select the skill want to work with
skill = 'Box and Whisker'

#initialize the prior parameter to 1e-40(=0)
model.coef_ = {skill: {'prior': 1e-40}} #model.coef_ is a distionary that stores all model parameters
model.coef_

#train the model with the pre-initialized parameters and eval.
model.fit(data_path = 'as.csv', skills = skill, multigs = True)
low_prior_auc = model.evaluate(data_path = 'as.csv', metric = 'auc')

#display the prior value after training and the AUC. 
print("Fitted Prior Value: %f" % model.coef_[skill]['prior'])
print("Training AUC: %f" % low_prior_auc)
#Result shows Fitted prior valu: 0.000 because 1e-40 is too extreme(assuming students have zero knowledge about Box and Whisker, but actually students probably come with some baseline understanding).

#re-initialize the prior to be more reasonable.
model.coef_ = {skill: {'prior': 0.5}} #reset the prior to 0.5 (50% probability students know the skill), which is more realistic

#train and eval again
model.fit(data_path = 'as.csv', skills = skill, multigs = True)
normal_prior_auc = model.evaluate(data_path = 'as.csv', metric = 'auc')

print("Fitted Prior Value: %f" % model.coef_[skill]['prior'])
print("Training AUC: %f" % normal_prior_auc)


#Visualize the parameters of our fitted model in a Pandas DataFrame once it has been fit.
#we can easily plot them learn, forget, guess, and slip rates for multilearn and multiguess models.
model.fit(data_path = 'as.csv', skills = skill,
          forgets = True, multilearn = True,
          multigs = True)
model.params()

#I might get warnings for using indexing past lexsort, so I will disable these warnings.
import warnings
warnings.simplefilter(action='ignore')

# Plot the learns, forgets, slips and guesses for each of the classes.
params = model.params()
plt.figure(figsize = (12, 6))
plt.plot(params.loc[(skill, 'guesses')], label = 'Guesses')
plt.plot(params.loc[(skill, 'learns')], label = 'Learns')
plt.plot(params.loc[(skill, 'forgets')], label = 'Forgets')
plt.plot(params.loc[(skill, 'slips')], label = 'Slips')
plt.xlabel('Template ID')
plt.ylabel('Rate')
plt.title('BKT Parameters per Template ID Class')
plt.legend();
#numbers on X-axis refers to template IDs, a group of similar problems.
#e.g., 30059: "Which of these is a Venn diagram?" problems
#e.g., 30060: "How many circles overlap?" problems
#e.g., 30799: "Shade the intersection" problems


#Extended Data and Model Configuration
#models can get excessively long, maybe due to the number of model variants used, the number of folds/seed/metric for crossvalidation etc.
#we create a configuration dictionary that describes all the parameters we will pass in.
#Note: this is not a pyBKT feature, but just a Python feature, which is very powerful.
config = {'multigs': True,
          'multilearn': True,
          'skills': ['Box and Whisker', 'Circle Graph'],
          'forgets': True,
          'metric': 'accuracy',
          'folds': 4,
          'seed': 42 * 42}
model.crossvalidate(data_path = 'as.csv', **config)

#for other non-Assistments/CogTutor style datasets, we will need to specify the columns corresponding to each required column 
# (i.e. the user ID, correct/incorrect). This is because pyBKT only supports the automatic inference of column names for the AS/CT datasets.
#For that, we use a defaults dictionary or specify parameters for each model variant.
defaults = {'order_id': 'custom_order', #order_id specified by the column custom_order
            'skill_name': 'custom_skill', #skill_name is specified by the custom_skill column
            'correct': 'custom_answer'} #etc.

defaults['multigs'] = 'custom_gs_classes' #defaults dictionary can also contain columns specifying what columns correspond to the desired classes

defaults

#now, we'll use the mapping with the modified CogTutor dataset with the names we chose for the columns.
columns = ['Row', 'Anon Student Id', 'KC(Default)',
           'Correct First Attempt', 'Problem Name']
my_df = ct_df[columns]
my_df.columns = ['custom_order', 'Anon Student Id',
                 'custom_skill', 'custom_answer',
                 'custom_gs_classes']
my_df['custom_skill'] += ' (Custom)'
my_df.head(5)

#we can also crossvalidate/fit given these default column mappings.
model.crossvalidate(data = my_df, metric = 'auc',
                    defaults = defaults)


#Exploratory questions
#skipped
