#Classification
import pandas as pd

data_for_classification = pd.read_csv('data/assignment_2_data.csv')
data_for_classification.head()
print(data_for_classification.head())

# Convert the 'certified' column to 0 and 1
data_for_classification['certified'] = data_for_classification['certified'].apply(lambda x: 0 if x == 'no' else 1)

# Encode other columns using one-hot encoding(creates multiple binary(numerical) colums from categorical columns for machine to understand)
data_encoded = pd.get_dummies(data_for_classification, columns=['forum.posts', 'grade', 'assignment'], drop_first=True)
#convert boolean columns to integers
data_encoded = data_encoded.astype(int)

# Obtain the processed data
processed_data_for_classification = data_encoded.copy()
print(processed_data_for_classification.head())


##Train a Classifier##
#Implement 3 model to train a classifier
#(1) Binary logistic regression model
#(2) Decision tree model
#(3) Naive Bayes model

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import numpy as np

#ModelEvaluator will handle model training & evaluation
class ModelEvaluator:
    def __init__(self, data, target_column, feature_columns):
        self.data = data
        self.target_column = target_column
        self.feature_columns = feature_columns
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        #X_train/X_test = features (input data), y_train/y_test = target (what you're predicting)
        self.models = {
            "logistic regression": LogisticRegression(),
            "decision tree": DecisionTreeClassifier(random_state=42),
            "naive bayes": GaussianNB(),
        }

    def prepare_data(self):
        # Extracting features and target variable
        features = self.data[self.feature_columns]
        target = self.data[self.target_column]
    
        # Splitting the data into training and testing sets
        #test_size=0.3 -> spliting data into 70% training, 30% testing
        #random_state -> without it, you get diff random split of data. With this, you always get the same split (42 is arbitrary num)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(features, target, test_size=0.3, random_state=42)

    def train_model(self, model_name):
        model = self.models[model_name.lower()]
        model.fit(self.X_train, self.y_train)
        #.fit() is the method that actually trains the model
        return model

    def evaluate_model(self, model_names):
        #create 2 empty dataframe to store results
        evaluate_result = pd.DataFrame()
        confusion_result = pd.DataFrame()

        for name in model_names:
            #converts to lowercase and checks if the model exists
            lower_name = name.lower()
            if lower_name not in self.models:
                raise ValueError(f"Unsupported model name: {name}. Supported models are 'Logistic Regression', 'Decision Tree', and 'Naive Bayes'.")

            #gets the model and trains it on training data
            model = self.models[lower_name]
            model.fit(self.X_train, self.y_train)

            # Model evaluation metrics
            y_pred_test = model.predict(self.X_test) #prediction on the test data (data that the model hasn't seen before)
            mse = mean_squared_error(self.y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            r2 = r2_score(self.y_test, y_pred_test)
            mae = mean_absolute_error(self.y_test, y_pred_test)

            result_dict = {
                "Model": name,
                "MSE": mse,
                "RMSE": rmse,
                "R2": r2,
                "MAE": mae
            }

            # Calculate the confusion matrix (shows how many predictions were correct/incorrect)
            confusion = confusion_matrix(self.y_test, y_pred_test)
            confusion_dict={
                "Model": name,
                "confusion":confusion
            }

            # Return the results
            evaluate_result = pd.concat([evaluate_result, pd.DataFrame([result_dict])], ignore_index=True)
            confusion_result = pd.concat([confusion_result, pd.DataFrame([confusion_dict])], ignore_index=True)

        return evaluate_result,confusion_result


# Build ModelEvaluator
#stores the target column name (what you're predicting)
target_column_name = 'certified'
#.toilist() converts columns to a list
feature_columns_name = [name for name in processed_data_for_classification.columns.tolist() if name != target_column_name] 
model_evaluator = ModelEvaluator(processed_data_for_classification, target_column=target_column_name, feature_columns=feature_columns_name)

# Preparing the data
model_evaluator.prepare_data()

# Train the data
model_names = ['Logistic Regression', 'Decision Tree', 'Naive Bayes']
evaluation_result,confusion_result = model_evaluator.evaluate_model(model_names)

print(evaluation_result)

#from the result, we see
#Naive Bayes model shows the lowest MSE and RMSE, suggesting its minimal predictive error. 
#Also, Naive Bayes model shows highest R2, suggesting its higher explanatory power for the target variable.
#All models show similar MAE values, with no significant distinctions.
#Overall, Naive Bayes proves as the most effective in this classification problem, probably due to its inherent robustness to small-sample data.


import matplotlib.pyplot as plt

# Create subplots with a shared y-axis
# creates figure with 1 row and 3 columns of 3 models
fig, axes = plt.subplots(1, len(confusion_result), figsize=(15, 5))

for i, row in confusion_result.iterrows():
    model_name = row['Model']
    confusion_matrix = np.array(row['confusion'])
    #extract the model name and confusion matrix from the row

    im = axes[i].imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    #display the confusion matrix as a color heatmap

    # sets axis labels to show prediction categories
    axes[i].set_xticks(np.arange(2))
    axes[i].set_yticks(np.arange(2))
    axes[i].set_xticklabels(['True Negative', 'True Positive'])
    axes[i].set_yticklabels(['True Negative', 'True Positive'])

    # Loop over data dimensions and create text annotations
    for j in range(len(confusion_matrix)):
        for k in range(len(confusion_matrix[0])):
            axes[i].text(k, j, str(confusion_matrix[j][k]), ha="center", va="center", color="w")

    axes[i].set_title(model_name)

    # Calculate and display accuracy
    total = np.sum(confusion_matrix)
    accuracy = (confusion_matrix[0][0] + confusion_matrix[1][1]) / total
    axes[i].text(0.5, -0.2, f'Accuracy: {accuracy:.2f}', ha="center", va="center", transform=axes[i].transAxes)
    #Accuracy = [(True positive + True negative)/Total predictions]

fig.tight_layout()
plt.show()