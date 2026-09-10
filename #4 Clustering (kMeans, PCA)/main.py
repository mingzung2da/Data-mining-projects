#Data is about the student's knowledge status about the subject of Electrical DC machines.(data obtained from UCI ML repository)
#STG (The degree of study time for goal object materials)
#SCG (The degree of repetition number of user for goal object materials)
#STR (The degree of study time of user for related objects with goal object)
#LPR (The exam performance of user for related objects with goal object)
#PEG (The exam performance of user for goal objects)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Read the data from the file and look at column names
userknow = pd.read_csv('#4 Clustering (kMeans, PCA)/data/user-knowledge.csv')
userknow.columns
print(userknow.columns)

# Keep only the columns containing the data about student's knowledge
data = userknow.iloc[:,:5]
data.head()
print(data)

data = data[data != 0].dropna()
data = data.reset_index(drop=True)
data.head()
print(data)

# Plot histograms of the featuers to visualize the data
data.hist(bins=50, figsize = (8,8))
plt.show()

# Perform k-Means Clustering with values of k from 1 to 10 and plot k within WCSS
wcss = [] #WCSS(Within Cluster Sum of Squares) measures how tightly clustered the data points are around their cluster centers
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=400, n_init=20, random_state=0)
    #n_clusters=i: Sets the number of clusters for this iteration
    #init='k-means++': initialization. faster convergence
    #max_iter=400: Maximum iterations allowed before stopping
    #n_init=20: Runs the algorithm 20 times with different initializations and picks the best result
    #random_state=0: Fixes the random seed for reproducibility
    kmeans.fit(data)
    #train keans algorithm on your data, finding optimal centers
    wcss.append(kmeans.inertia_)
    #find the inertia from the fitted model and adds it to the list (lower inertia = tighter clusters)

plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Perform K-Means Clustering with 3 clusters
kmeans = KMeans(n_clusters=3, init='k-means++', max_iter=400, n_init=20, random_state=0)
kmeans.fit(data)
k_class = kmeans.predict(data)

# Using PCA to visualize the clusters (not to build a classifier)
pca = PCA(n_components=3) #to reduce data to 3 principal components
principalComponents = pca.fit_transform(data) #fit PCA to your data and transform it to 3 dimension
PDF = pd.DataFrame(data = principalComponents, columns = ['PC1', 'PC2', 'PC3']) #creates dataframes from the 3 principal components
# Add a column 'Class' to the data sets
PDF.loc[:, 'Cluster'] = pd.Series(k_class) #add Cluster column
data_class = data.copy()
data_class['Class'] = k_class
# Count of points in each cluster
print(PDF['Cluster'].value_counts())

# Assign a color to each cluster
PDF['Color'] = PDF['Cluster'].map({0 : 'red', 1 : 'blue', 2 : 'green'})
# Plot the first 2 principal components and color by cluster
a1 = PDF['PC1']
a2 = PDF['PC2']
a3 = PDF['PC3']
c1 = PDF['Color']
plt.scatter(a1, a2, c = c1, alpha=0.3)
plt.show()

print(data_class.groupby(['Class']).mean())

#Classification and Comparison
#Split the data into train and test data sets
X = data_class.iloc[:, :-1] #select all rows and columns excpet the last one
Y = data_class.iloc[:, -1]
xTrain, xTest, yTrain, yTest = train_test_split(X, Y, test_size = 0.30, random_state = 0)


# KNN(K-nearest neighbors) for various values of k 
#Now, we're testing KNN to "classify" new data points into clusters
#so far workflow: 
# (1)k-means created clusters(unsupervised)
# (2)PCA visualized the clusters in 2D/3D
# (3)KNN builds a classifier to predict cluster membership(supervised learning)
from sklearn.neighbors import KNeighborsClassifier
accuracy = []
for i in range(1,12):
    knn = KNeighborsClassifier(n_neighbors = i).fit(xTrain, yTrain)
    accuracy.append(knn.score(xTest, yTest))
    #testing different k values (1 to 11) in KNN

#plots which k gives the best accuracy
plt.plot(range(1,12), accuracy)
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('k vs. Accuracy for KNN')

# KNN model and evaluation for optimal value of k (8 in this case)
knn = KNeighborsClassifier(n_neighbors = accuracy.index(max(accuracy))+1).fit(xTrain, yTrain)
knn_predictions = knn.predict(xTest)
knn_accuracy = knn.score(xTest, yTest)
print('Accuracy for knn = ',knn_accuracy)

# Confusion Matrix for knn
knn_CM = confusion_matrix(yTest, knn_predictions)
print('Confusion Matrix for knn \n', knn_CM)



# Decision Tree Classifier and evaluation for optimal value of k
from sklearn.tree import DecisionTreeClassifier
dtree_model = DecisionTreeClassifier(max_depth = 2).fit(xTrain, yTrain)
dtree_predictions = dtree_model.predict(xTest)
dt_accuracy = dtree_model.score(xTest, yTest)
print('Accuracy for Decision Tree Classifier = ', dt_accuracy)

# Confusion Matrix for Decision Tree
DT_confmat = confusion_matrix(yTest, dtree_predictions)
print('Confusion Matrix for Decision Tree \n', DT_confmat)



# Naive Bayes model and evaluation for optimal value of k
from sklearn.naive_bayes import GaussianNB

NB = GaussianNB().fit(xTrain, yTrain)
NB_predictions = NB.predict(xTest)
NB_accuracy = NB.score(xTest, yTest)
print('Accuracy for Gaussian Naive Bayes =', NB_accuracy)

# Confusion Matrix for Naive Bayes
NB_confmat = confusion_matrix(yTest, NB_predictions)
print('Confusion Matrix for Naive Bayes \n', NB_confmat)

#Result shows that KNN classifier performed better than Naive Bayes and Decision Tree classifier in terms of accuracy

