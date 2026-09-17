#This project is to visualize the data with different libraries.

import pandas as pd
import numpy as np
D1 = pd.read_csv('#10 Visualization/data/video-data.csv')

#Histograms
import matplotlib.pyplot as plt

# Generate a histogram of the watch time for the year 2018
D1_2018 = D1[D1['year'] == 2018]

plt.hist(D1_2018['watch.time'], bins=10)
plt.xlabel('Watch Time')
plt.ylabel('Frequency')
plt.title('Histogram of Watch Time for 2018')
plt.show()
# Change the number of bins to 100, do you get the same impression?
# Change the bins so that they are 0-5, 5-20, 20-25, 25-35. Hint: bins=range(0, 35, 5)

#plot confusion points vs watch time
jitter = np.random.normal(0,0.1,len(D1_2018))
plt.scatter(D1_2018['watch.time']+jitter, D1_2018['confusion.points']+jitter)
plt.xlabel('Watch Time')
plt.ylabel('Confusion Points')
plt.title('Scatterplot of Watch Time vs Confusion Points for 2018')
plt.show()
# Add jitter to make points visible that are on top of each other, using 
#jitter = np.random.normal(0,0.1,len(D1_2018))

# Also make the points transparent (alpha) to better distinguish different points, using
alpha=0.25


#Bar plots
#create plot of watch.time by year, showing mean and standard error
D1_by_year = D1.groupby('year').agg({'watch.time': ['mean', 'sem']})
D1_by_year.plot.bar(y='watch.time', yerr='sem')
plt.xlabel('Year')
plt.ylabel('Watch Time')
plt.title('Mean Watch Time by Year')
plt.show()

#plot similar bar plots for confusion.points and key.points in the same figure
D1_by_year_confusion = D1.groupby('year').agg({'confusion.points': ['mean', 'sem']})
D1_by_year_key = D1.groupby('year').agg({'key.points': ['mean', 'sem']})
fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10, 8))

# first subplot
D1_by_year_confusion['confusion.points']['mean'].plot.bar(yerr=D1_by_year_confusion['confusion.points']['sem'], ax=ax1, capsize=4)
ax1.set_xlabel('Year')
ax1.set_ylabel('Confusion Points')
ax1.set_title('Mean Confusion Points by Year')
ax1.set_ylim(0,6)

# second subplot
D1_by_year_key['key.points']['mean'].plot.bar(yerr=D1_by_year_key['key.points']['sem'], ax=ax2,
capsize=4)
ax2.set_xlabel('Year')
ax2.set_ylabel('Key Points')
ax2.set_title('Mean Key Points by Year')
ax2.set_ylim(0,6)
plt.tight_layout()
plt.show()

# also use the package plotnine to make plots using syntax similar to ggplot2 from R
from plotnine import *
print(ggplot(D1, aes(x='watch.time')) + geom_histogram(bins=10) + geom_histogram(bins=100) +
facet_wrap('~year'))


#Line Plots
# Plot the average watch time, confusion points, and key points by year as line plots with error bars
D1_by_year = D1.groupby('year').agg({'watch.time': ['mean', 'sem'], 'confusion.points': ['mean',
'sem'], 'key.points': ['mean', 'sem']})
fig, ax = plt.subplots()
ax.errorbar(D1_by_year.index, D1_by_year['watch.time']['mean'], yerr=D1_by_year['watch.time']
['sem'], label='Watch Time', fmt='-o')
ax.errorbar(D1_by_year.index, D1_by_year['confusion.points']['mean'],
yerr=D1_by_year['confusion.points']['sem'], label='Confusion Points', fmt='-o')
ax.errorbar(D1_by_year.index, D1_by_year['key.points']['mean'], yerr=D1_by_year['key.points']
['sem'], label='Key Points', fmt='-o')
ax.set_xlabel('Year')
ax.set_ylabel('Values')
plt.title('Mean Watch Time, Confusion Points, and Key Points by Year')
ax.legend()
plt.show()


#3D scatter & GIF
# plot 3d scatterplot of watch.time, confusion.points, and key.points, and color according to video, use only the rows where participation = 1
# make a gif view from multiple angles
# get index of participation = 1
import matplotlib.animation as animation

participated = D1[D1['participation'] == 1].index

# Create a color map for the 'video' categories
unique_videos = D1['video'].unique()
colors = plt.cm.jet(np.linspace(0, 1, len(unique_videos)))
color_map = dict(zip(unique_videos, colors))
colors = D1['video'][participated].map(color_map)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(D1['watch.time'][participated], D1['confusion.points'][participated], D1['key.points']
[participated], color=colors)
ax.set_xlabel('Watch Time')
ax.set_ylabel('Confusion Points')
ax.set_zlabel('Key Points')

# legend
for video in unique_videos:
    ax.scatter([], [], [], color=color_map[video], label=video)
ax.legend(loc='upper right')
def rotate(angle):
    ax.view_init(azim=angle)

ani = animation.FuncAnimation(fig, rotate, frames=range(0, 360, 2), interval=100)
ani.save('rotation.gif', writer='Pillow')
plt.show() 

from IPython.display import Image, display
display(Image(filename='rotation.gif'))


#tSNE
#to visualize more than 3 dimensions, I need to first reduce the dimensionality (-> use t-SNE)
from sklearn.manifold import TSNE

D2 = pd.read_csv('#10 Visualization/data/user-knowledge.csv')

# make tsne plot of the data
tsne = TSNE(n_components=2, random_state=0)
tsne_obj = tsne.fit_transform(D2)
tsne_df = pd.DataFrame({'X': tsne_obj[:, 0], 'Y': tsne_obj[:, 1]})
tsne_df.plot.scatter(x='X', y='Y')
plt.title('TSNE Plot of User Knowledge Data')
plt.show()

# perform k-means clustering on the data
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5)
kmeans.fit(D2)
y_kmeans = kmeans.predict(D2)
plt.scatter(tsne_df['X'], tsne_df['Y'], c=y_kmeans, s=50, cmap='viridis')
plt.title('K-Means Clustering of User Knowledge Data')
plt.show()

#do the same thing with Tableau
