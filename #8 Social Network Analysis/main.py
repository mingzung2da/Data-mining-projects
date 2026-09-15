#In this project, I analyze relationships between people
#Part1,2: analyze student comments (who comments on whose posts)
#Part3: analyze students by shared classes (who takes classes together)
#Part4,5: repeat in Gephi and reflect

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

D1 = pd.read_csv('#8 Social Network Analysis/data/comment-data.csv')
print(D1.head())
print(D1.shape)     
print(D1.dtypes) 
print('\n')

#convert student id variables to string type
#why? networkx requires ids to be strings, not integers. 
#if we keep them as integers, networkx might treat them as numeric objects and cause issues later
D1['comment.from'] = D1['comment.from'].astype(str)
D1['comment.to'] = D1['comment.to'].astype(str)
print(D1.dtypes)

#create edge list and count pairs (edge means connection from one stud to another stud)
#edge list = table showing who comments on whom and how many times
#eg, if student 3 commented on student 22 five times, EDGE will have one row with from=3, to=22, count=5
EDGE = D1.groupby(['comment.from', 'comment.to']).size().reset_index(name='count')

#rename columns to simpler names 
EDGE.columns = ["from", "to", "count"]

print(EDGE.head())
print(EDGE.shape)

# create node list with student characteristics
#why? some students only receive comments but never comment themselves
#we need ALL students in our network, so we build from both "from" "to" columns.

# extract all commenters and their info
V_FROM = D1[['comment.from', 'from.gender', 'from.major']]
V_FROM.columns = ['id', 'gender', 'major']

# extract all commentees and their info
V_TO = D1[['comment.to', 'to.gender', 'to.major']]
V_TO.columns = ['id', 'gender', 'major']

# combine both lists to get all unique students
VERTEX = pd.concat([V_FROM, V_TO], ignore_index=True)

# remove duplicates (same student might appear in both V_FROM and V_TO)
VERTEX = VERTEX.drop_duplicates()

# reset index to make it clean
VERTEX = VERTEX.reset_index(drop=True)

print(VERTEX.head()) 
print(VERTEX.shape)
print(VERTEX.dtypes)
#completed list of all students with their gender & major. This will be attached to the graph as node.

#create directed graph from edge list
#directed = arrows show direction of comments
G = nx.from_pandas_edgelist(EDGE, 'from', 'to', ['count'], create_using=nx.DiGraph())

print(f"Number of nodes: {G.number_of_nodes()}")
print(f"Number of edges: {G.number_of_edges()}")

#Add student characteristics to each node in the graph
# create dictionary mapping student id to gender
gender_dict = dict(zip(VERTEX['id'], VERTEX['gender']))

# create dictionary mapping student id to major
major_dict = dict(zip(VERTEX['id'], VERTEX['major']))

#attach those characteristics as node
nx.set_node_attributes(G, gender_dict, 'gender')
nx.set_node_attributes(G, major_dict, 'major')

print(G.nodes['3']) #just checking what attributes node '3' has


#Visualize the network
#map gender to colors
color_map = {'A': 'blue', 'B': 'pink'}  # define what color represents each gender
colors = [color_map.get(G.nodes[node]['gender']) for node in G.nodes()]

#make edge width according to 'count'
edge_widths = [G[u][v]['count']/10 for u, v in G.edges()]
#calculate node positions 
pos = nx.spring_layout(G, k=0.5, iterations=50) #k=strength of repulsion (higher = more spread out), iterations=how many times the algorithm runs (higher = better positioning)

plt.figure(figsize=(12, 10))
nx.draw(G, pos, 
        node_color=colors,
        width=edge_widths,
        with_labels=True,
        )

plt.title("Student Comment Network")
plt.show()


#Part2: modify the graph
from matplotlib.patches import Patch

#size by comments received 
comments_received = dict(G.in_degree(weight='count'))
max_received = max(comments_received.values())
#base size of 100 -> keeps students with 0 comments visible
node_sizes = [100 + 1500 * comments_received[n] / max_received for n in G.nodes()]

#color by major
majors = sorted(set(nx.get_node_attributes(G, 'major').values()))
palette = plt.cm.tab10.colors
major_colors = {m: palette[i % len(palette)] for i, m in enumerate(majors)}
node_colors = [major_colors[G.nodes[n]['major']] for n in G.nodes()]

#thinner edges scaled to the max count so they don't cover nodes
max_count = max(nx.get_edge_attributes(G, 'count').values())
edge_widths = [0.5 + 2.5 * G[u][v]['count'] / max_count for u, v in G.edges()]

pos = nx.spring_layout(G, k=0.8, iterations=100, seed=42)

plt.figure(figsize=(14, 12))
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color='gray', alpha=0.4,
                       arrowstyle='-|>', node_size=node_sizes)
nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes,
                       edgecolors='black', linewidths=0.5)
nx.draw_networkx_labels(G, pos, font_size=8)

legend = [Patch(facecolor=major_colors[m], label=m) for m in majors]
plt.legend(handles=legend, title='Major', loc='upper left')
plt.title("Student Comment Network (Color = Major, Size = Comments Received, # in Node = Student ID)")
plt.tight_layout()
plt.show()


#Part 3: class network analysis (how students are connected through the courses)
#To see: who share classes with the most people, whether students with the same interest cluster together.
D2 = pd.read_csv('#8 Social Network Analysis/data/hudk4050-classes.csv')
print(D2.head())

#Data cleaning: the data is exported from Qualtrics, so it might be messy
#look for missing values, weird column names, formatting issues.
# remove row 0 (question labels) and row 1 (Qualtrics ImportId metadata)
D2 = D2.drop([0, 1]).reset_index(drop=True)

# rename columns to simpler names
class_columns = [f'class_{i}' for i in range(1, 7)]
D2.columns = ['first_name', 'last_name'] + class_columns + ['interest']
#eg, Q8=first_name, Q9=last_name, Q1=class_1, Q3=class_2, Q4=class_3, 
# Q5=class_4, Q6=class_5, Q7=class_6, Q18=interest

# combine first and last name
D2['name'] = D2['first_name'].str.strip() + ' ' + D2['last_name'].str.strip()

# standardize class names: remove spaces and make uppercase
for col in class_columns:
    D2[col] = D2[col].str.replace(' ', '').str.upper()

print("Cleaned data:")
print(D2[['name', 'class_1', 'class_2', 'class_3', 'interest']].head())

#Reshape to person-class format
D2_long = D2.melt(id_vars=['name', 'interest'], 
                   value_vars=class_columns,
                   value_name='class_name')
#why person-class first and then person-person format?
#my network's nodes are people, so networkx need people-to-people connections.
#the person-class format needs to be done first, because you can't see who shares a class until knowing each person take which lass.

# remove empty values
D2_long = D2_long.dropna(subset=['class_name'])
D2_long = D2_long[D2_long['class_name'] != 'HUDK4050'] #without this, the result isn't instructive, since almost everyone is connected to HUDK 4050
print(f"Person-class pairs: {D2_long.shape[0]}")
print(f"Unique classes: {D2_long['class_name'].nunique()}")

#build person-person adjacency matrix
students = sorted(D2_long['name'].unique())
person_person_matrix = pd.DataFrame(0, index=students, columns=students)

for class_name in D2_long['class_name'].unique():
    students_in_class = D2_long[D2_long['class_name'] == class_name]['name'].unique()
    for student1 in students_in_class:
        for student2 in students_in_class:
            if student1 != student2:
                person_person_matrix.loc[student1, student2] += 1

#convert to graph
G2 = nx.from_pandas_adjacency(person_person_matrix, create_using=nx.Graph())

print(f"Students: {G2.number_of_nodes()}")
print(f"Connections: {G2.number_of_edges()}")
print(f"Density: {nx.density(G2):.4f}")


#calculate centrality metrics 
#centrality = measure how influential a node is in the network
degree_centrality = nx.degree_centrality(G2)
betweenness_centrality = nx.betweenness_centrality(G2)

print("Degree centrality")
for student, cent in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{student}: {cent:.4f}")
#degree = most connected students

print("\nBetweenness centrality")
for student, cent in sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:5]:
    print(f"{student}: {cent:.4f}")
#betweenness = bridge students connecting different groups

#map interest to nodes
interest_dict = dict(zip(D2['name'], D2['interest']))
nx.set_node_attributes(G2, interest_dict, 'interest')
print(f"Interest distribution:")
print(D2['interest'].value_counts())


#Visualize
unique_interests = sorted(set(interest_dict.values()))
palette = plt.cm.tab10.colors
interest_colors = {interest: palette[i % len(palette)] for i, interest in enumerate(unique_interests)}

node_colors = [interest_colors[G2.nodes[node]['interest']] for node in G2.nodes()]
node_sizes = [betweenness_centrality[node] * 5000 + 100 for node in G2.nodes()]
#node_sizes = [degree_centrality[node] * 3000 + 100 for node in G2.nodes()]

pos2 = nx.spring_layout(G2, k=0.9, iterations=100, seed=42)

plt.figure(figsize=(14, 12))
nx.draw_networkx_edges(G2, pos2, width=0.5, edge_color='gray', alpha=0.3)
nx.draw_networkx_nodes(G2, pos2, node_color=node_colors, node_size=node_sizes, 
                       edgecolors='black', linewidths=0.5)
nx.draw_networkx_labels(G2, pos2, font_size=7)

legend = [Patch(facecolor=interest_colors[i], label=i) for i in unique_interests]
plt.legend(handles=legend, loc='upper left', title='Interest')
plt.title("Class Network (Color = Interest, Size = Degree Centrality)")
plt.axis('off')
plt.tight_layout()
plt.show()

#detect clusters
from networkx.algorithms import community
clusters = community.greedy_modularity_communities(G2)

for i, cluster in enumerate(clusters):
    print(f"Cluster {i+1}: {len(cluster)} students")
    cluster_interests = [interest_dict[s] for s in cluster]
    print(f"  Interests: {dict(pd.Series(cluster_interests).value_counts())}")


#Part4: do the same thing with Gephi
# Export nodes
nodes_list = []
for node in G2.nodes():
    nodes_list.append({
        'Id': node,
        'interest': interest_dict.get(node, 'Unknown'),
        'degree': degree_centrality[node]
    })
nodes_df = pd.DataFrame([{'Id': n, 'Label': n, 'interest': interest_dict[n]} for n in G2.nodes()])
nodes_df.to_csv('nodes.csv', index=False)

# Export edges
edges_list = []
for u, v in G2.edges():
    edges_list.append({'Source': u, 'Target': v})
edges_df = pd.DataFrame(edges_list)
edges_df.to_csv('edges.csv', index=False)

print("Exported nodes.csv and edges.csv")