#ARM finds relationships between items in data. In this case, 'what student behaviors and affects tend to occur together?'
#eg, if student is on-task AND on-task conversation -> they're likely concentrating
#Why anonid?: anonid(anonymous ID) identifies individual students as protecting privacy.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
from prefixspan import PrefixSpan
from collections import defaultdict

# load data
dat = pd.read_csv('#7 Relationship Mining (ARM)/data/assignment_6_data.csv')

#look at the data
print("Data shape:", dat.shape)
print(dat.head())

#create lists of behavior and affective items to analyze
behavior_items = ['behavior-ontask', 'behavior-ontaskconv', 'behavior-offtask']
affect_items = ['affect-frustrated', 'affect-concentrating', 'affect-confused', 'affect-bored']

#how many unique anonids(anonymous identifiers) are there?
#=how many individual, unidentified students are there
print("Number of unique students:", len(dat.anonid.unique()))

#descriptive statistics for 'how many observations per student'
#why do this?: if a student has only few observations, any pattern we find is probably noise/random chance
#so a student with eg, 100 observations gives you much more reliable data about their actual behaviors
print(dat.groupby('anonid').size().describe())

# histogram of rows per student
print("Histogram of observations per student")
fig1 = plt.figure(figsize=(10, 5))
dat.groupby('anonid').size().hist(bins=20, edgecolor='black')
plt.xlabel('Number of Observations')
plt.ylabel('Number of Students')
plt.title('Distribution of Observations per Student')
plt.tight_layout()
plt.show()
#dat.groupby('anonid').size().hist()


# what percent of students have 25 or more obervations?
print(sum(dat.groupby('anonid').size() >= 25) / len(dat.groupby('anonid').size()))

# what percent of students have 5 or less obervations?
print(sum(dat.groupby('anonid').size() <= 5) / len(dat.groupby('anonid').size()))

# we could consider keep students with a certain specified number of observations
# because the rule algorithms don't consider student as a variable
# so we might not want students with too few rows or too many rows
# but for now, I won't filter out any students


# for each anonid, line plot for number of 1's for each item
# create a list of the items
items = behavior_items + affect_items

# create a list of the anonids(unique student IDs)
anonids = dat.anonid.unique()

#for each student, count how many times each behavior/affect occurred
item_counts = []
for anonid in anonids:
    item_counts.append(dat[dat.anonid == anonid][items].sum())

#convert to dataframe for easier plotting
item_counts_df = pd.DataFrame(item_counts)
print("Item counts per student\n")
print(item_counts_df.head())

# line plot for each anonid(behavior/affect trend across students)
print("\nLine plots for each behavior/affect")
fig2 = plt.figure(figsize=(12, 10))
item_counts_df.plot(subplots=True, layout=(3, 3), figsize=(12, 10))
plt.suptitle('Behavior and Affect Counts per Student')
plt.tight_layout()
plt.show()
#item_counts_df.plot.line(subplots=True, layout=(4,3), figsize=(10,10))


#Prepare data for ARM algorithm by removing ID columns
dat_items = dat.drop(['anonid', 'obsnum'], axis=1)

# Removing the 'behavior-ontask' as it was not used in the study (it was used a student's baseline behavior)
dat_items = dat_items.drop(['behavior-ontask'], axis=1)

#calculate how often each item appears (what % of observations had each behavior/affect).
support = dat_items.mean()
print(support)


#Association Mining with Apriori (Apriori algorithm finds frequent patterns in data)
#eg., with 7 behavior/affect items, there're 2*7=128 possible combinations. The algorithm automatically finds which combinations occur together frequently.
#so it finds itemsets that occur together often enough to be considered "frequent"
frequent_itemsets = apriori(dat_items, min_support=0.001, use_colnames=True) #find all itemsets that appear together frequently

#generate rules from the frequent itemsets
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1) 
#metric="lift" measures how much more likely itemsets occur together than by chance.
#lift=1: items are independent (no relationship)
#lift>1: items occur together more than expected (positive relationship)
#lift<1: items avoid each other (negative relationship)
#min_threshold=1: only keep rules with lift >=1

#Result shows co-occurance of 'offtask-bored' and 'ontaskconv-confused'.
#and 'ontaskconv-frustrated' was not significant. 

# View the rules
rules.sort_values('lift', ascending=False, inplace=True)
print(rules.head(10))

# Generating association rules
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0)  
#min_threshold=0: keep all rules. This shows which behavior/affects avoid each other

#Result of reducing the minimun Lift gets me a couple more associations, 'ontaskconv-concentrating', 'offtask-concentrating'

# View the rules
rules.sort_values('lift', ascending=False, inplace=True)
print(rules.head(10))


#Sequence Mining to get Transition Probabilities
#why we need Sequence Mining?: ARM(Association Rule Mining) (Apriori) finds relationships that occur together but ignores order.
#it answers "What is the order and flow of behaviors and affects? What comes after what?"
#What's it for?: To understand behavioral transitions. (eg, if you know a student is on-task conversation, what's likely to happen next?)

# preprocess data
dat_filtered = dat.drop(columns=['behavior-ontask']) #drop because behavior-ontask was baseline, not analyzed.
dat_filtered = dat_filtered.loc[dat_filtered.filter(regex='behavior-|affect-').sum(axis=1) > 1] #remove observations with no behavior or affect

#Sort by stident ID and obervation number so sequences are in chronological order for each student
dat_filtered = dat_filtered.sort_values(['anonid', 'obsnum'])

#Identify behaviors and affects
behaviors = dat_filtered.filter(regex='behavior-').columns.tolist() #extract column names containing 'bahavior-' and 'affect-'
affects = dat_filtered.filter(regex='affect-').columns.tolist()

#Build sequences: create a dictionary to store sequences for each student
sequences_dict = defaultdict(list)

for index, row in dat_filtered.iterrows():
    active_behavior = None #initialize placeholders 
    active_affect = None

    # Check each behavior to find which one is active (eg, behavior-ontaskconv=1 is active; behavior-offtask=0 is Not active)
    for behavior in behaviors:
        if row[behavior] == 1:  
            active_behavior = behavior
            break  #assuming only one active behavior at a time

    # Check each affect to find which one is active
    for affect in affects:
        if row[affect] == 1:  # or use > 0 if it can be a different positive number
            active_affect = affect
            break  # assuming only one active affect at a time

    if active_behavior is not None and active_affect is not None:
        sequences_dict[row['anonid']].append((active_behavior, active_affect))

data_sequences = list(sequences_dict.values())

# Use PrefixSpan to find frequent sequences
ps = PrefixSpan(data_sequences)
frequent_sequences = ps.frequent(2)  #frequent(2) means support threshold 2 (at least occurences)

# Calculate transition probabilities
transition_counts = defaultdict(lambda: defaultdict(int)) #create dictionaries
state_counts = defaultdict(int)

for sequence in data_sequences:
    for i in range(len(sequence) - 1):
        transition_counts[sequence[i]][sequence[i + 1]] += 1 #count how many times stateA transitions to stateB
        state_counts[sequence[i]] += 1 #count total times stateA appears

#convert counts to probabilities
transition_probabilities = {
    state: {next_state: count / state_counts[state] for next_state, count in transitions.items()}
    for state, transitions in transition_counts.items()
}

# Print the transition probabilities
for state_current, transitions in transition_probabilities.items():
    for state_next, probability in transitions.items():
        print(f"Transition from {state_current} to {state_next}: {probability:.2f}")


#Visualize as Network Graph
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict

# Create a directed multi-graph
G = nx.MultiDiGraph()

# Add edges to the graph based on transition probabilities
for source, targets in transition_probabilities.items():
    for target, probability in targets.items():
        # Use probability as weight
        G.add_edge(source, target, weight=probability)

#Scale edge widths so stronger transitions (higher probability) appear thicker
edge_weights = [data['weight'] for _, _, data in G.edges(data=True)]
max_edge_weight = max(edge_weights)
edge_widths = [0.2 + 2 * data['weight'] / max_edge_weight for _, _, data in G.edges(data=True)]

plt.figure(figsize=(12, 12))

# Create a layout for our nodes
pos = nx.spring_layout(G)

# Draw the graph, specifying we're using a MultiDiGraph
nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, edge_cmap=plt.cm.Blues, font_size=10, width=edge_widths, connectionstyle='arc3,rad=0.1')

#create probability labels to the arrows in the network diagram
edge_labels = {}
for source, target, data in G.edges(data=True):
    if (source, target) in edge_labels:
        edge_labels[(source, target)].append('{:.2f}'.format(data['weight']))
    else:
        edge_labels[(source, target)] = ['{:.2f}'.format(data['weight'])]

#position labels on the graph 
for (source, target), labels in edge_labels.items():
    x = (pos[source][0] + pos[target][0]) / 2
    y = (pos[source][1] + pos[target][1]) / 2
    offset = 1  
    for i, label in enumerate(labels):
        adjusted_x = x + offset * (i - len(labels) // 2)
        adjusted_y = y + offset * (i - len(labels) // 2)  # Separate labels vertically
        # plt.text(adjusted_x, adjusted_y, label, color='red', ha='center', va='center')

# Improve display of the plot
plt.xlim(-1.5, 1.5)  # Widen x limits
plt.ylim(-1.5, 1.5)  # Widen y limits
plt.show()