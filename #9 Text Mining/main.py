#This project is to build a small text mining pipeline on university student feedback.
#students rated 6 areas (teaching, course content, examination, lab work, library facilities, extracurricular) with a label (1 positive, 0 neutral, -1 negative) and written comment.
#the goal is to turn those comments into numbers (workflow below)
#1. Clean text so it's consistent
#2. Build a document-term matrix(DTM): each row is comment, each column is word, each cell is count
#3. Explore which labels and words are most common
#4. Find common topics with LDA(Latent Dirichlet Allocation)
#5. Classify sentiment with Naive Bayes: can the words predict whether a commnent is positive, neutral, or negative?

import pandas as pd, re

raw = pd.read_csv('#9 Text Mining/data/eval_data.csv')

cats = ['teaching','course_content','examination',
        'lab_work','library_facilities','extracurricular']

#Rename all 12 columns (ie, teaching_score, teaching_comment... extracurricular_comment)
raw.columns = [f'{c}_{t}' for c in cats for t in ['score','comment']]

#For each category, build a small table with 3 columns
data = pd.concat([pd.DataFrame({'category': c,
                                'score': raw[f'{c}_score'],
                                'comment': raw[f'{c}_comment']})
                  for c in cats], ignore_index=True)

#Clean text
def clean_text(t):               #define function that takes one comment (t) and returns cleaned version
    t = str(t).lower()           #str() makes sure the value is text
    t = re.sub(r'\d+', '', t)    #digits are replaces with nothing (eg,"2 labs"-> "labs")
    return re.sub(r'[^\w\s]', '', t) #any character that isn't a letter/number is removed.

data['comment'] = data['comment'].fillna('').apply(clean_text) #fillna('') replaces empty comments with black text (otherwise they become "nan")

#DTM(Document-term matrix)
from sklearn.feature_extraction.text import CountVectorizer

#create converter
vec = CountVectorizer(stop_words='english') #split text into words(tokenization) and ignore common English words like "the" "is" "and"

dtm = vec.fit_transform(data['comment'])  #fit learns the words, transform counts each word in comment 
print("DTM results\n", dtm)
#(result is a matrix: rows are comments, columns are words, cells are counts)

#just to see which word the numbers represent
words = vec.get_feature_names_out() 
print(words)
print(words[877])  #a word that 877 represents 

#Exploratory analysis
print("Exploratory analysis\n", pd.crosstab(data['score'], data['category']))

#counts how many comments have each label(-1,0,1), showing whether comments are mostly negative
freq = pd.Series(dtm.sum(axis=0).A1, index=words)
print("\nHow many comments have each label")
print(freq.sort_values(ascending=False).head(15))

#LDA topic modeling
#assume each comment is a mix of hidden topics and mix of words.
from sklearn.decomposition import LatentDirichletAllocation

#n_component=3 asks 3 topics. fit learns the topics from the word counts
lda = LatentDirichletAllocation(n_components=3, random_state=1).fit(dtm) 

#holds one row per topic, with a weight for every word (higher weight=more important)
print("\nCommon topics identified\n")
for i, topic in enumerate(lda.components_):
    print(f'Topic {i+1}:', [words[j] for j in topic.argsort()[-8:][::-1]]) #returns word position sorted from lowest to highest weight


#Naive Bayes classifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

#converts counts to binary (if the word appears in the comment=1, if not=0)
X = (dtm > 0).astype(int)

#split data 80% for training, 20% for testing. 
X_tr, X_te, y_tr, y_te = train_test_split(
    X, data['score'], train_size=0.8, random_state=456, stratify=data['score'])

#train the model (it learns how likely each word is to appear in positive, neutral, and negative comments)
nb = MultinomialNB().fit(X_tr, y_tr)

#predicts a label for each test comments based on the words it contains
pred = nb.predict(X_te)
print("Accuracy score=", accuracy_score(y_te, pred)) #eg, 0.85 means 85% predicted correctly
print("Confusion Matrix=\n", confusion_matrix(y_te, pred)) #3x3 table

#Heatmap
import seaborn as sns, matplotlib.pyplot as plt

sns.heatmap(confusion_matrix(y_te, pred), annot=True, fmt='d', cmap='Blues',
            xticklabels=nb.classes_, yticklabels=nb.classes_)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.show()

#Result: the model performs very well at detecting positive comments (158 out of 162 are correct)
#but it struggles with negative and neutral comments, often confusing them as positive

#Do the same work with Orange
#data.to_csv('#9 Text Mining/eval_long.csv', index=False)


# Save the cleaned Document-Term Matrix as CSV
data.to_csv('#9 Text Mining/dtm_matrix.csv', index=False)

# Save the binary version for Naive Bayes
X_binary = (dtm > 0).astype(int)
X_df = pd.DataFrame(X_binary.toarray(), columns=words)
X_df['score'] = data['score'].reset_index(drop=True)
X_df.to_csv('#9 Text Mining/dtm_for_classification.csv', index=False)