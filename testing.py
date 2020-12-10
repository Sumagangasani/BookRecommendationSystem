import pandas as pd
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import correlation
from sklearn.metrics.pairwise import pairwise_distances
import ipywidgets as widgets
from IPython.display import display, clear_output
from contextlib import contextmanager
import warnings
from warnings import simplefilter
import random
warnings.filterwarnings('ignore')
import os, sys
import re
import seaborn as sns
import sklearn
from sklearn import model_selection
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import seaborn as sns


books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']
books.drop(['imageUrlS', 'imageUrlM', 'imageUrlL'],axis=1,inplace=True)

#checking the shapes of the dataset
print (books.shape)
print (users.shape)
print (ratings.shape)

print(books.yearOfPublication.unique())

#investigating the rows having 'DK Publishing Inc' as yearOfPublication
books.loc[books.yearOfPublication == 'DK Publishing Inc',:]
#From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','yearOfPublication'] = 2000
books.loc[books.ISBN == '0789466953','bookAuthor'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','bookTitle'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','yearOfPublication'] = 2000
books.loc[books.ISBN == '078946697X','bookAuthor'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','bookTitle'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

#investigating the rows having 'Gallimard' as yearOfPublication
#making required corrections as above, keeping other fields intact
books.loc[books.ISBN == '2070426769','yearOfPublication'] = 2003
books.loc[books.ISBN == '2070426769','bookAuthor'] = "Jean-Marie Gustave Le ClÃ?Â©️zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','bookTitle'] = "Peuple du ciel, suivi de 'Les Bergers"
#Correcting the dtypes of yearOfPublication
books.yearOfPublication=pd.to_numeric(books.yearOfPublication, errors='coerce')
print(books.yearOfPublication.unique())


books.loc[(books.yearOfPublication > 2006) | (books.yearOfPublication == 0),'yearOfPublication'] = np.NAN
books.yearOfPublication.fillna(round(books.yearOfPublication.mean()), inplace=True)

#exploring 'publisher' column
books['publisher'] = books['publisher'].fillna('other')




#two NaNs
print (sorted(users.Age.unique()))
#Age column has some invalid entries like nan, 0 and very high values like 100 and above
users.loc[(users.Age > 90) | (users.Age < 5), 'Age'] = np.nan
users.Age = users.Age.fillna(users.Age.mean())
#setting the data type as int
users.Age = users.Age.astype(np.int32)


#ratings dataset will have n_users*n_books entries if every user rated every item, this shows that the dataset is very sparse

ratings.bookRating.unique()

#ratings dataset should have books only which exist in our books dataset, unless new books are added to books dataset
ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
print(ratings)
print(ratings_new)

#ratings dataset should have ratings from users which exist in users dataset, unless new users are added to users dataset
ratings = ratings[ratings.userID.isin(users.userID)]



#As quoted in the description of the dataset -
#BX-Book-Ratings contains the book rating information. Ratings are either explicit, expressed on a scale from 1-10
#higher values denoting higher appreciation, or implicit, expressed by 0
ratings_explicit = ratings_new[ratings_new.bookRating != 0]
ratings_implicit = ratings_new[ratings_new.bookRating == 0]

#plotting count of bookRating
sns.countplot(data=ratings_explicit , x='bookRating')
plt.show()
#It can be seen that higher ratings are more common amongst users and rating 8 has been rated highest number of times

#Similarly segregating users who have given explicit ratings from 1-10 and those whose implicit behavior was tracked
users_exp_ratings = users[users.userID.isin(ratings_explicit.userID)]
users_imp_ratings = users[users.userID.isin(ratings_implicit.userID)]

counts1 = ratings_explicit['userID'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['userID'].isin(counts1[counts1 >= 100].index)]
counts = ratings_explicit['bookRating'].value_counts()
ratings_explicit = ratings_explicit[ratings_explicit['bookRating'].isin(counts[counts >= 100].index)]

#Generating ratings matrix from explicit ratings table
ratings_matrix = ratings_explicit.pivot(index='userID', columns='ISBN', values='bookRating')
userID = ratings_matrix.index
ISBN = ratings_matrix.columns
print(ratings_matrix.shape)
print(ratings_matrix.head())
#Notice that most of the values are NaN (undefined) implying absence of ratings


n_users = ratings_matrix.shape[0] #considering only those users who gave explicit ratings
n_books = ratings_matrix.shape[1]
print (n_users, n_books)

#since NaNs cannot be handled by training algorithms, replacing these by 0, which indicates absence of ratings
#setting data type
ratings_matrix.fillna(0, inplace = True)
ratings_matrix = ratings_matrix.astype(np.int32)



books.drop(['bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher'],axis=1,inplace=True)
users.drop(['Location'],axis=1,inplace=True)
res2 = ratings.merge(books,left_on='ISBN',right_on='ISBN')
output = res2.merge(users,left_on='userID',right_on='userID')

print(output.head())

print(output.shape)
print(output.describe())

columns = output.columns.tolist()

columns = [c for c in columns if c not in ["ISBN","userID"]]

target= "ISBN"

x =   output[columns]
y= output[target]
X = x.iloc[:15000,:]
Y = y.iloc[:15000]

print(X.shape)
print(Y.shape)




seed=7
#scoring= 'accuracy'
simplefilter(action='ignore',category=FutureWarning)
from sklearn.ensemble import RandomForestClassifier

#models = []
#models.append(('SVM',SVC(gamma='auto',kernel='rbf')))
#models.append(('RFC',RandomForestClassifier(max_depth=5,n_estimators=40)))
#models.append(('LR', LogisticRegression()))
#models.append(('LDA', LinearDiscriminantAnalysis()))
#models.append(('KNN', KNeighborsClassifier()))
#models.append(('CART', DecisionTreeClassifier()))
#models.append(('NB', GaussianNB()))
#models.append(('SVM', SVC()))

test_size = 0.33
'''
num_instances = len(X)
kfold = model_selection.ShuffleSplit(n_splits=10, test_size=test_size, random_state=seed)
model = LogisticRegression()
results = model_selection.cross_val_score(model, X, Y, cv=kfold)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)

model = LogisticRegression()

model.fit(X_train, Y_train)
print("checking")
result = model.score(X_test, Y_test)
print("Accuracy: %.3f%%" % (result*100.0))
'''
#split our data
X_train,X_test,y_train,y_test = model_selection.train_test_split(X,Y,test_size=test_size,random_state=7)

seed=7
scoring= 'accuracy'
simplefilter(action='ignore',category=FutureWarning)
from sklearn.ensemble import RandomForestClassifier

models = []
models.append(('SVM',SVC(gamma='auto',kernel='rbf')))
models.append(('RFC',RandomForestClassifier(max_depth=5,n_estimators=40)))
models.append(('KNN', LogisticRegression()))
models.append(('LR', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))



results=[]
names =[]
for name,model in models:
    kfold = model_selection.KFold(n_splits =5,random_state= seed)
    cv_results = model_selection.cross_val_score(model,X_train,y_train,cv=kfold,scoring=scoring)
    results.append(cv_results)
    names.append(name)
    msg= "%s: %.3f%% (%.3f%%)" % (name,cv_results.mean()*100,cv_results.std()*100)
    print(msg)




