from flask import render_template, Flask, request
import pandas as pd
from keras.losses import cosine

from lightfmTest import *

app = Flask(__name__)
global UserID
#ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
#ratings.columns = ['userID', 'ISBN', 'bookRating']
bookDetails = getBookDetails()
bookDetails.drop('publisher',axis=1,inplace=True)

@app.route('/')
def temp():
    return render_template("signin.html")

@app.route('/books',methods=['POST'])
def books():
    id = request.form['j_username']
    UserID = int(id)
    ratings = getRatings(id)
    #print(ratings)
    bookIsbn = ratings.loc[:,'ISBN']
    result = bookDetails.loc[bookDetails['ISBN'].isin(bookIsbn)]
    isbnlist = ratings.loc[:,["ISBN", "bookRating"]]
    out = pd.merge(result,isbnlist, on='ISBN')
    #user_rat = ratings.loc[ratings["userID"] == int(id)]
    #return render_template("books.html", id=id, rat=ratings.loc[:,["ISBN", "bookRating"]])
    #rat_matrix = getRatingMatrix()
    #metric = cosine
    #predictedBooks = recommendItem(int(id), rat_matrix, metric)
    #boooks = getBookDetails()
    #result = []
    #for i in range(10):
    #    result.append(boooks.ISBN[predictedBooks.index[i]].encode('utf-8'))

    return render_template("books.html", id=id, rat=out)



@app.route('/recommend', methods=['POST'])
def recommend():
    NoOfBooks = request.form['noOfBooks']
    rat_matrix = getRatingMatrix()
    metric = cosine
    predictedBooks = recommendItem(109901,rat_matrix,metric)
    books = getBookDetails()
    result = []
    for i in range(int(NoOfBooks)):
        result.append(books.ISBN[predictedBooks.index[i]].encode('utf-8'))

    return render_template("recommend.html", id=UserID,BookList=result)

if __name__ == "__main__":
    app.run()