#import dependencies
import numpy as np
import pandas as pd
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
#data colllection and preprocessing
movies = pd.read_csv('./datasets/tmdb_5000_movies.csv')
credits = pd.read_csv('./datasets/tmdb_5000_credits.csv')
ratings = pd.read_csv('./datasets/ratings.csv')

# merging the datasets
movies = movies.merge(credits, on = 'title')

# Keeping important columns for recommendation
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


movies.dropna(inplace=True)


#for converting str to list
import ast 

def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name']) 
    return L
movies['genres'] = movies['genres'].apply(convert)


movies['keywords'] = movies['keywords'].apply(convert)



#keeping top 3 cast
def convert_cast(text):
    C = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            C.append(i['name'])
        counter+=1
    return C


def fetch_director(text):
    D = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            D.append(i['name'])
            break
    return D
movies['crew'] = movies['crew'].apply(fetch_director)



#converting to the list
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.sample()

#removing spaces
def remove_space(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1

movies['cast'] = movies['cast'].apply(remove_space)
movies['crew'] = movies['crew'].apply(remove_space)
movies['genres'] = movies['genres'].apply(remove_space)
movies['keywords'] = movies['keywords'].apply(remove_space)

# Concatinate all
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# droping those extra columns
new_df = movies[['movie_id','title','tags']]

# Converting list to str
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x))


# Converting to lower case
new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


from nltk.stem import PorterStemmer
ps = PorterStemmer()
def stems(text):
    T = []
    
    for i in text.split():
        T.append(ps.stem(i))
    
    return " ".join(T)
new_df['tags'] = new_df['tags'].apply(stems)

from sklearn.feature_extraction.text import TfidfVectorizer

# Create a TfidfVectorizer object
tfidf_vectorizer = TfidfVectorizer(max_features=5000,stop_words='english')

# Fit and transform your data
tfidf_matrix = tfidf_vectorizer.fit_transform(new_df['tags']).toarray()

# Get the feature names
feature_names = tfidf_vectorizer.get_feature_names_out()


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(tfidf_matrix)

# similarity
new_df[new_df['title'] == 'Spider-Man'].index[0]

from sklearn.metrics import confusion_matrix


def recommend(movie_name):


    list_of_all_titles = movies['title'].tolist()

    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)

    if not find_close_match:
        print("Movie not found.")
        return

    close_match = find_close_match[0]

    matching_rows = movies[movies['title'] == close_match]

    if matching_rows.empty:
        print("Movie not found.")
        return

    similarity_score = list(enumerate(similarity[matching_rows.index[0]]))

    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    print('Movies suggested for you: \n')
    recommended_movies = []
    for i, movie in enumerate(sorted_similar_movies):
        index = movie[0]
        title_from_index = movies[movies.index == index]['title'].values
        if len(title_from_index) > 0:
            recommended_movies.append(title_from_index[0])
            print(title_from_index[0])
        if i == 9:
            break
    return recommended_movies

def get_feedback(recommended_movies):
        # User feedback
    user_feedback = input('Did you like any of the recommended movies? (y/n): ')
    user_liked_movies = []
    if user_feedback.lower() == 'y':
        liked_movies = input('Enter the names of the movies you liked (comma-separated): ')
        user_liked_movies = [movie.strip() for movie in liked_movies.split(',')]

    # Calculate precision and recall
    precision = calculate_precision(recommended_movies, user_liked_movies)
    recall = calculate_recall(recommended_movies, user_liked_movies)

    print('Precision:', precision)
    print('Recall:', recall)

def calculate_precision(recommended_movies, user_liked_movies):
    if len(recommended_movies) == 0:
        return 0

    relevant_recommendations = set(recommended_movies) & set(user_liked_movies)
    precision = len(relevant_recommendations) / len(recommended_movies)
    return precision

def calculate_recall(recommended_movies, user_liked_movies):
    if len(user_liked_movies) == 0:
        return 0

    relevant_recommendations = set(recommended_movies) & set(user_liked_movies)
    recall = len(relevant_recommendations) / len(user_liked_movies)
    return recall

