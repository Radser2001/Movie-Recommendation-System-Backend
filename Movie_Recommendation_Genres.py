import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data
movies = pd.read_csv("D:\\tmdb_5000_movies.csv")

# Clean the genre column
movies['genres'] = movies['genres'].fillna('')  # Fill missing values with empty string
movies['genres'] = movies['genres'].apply(lambda x: ' '.join(sorted(x.split('|'))))  # Sort and join genre names

# Create a TF-IDF vectorizer
tfidf = TfidfVectorizer(stop_words='english')

# Fit and transform the genre column
tfidf_matrix = tfidf.fit_transform(movies['genres'])

# Compute the cosine similarity matrix
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)


# Function to recommend movies
def get_recommendations(genres, cosine_sim=cosine_sim, movies=movies):
    # Filter movies based on selected genres
    selected_movies = movies[movies['genres'].str.contains('|'.join(genres))]

    # Get the indices of the selected movies
    selected_indices = selected_movies.index

    # Compute the average similarity scores for the selected movies
    avg_scores = cosine_sim[selected_indices].mean(axis=0)

    # Sort the movies based on average similarity scores
    sorted_indices = avg_scores.argsort()[::-1]

    # Get the top 5 most similar movies
    top_movies_indices = sorted_indices[:5]
    top_movies = movies['title'].iloc[top_movies_indices].tolist()

    return top_movies



