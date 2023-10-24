import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import math 

# Load the data
movies = pd.read_csv('D:\\Y3S1\\Information Retrieval and Web Analytics - IT3041\\Projects\\Movie-Recommendation-System-Backend\\dataset\\tmdb_5000_movies.csv')
credits = pd.read_csv('D:\\Y3S1\\Information Retrieval and Web Analytics - IT3041\\Projects\\Movie-Recommendation-System-Backend\\dataset\\tmdb_5000_credits.csv')
ratings = pd.read_csv('D:\\Y3S1\\Information Retrieval and Web Analytics - IT3041\\Projects\\Movie-Recommendation-System-Backend\\dataset\\ratings.csv')


# Merge the dataframes
credits = credits.merge(ratings, on='movie_id')
movies = movies.merge(credits, on='title')


# Create a Count Vectorizer for movie genres
cv = CountVectorizer()
genre_matrix = cv.fit_transform(movies['genres'])

# Create a user-movie rating matrix
rating_matrix = movies.pivot_table(index='userId', columns='title', values='rating').fillna(0)

# Calculate cosine similarity between movies based on user ratings
cosine_sim = cosine_similarity(rating_matrix)

# Create a mapping of movie titles to indices
movie_to_idx = {title: i for i, title in enumerate(rating_matrix.columns)}

# Function to recommend movies
def recommend_movies(user_id, cosine_sim=cosine_sim):
    # Get the user's ratings
    user_ratings = rating_matrix.loc[user_id]

    # Initialize an empty list to store movie recommendations
    recommendations = set()

    # Iterate over the user's rated movies
    for movie_title, user_rating in user_ratings.iteritems():
        if user_rating > 0:
            
            # Convert the rating to its ceiling value
            user_rating = math.ceil(user_rating)
            
            # Find movies similar to the user-rated movie
            movie_idx = movie_to_idx.get(movie_title)
            if movie_idx is not None:
                similar_movies = list(enumerate(cosine_sim[movie_idx]))

                # Sort the list of similar movies by similarity score
                similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

                # Extract the movie indices (excluding the movie itself)
                similar_movie_indices = [i for i, _ in similar_movies if i != movie_idx]

                # Recommend the top 10 similar movies
                top_similar_movies = similar_movie_indices[:10]

                # Add the recommended movies to the set
                recommendations.update(top_similar_movies)

    # Filter out-of-bounds indices and return the titles of the recommended movies
    valid_recommendations = [rating_matrix.columns[i] for i in recommendations if i < len(rating_matrix.columns)]
    return valid_recommendations

# Calling the recommend_movies() function for user ID 1
recommended_movies = recommend_movies(2)

