from flask import Flask, request, jsonify
from Movie_Recommendation_Search_Results import recommend
from Movie_Recommendation_Genres import recommend_movies
import pymongo
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
connection_string = "mongodb+srv://DataDetectives:xHZ6tg7X2xauy78n@mrs01.cmrmjch.mongodb.net/"

# Create a MongoClient instance
client = pymongo.MongoClient(connection_string)

# Access your database and collection
db = client.MRS
collection = db.Movies


def retrieve_movie_details(movie_title):
    # Search the MongoDB collection for the movie by title
    movie_details = collection.find_one({"title": movie_title})
    # Convert the ObjectId to a string
    if movie_details and "_id" in movie_details:
        movie_details["_id"] = str(movie_details["_id"])
    return movie_details

@app.route("/recommended_movies/<int:user_id>", methods=["GET"])
def Genre_Based_Recommended_Movies(user_id):
    # The user_id is obtained from the URL path as an integer

    movie_list = recommend_movies(user_id)  # Get recommended movies

    recommended_movies_with_details = []

    for title in movie_list:
        # Extract the movie title from the DataFrame
        movie_title = title

        # Retrieve movie details from MongoDB
        movie_details = retrieve_movie_details(movie_title)

        if movie_details:
            recommended_movies_with_details.append(movie_details)

    return jsonify({"recommended_movies_genre": recommended_movies_with_details})

@app.route("/recommended_movies/<string:movie_name>", methods=["GET"])
def Search_Based_Recommended_Movies(movie_name):
    max_recommendations = 10  # Define the maximum number of recommended movies

    movie_list = recommend(movie_name)  # Get recommended movies
    print(movie_list)
    # Ensure the list of recommended movies does not exceed the maximum
    if len(movie_list) > max_recommendations:
        movie_list = movie_list[:max_recommendations]

    recommended_movies_with_details = []

    for title in movie_list:
        # Extract the movie title from the DataFrame
        movie_title = title

        # Retrieve movie details from MongoDB
        movie_details = retrieve_movie_details(movie_title)

        if movie_details:
            recommended_movies_with_details.append(movie_details)

    return jsonify({"recommended_movies_search": recommended_movies_with_details})



if __name__ == "__main__":
    app.run(debug=True)