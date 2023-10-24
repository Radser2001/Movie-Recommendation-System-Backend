from flask import Flask, request, jsonify
from Movie_Recommendation_Search_Results import recommend
from Movie_Recommendation_Genres import recommend_movies

app = Flask(__name__)


# Movie recommendation from search results
print(recommend())

# Movie recommendation from genres
print(recommend_movies(1))


@app.route("/", methods=["GET"])
def Home():
    return "Hello World"

if __name__ == "__main__":
    app.run(debug=True)