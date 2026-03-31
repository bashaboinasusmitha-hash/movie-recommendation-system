import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
# Load data
data = pd.read_csv("/content/movie_recommendation.csv")
#clean movie titles
data["movie_title"]=data["movie_title"].str.strip().str.lower()
# Encode Genres 
encoder = LabelEncoder()
data["genre"] = encoder.fit_transform(data["genre"])
# Create similarity matrix
similarity_matrix = cosine_similarity(data[["genre"]])
# Recommendation function
def movie_recommendation(movie_name):
    if movie_name not in data["movie_title"].values:
        print("Movie not found!")
        print("Available movies:")
        print(data["movie_titles"].head(10))
        return
    movie_index = data[data["movie_title"] == movie_name].index[0]
    similarity_scores = list(enumerate(similarity_matrix[movie_index]))
    # Sort by similarity
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)
    print(f"\nRecommended Movies for {movie_name}:")
    for i in similarity_scores[1:4]:
        print(data.iloc[i[0]]["movie_title"])
# Test
movie_recommendation("Hidden Dream")
movie_recommendation("Dark Island")
movie_recommendation("Avengers")
