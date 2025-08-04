from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load("lasso_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form values
        budget = float(request.form["Budget"])
        cast_popularity = float(request.form["Cast_Popularity"])
        director_success = float(request.form["Director_Success_Rate"])
        social_media_mentions = float(request.form["Social_Media_Mentions"])
        genre_popularity = float(request.form["Genre_Popularity"])
        trailer_views = float(request.form["Trailer_Views"])
        release_month = int(request.form["Release_Month"])
        expected_screens = int(request.form["Expected_Screens"])

        # Prepare data for prediction
        features = np.array([[budget, cast_popularity, director_success,
                               social_media_mentions, genre_popularity,
                               trailer_views, release_month, expected_screens]])

        # Make prediction
        prediction = model.predict(features)[0]

        return render_template("result.html", prediction=round(prediction, 2))

    except Exception as e:
        return render_template("result.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)
