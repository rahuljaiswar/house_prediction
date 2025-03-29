from flask import Flask, render_template, request
import pickle
import numpy as np

# Load trained model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get user inputs from form
        area = float(request.form["area"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        garage_spaces = int(request.form["garage_spaces"])
        floors = int(request.form["floors"])
        lot_size = float(request.form["lot_size"])
        school_rating = float(request.form["school_rating"])
        crime_rate = float(request.form["crime_rate"])

        # Convert inputs into a NumPy array
        user_input = np.array([[area, bedrooms, bathrooms, garage_spaces, 
                                floors, lot_size, school_rating, crime_rate]])

        # Predict house price
        predicted_price = model.predict(user_input)[0]

        # Display result
        return render_template("result.html", price=round(predicted_price, 2))

    except Exception as e:
        return f"Error: {e}"

if __name__ == "__main__":
    app.run(debug=True)
