import pickle
import numpy as np
from flask import Flask, render_template, request
from algorithm import LinearRegression  


app = Flask(__name__)

# Load the new model, scaler, and polynomial transformer
new_model = pickle.load(open('new_model.pkl', 'rb'))
new_scaler = pickle.load(open('new_scaler.pkl', 'rb'))
new_poly = pickle.load(open('new_poly.pkl', 'rb'))

# Load the old model and old scaler
old_model = pickle.load(open('car_prediction.model', 'rb'))
old_scaler = pickle.load(open('old_scaler.pkl', 'rb'))

# Initialize a dictionary to hold prediction history for both models
prediction_history = {
    "old": [],
    "new": []
}

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict_old", methods=["GET", "POST"])
def predict_old():
    return predict(model=old_model, scaler=old_scaler, poly=None, history_key="old", model_name="Old Model")

@app.route("/predict_new", methods=["GET", "POST"])
def predict_new():
    return predict(model=new_model, scaler=new_scaler, poly=new_poly, history_key="new", model_name="New Model")

def predict(model, scaler, poly, history_key, model_name):
    if request.method == "GET":
        return render_template("predict.html", prediction_history=prediction_history[history_key], model_type=model_name)

    # Handle form inputs and assign default values
    year = request.form.get("year", "").strip()
    max_power = request.form.get("max_power", "").strip()
    engine = request.form.get("engine", "").strip()
    owner = request.form.get("owner", "1").strip()
    fuel = request.form.get("fuel", "0").strip()
    transmission = request.form.get("transmission", "0").strip()
    action = request.form.get("action")  # Get the action (e.g., "Clear")

    if action == "Clear":  # Clear history for this model
        prediction_history[history_key] = []
        return render_template("predict.html", prediction_history=prediction_history[history_key], model_type=model_name)

    # Convert inputs safely (use defaults if empty)
    year = int(year) if year.isdigit() else 2015
    max_power = float(max_power) if max_power.replace(".", "", 1).isdigit() else 80.0
    engine = int(engine) if engine.isdigit() else 1500
    owner = int(owner) if owner.isdigit() else 1
    fuel = int(fuel) if fuel.isdigit() else 0
    transmission = int(transmission) if transmission.isdigit() else 0

    input_data = np.array([[year, max_power, engine, owner, fuel, transmission]])

    # Scale input first!
    scaled_input = scaler.transform(input_data)

    # Apply polynomial transformation if needed (AFTER SCALING)
    if poly:
        transformed_input = poly.transform(scaled_input)
    else:
        transformed_input = scaled_input

    # Predict price
    predicted_price_log = model.predict(transformed_input)
    predicted_price = np.exp(predicted_price_log[0])  # Convert log price back to normal price

    # Prepare parameters for display
    parameters = f"Year: {year}, Max Power: {max_power}, Engine: {engine}, " \
                 f"Owner: {['First', 'Second', 'Third', 'Fourth & Above'][owner - 1]}, " \
                 f"Fuel: {['Petrol', 'Diesel'][fuel]}, Transmission: {['Manual', 'Automatic'][transmission]}"

    # Append prediction and parameters to history
    prediction_history[history_key].append({"price": f"${predicted_price:,.2f}", "parameters": parameters})

    # Render the prediction page
    return render_template("predict.html", prediction_history=prediction_history[history_key], model_type=model_name)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)

