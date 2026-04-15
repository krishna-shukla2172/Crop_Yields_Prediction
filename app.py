from flask import Flask, request, render_template
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Load
model = pickle.load(open('rfr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Safe float
def safe_float(val, default):
    try:
        return float(val)
    except:
        return default

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Input
        Year = safe_float(request.form.get('Year'), 2020)
        rain = safe_float(request.form.get('average_rain_fall_mm_per_year'), 1000)
        pesticides = safe_float(request.form.get('pesticides_tonnes'), 50)
        temp = safe_float(request.form.get('avg_temp'), 25)
        Area = request.form.get('Area', 'India')
        Item = request.form.get('Item', 'Wheat')

        # DataFrame (FIXED)
        data = pd.DataFrame([{
            'Year': Year,
            'average_rain_fall_mm_per_year': rain,
            'pesticides_tonnes': pesticides,
            'avg_temp': temp,
            'Area': Area,
            'Item': Item
        }])

        print("INPUT DATA:\n", data)

        # Transform + Predict
        transformed = preprocessor.transform(data)
        prediction = model.predict(transformed)[0]

        # 📊 GRAPH
        labels = ['Rainfall','Pesticides','Temp','Yield']
        values = [rain, pesticides, temp, prediction]

        plt.figure()
        plt.bar(labels, values)
        plt.title("Crop Yield Prediction")

        graph_path = "static/graph.png"
        plt.savefig(graph_path)
        plt.close()

        return render_template('index.html',
                               prediction=round(prediction,2),
                               graph=graph_path)

    except Exception as e:
        print("🔥 FULL ERROR:", str(e))
        return render_template('index.html',
                               prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
