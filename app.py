from flask import Flask, request, render_template
import numpy as np
import pickle

# Flask app
app = Flask(__name__)

# Load models
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# Home route
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    def safe_float(val, default):
        try:
            return float(val)
        except:
            return default

    Year = safe_float(request.form.get('Year'), 2020)
    rain = safe_float(request.form.get('average_rain_fall_mm_per_year'), 1000)
    pesticides = safe_float(request.form.get('pesticides_tonnes'), 50)
    temp = safe_float(request.form.get('avg_temp'), 25)

    Area = request.form.get('Area', 'India')
    Item = request.form.get('Item', 'Wheat')

    print("INPUT:", Year, rain, pesticides, temp, Area, Item)

    try:
        features = np.array([[Year, rain, pesticides, temp, Area, Item]])

        transformed = preprocessor.transform(features)
        prediction = dtr.predict(transformed)

        return render_template('index.html', prediction=round(prediction[0], 2))

    except Exception as e:
        print("🔥 MODEL ERROR:", e)

        approx = (rain * 0.3) + (temp * 10) - (pesticides * 0.2)
        return render_template('index.html', prediction=f"Approx Result: {round(approx,2)}")
    if __name__ == '__main__':
        print("🔥 Server start ho raha hai...")
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
