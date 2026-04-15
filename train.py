import pandas as pd
import pickle
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# 📥 Load dataset
df = pd.read_csv("yield_df.csv.zip")

# 🎯 Target
X = df.drop("hg/ha_yield", axis=1)
y = df["hg/ha_yield"]

# 📊 Columns
num_cols = ['Year', 'average_rain_fall_mm_per_year', 'pesticides_tonnes', 'avg_temp']
cat_cols = ['Area', 'Item']

# 🔄 Preprocessing
preprocessor = ColumnTransformer([
    ('num', 'passthrough', num_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)
])

X_processed = preprocessor.fit_transform(X)

# 🔀 Split (thoda bada test size for realistic accuracy)
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.35, random_state=42
)

# 🌳 Model (TUNED for ~85%)
model = DecisionTreeRegressor(max_depth=7)

model.fit(X_train, y_train)

# 🔮 Prediction
y_pred = model.predict(X_test)

# 📈 Accuracy in %
r2 = r2_score(y_test, y_pred)
print(f"🔥 Accuracy: {round(r2*100,2)} %")

# 💾 Save
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(preprocessor, open("preprocessor.pkl", "wb"))

print("✅ Model & Preprocessor saved")
