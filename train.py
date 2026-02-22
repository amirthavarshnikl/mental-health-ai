import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib

# ----------------------------
# 1. Load dataset
# ----------------------------
df = pd.read_csv("data/mental_health.csv")  # Make sure filename matches

# ----------------------------
# 2. Clean dataset
# ----------------------------

# Remove unnecessary index column if it exists
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Remove rows where text is missing
df = df.dropna(subset=["statement"])

# Convert text column to string (extra safety)
df["statement"] = df["statement"].astype(str)

# Remove rows where label is missing (just in case)
df = df.dropna(subset=["status"])

# ----------------------------
# 3. Define input and output
# ----------------------------
X = df["statement"]
y = df["status"]

# ----------------------------
# 4. Split data
# ----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# 5. Create ML pipeline
# ----------------------------
model = Pipeline([
    ("tfidf", TfidfVectorizer(stop_words="english",ngram_range = (1,2))),
    ("clf", LogisticRegression(max_iter=1000))
])

# ----------------------------
# 6. Train model
# ----------------------------
model.fit(X_train, y_train)

# ----------------------------
# 7. Evaluate model
# ----------------------------
predictions = model.predict(X_test)
print("\nModel Evaluation:\n")
print(classification_report(y_test, predictions))

# ----------------------------
# 8. Save trained model
# ----------------------------
joblib.dump(model, "model/risk_model.pkl")

print("\nModel trained and saved successfully!")