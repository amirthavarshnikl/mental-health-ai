import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# 1. Load trained pipeline
# -------------------------
model = joblib.load("model/risk_model.pkl")

vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["clf"]

# -------------------------
# 2. Load dataset for background
# -------------------------
df = pd.read_csv("data/mental_health.csv")

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

df = df.dropna(subset=["statement"])
df["statement"] = df["statement"].astype(str)

background = df["statement"].sample(100, random_state=42)
background_transformed = vectorizer.transform(background)

# -------------------------
# 3. Create SHAP explainer
# -------------------------
explainer = shap.LinearExplainer(classifier, background_transformed)

# -------------------------
# 4. Test sentence
# -------------------------
test_text = ["I feel hopeless and tired of everything"]
test_transformed = vectorizer.transform(test_text)

# -------------------------
# 5. Get SHAP values
# -------------------------
shap_values = explainer(test_transformed)

# -------------------------
# 6. Get predicted class
# -------------------------
predicted_class = classifier.predict(test_transformed)[0]
class_index = list(classifier.classes_).index(predicted_class)

# -------------------------
# 7. Extract explanation for predicted class only
# -------------------------
single_explanation = shap.Explanation(
    values=shap_values.values[0][:, class_index],
    base_values=shap_values.base_values[0][class_index],
    data=test_transformed.toarray()[0],
    feature_names=vectorizer.get_feature_names_out()
)

# -------------------------
# 8. Plot waterfall
# -------------------------
shap.plots.waterfall(single_explanation, max_display=15)
plt.show()