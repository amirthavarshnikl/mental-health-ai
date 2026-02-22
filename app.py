import streamlit as st
import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Mental Health Risk Detector", layout="wide")

st.title("🧠 Mental Health Risk Prediction System")
st.write("Enter a statement below to analyze mental health risk and view explanation.")

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/risk_model.pkl")

model = load_model()
vectorizer = model.named_steps["tfidf"]
classifier = model.named_steps["clf"]

# -----------------------------
# User Input
# -----------------------------
user_input = st.text_area("Enter text here:")

if st.button("Analyze"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        # Transform input
        transformed_input = vectorizer.transform([user_input])

        # Prediction
        prediction = classifier.predict(transformed_input)[0]
        probabilities = classifier.predict_proba(transformed_input)[0]

        st.subheader("🔎 Prediction Result")
        st.success(f"Predicted Category: **{prediction}**")

        # Show probability table
        prob_df = pd.DataFrame({
            "Class": classifier.classes_,
            "Probability": probabilities
        }).sort_values(by="Probability", ascending=False)

        st.dataframe(prob_df)

        # -----------------------------
        # SHAP Explanation
        # -----------------------------
        st.subheader("📊 Explainable AI (SHAP Analysis)")

        # Background sample
        background = vectorizer.transform(["sample text"])

        explainer = shap.LinearExplainer(classifier, background)
        shap_values = explainer(transformed_input)

        # Get predicted class index
        class_index = list(classifier.classes_).index(prediction)

        # Create explanation object
        explanation = shap.Explanation(
            values=shap_values.values[0][:, class_index],
            base_values=shap_values.base_values[0][class_index],
            data=transformed_input.toarray()[0],
            feature_names=vectorizer.get_feature_names_out()
        )

        fig = plt.figure()
        shap.plots.waterfall(explanation, max_display=15, show=False)
        st.pyplot(fig)