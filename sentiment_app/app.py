import streamlit as st
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
import shap
import torch

@st.cache_resource
def load_pipeline():
    model = AutoModelForSequenceClassification.from_pretrained("model")
    tokenizer = AutoTokenizer.from_pretrained("model")
    pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, return_all_scores=True)
    return pipe, tokenizer, model

pipe, tokenizer, model = load_pipeline()

st.title("✈️ Airline Tweet Sentiment Analyzer (BERT + SHAP)")
user_input = st.text_area("✍️ Enter a tweet:", value="The flight was delayed and the staff was rude.")

if st.button("Predict"):
    prediction = pipe(user_input)
    pred_label = torch.argmax(torch.tensor([p["score"] for p in prediction[0]]))
    label_map = {0: "Negative", 1: "Neutral", 2: "Positive"}

    st.markdown(f"### 🧠 Sentiment: **{label_map[pred_label.item()]}**")
    st.json(prediction[0])

    st.markdown("---")
    st.subheader("🔍 SHAP Explainability")
    explainer = shap.Explainer(pipe)
    shap_values = explainer([user_input])
    st.set_option('deprecation.showPyplotGlobalUse', False)
    shap.plots.text(shap_values[0], display=False)
    st.pyplot(bbox_inches='tight')
