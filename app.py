import streamlit as st
import openai
import pandas as pd
import json

# Load disease descriptions
with open("diseases.json", "r") as f:
    disease_data = json.load(f)

# Set API key
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
model = genai.GenerativeModel("models/gemini-1.5-flash-lastest")

st.title("🧠 AI Medical Agent")
st.write("Upload symptoms as text or CSV file and get disease predictions.")

# Upload symptoms
uploaded_file = st.file_uploader("Upload .txt or .csv file", type=["txt", "csv"])

def extract_symptoms(file):
    if file.name.endswith(".txt"):
        return file.read().decode("utf-8")
    elif file.name.endswith(".csv"):
        df = pd.read_csv(file)
        return " ".join(df.astype(str).apply(lambda x: ' '.join(x), axis=1))
    return ""

if uploaded_file:
    symptoms = extract_symptoms(uploaded_file)

    if symptoms:
        with st.spinner("Analyzing..."):
            prompt = f"Given the following symptoms: {symptoms}, which disease does the patient most likely have?"
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": prompt}]
            )
            disease = response['choices'][0]['message']['content'].strip()

            st.subheader("🩺 Likely Disease:")
            st.write(disease)

            # Find description
            disease_clean = disease.lower().split()[0]
            description = disease_data.get(disease_clean, "No description found.")
            st.subheader("📄 Description:")
            st.write(description)
