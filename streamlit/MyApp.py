import streamlit as st
from PyPDF2 import PdfReader
import time
import matplotlib.pyplot as plt
import numpy as np
from upgrade_function import augmenter_dataframe
import pandas as pd
import joblib
from transformers import CamembertForSequenceClassification, CamembertTokenizer
from matplotlib.patches import Ellipse

st.title("Detecting the difficulty level of French texts")

with st.sidebar:
    uploaded_file = st.file_uploader("Choose a PDF or text file", type=["pdf", "txt"])

def map_level_to_progress(level):
    level_scale = {'A1': 10, 'A2': 20, 'B1': 40, 'B2': 60, 'C1': 80, 'C2': 100}
    return level_scale.get(level, 0)

def get_color(level):
    color_mapping = {'A1': 'red', 'A2': 'orange', 'B1': 'yellow', 'B2': 'green', 'C1': 'blue', 'C2': 'purple'}
    return color_mapping.get(level, 'gray')

if uploaded_file is not None:
    st.header("Analysis in progress...")
    with st.spinner('Treatment in progress...'):
        file_content = ""
        # Lecture du fichier
        if uploaded_file.type == "application/pdf":
            pdf_reader = PdfReader(uploaded_file)
            for page_number in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_number]
                file_content += page.extract_text()
        elif uploaded_file.type == "text/plain":
            file_content = str(uploaded_file.read(), 'utf-8')

        # Displaying file content
        st.header("File content :")
        st.write(file_content)

        # Traitements...
        df = pd.DataFrame({'sentence': [file_content]})
        df_augmente = augmenter_dataframe(df)

        # Load tokenizer from local path
        tokenizer_path = "/Users/phil/Documents/GitHub/FrenchTextProjet/streamlit/tokenizer"
        loaded_tokenizer = CamembertTokenizer.from_pretrained(tokenizer_path)

        # Load model from local path
        model_path = "/Users/phil/Documents/GitHub/FrenchTextProjet/streamlit/modele_camembert"
        loaded_model = CamembertForSequenceClassification.from_pretrained(model_path)

        # Load SVM model from local path
        svm_model_path = "/Users/phil/Documents/GitHub/FrenchTextProjet/streamlit/svm_model.pkl"
        loaded_svm_model = joblib.load(svm_model_path)

        sentences = df_augmente['sentence'].tolist()

        all_features2 = []
        # Loop over data with a progress bar
        for sentence in sentences:
            # Tokenisation et extraction
            inputs = loaded_tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)
            outputs = loaded_model(**inputs)
            features = outputs.logits.squeeze().detach().numpy()
            
            all_features2.append(features)
        

        test_data_x = df_augmente.drop(columns=["sentence",])
        combined_features = np.concatenate((all_features2, test_data_x), axis=1)
        predicted_labels = loaded_svm_model.predict(combined_features)

        difficulty_mapping = {
        0: 'A1',
        1: 'A2',
        2: 'B1',
        3: 'B2',
        4: 'C1',
        5: 'C2'
        }

        map_function = np.vectorize(lambda x: difficulty_mapping[x])
        predicted_labels = map_function(predicted_labels)  

        st.header("Level of language :")

        level_values = [map_level_to_progress(level) for level in predicted_labels]

        # Display circles with values inside
        fig, ax = plt.subplots(figsize=(4, 4))

        for i, (label, value) in enumerate(zip(predicted_labels, level_values)):
            color = get_color(label)
            circle = Ellipse((0.5, 0.5), width=0.03, height=0.03, color=color, alpha=0.7)
            ax.add_patch(circle)

            ax.text(0.5, 0.5, label, ha='center', va='center', fontsize=14, color='black', weight='bold')

        ax.axis('equal')
        ax.axis('off')

        st.pyplot(fig)

        # End of analysis message
        st.success('Analysis complete!')

# help or documentation section
with st.expander("Help and Documentation"):
    st.write("""
        Welcome to the French text level assessment tool. 
        Upload a PDF or text file, and the application will evaluate its language level.
    """)
