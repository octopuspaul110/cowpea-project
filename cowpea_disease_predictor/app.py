import pandas as pd
import numpy as np
import joblib
import streamlit as st
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import os
import path
import sys
import sqlite3
from sqlalchemy import create_engine,text
from PIL import Image

# Initialize database connection
DB_FILE = "app_data.db"
engine = create_engine(f"sqlite:///{DB_FILE}")

def initialize_db():
    with engine.connect() as conn:
        conn.execute(text("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            REP INTEGER,
            PH2 REAL,
            PH4 REAL,
            PH6 REAL,
            PH8 REAL,
            NLB2 INTEGER,
            NLB4 INTEGER,
            NLB6 INTEGER,
            NLB8 INTEGER,
            Sample STRING
        );
        """))
# Save data to database
def save_to_db(data):
    with engine.connect() as conn:
        data.to_sql("predictions", con=conn, if_exists="append", index=False)

# Fetch all data from database
def fetch_all_data():
    with engine.connect() as conn:
        query = "SELECT * FROM predictions"
        return pd.read_sql(query, conn)

# Initialize database
initialize_db()
#models_dir = os.path.join(os.path.dirname(__file__),'models')
#model_filepath = os.path.join(models_dir, 'disease_predictor_model.joblib')
#encoder_filepath = os.path.join(models_dir, 'one_hot_encoder.pkl')
                          
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "cowpea_disease", "cowpea-project"))
models_dir = os.path.join(base_dir, "helpers")
model_filepath = os.path.join(models_dir, "disease_predictor_model.joblib")
encoder_filepath = os.path.join(models_dir, "one_hot_encoder.pkl")
print("Model Filepath:", model_filepath)
print("Encoder Filepath:", encoder_filepath)

# File paths for the pre-trained model and encoder
model_filepath = "./cowpea-project/helpers/disease_predictor_model.joblib"
#encoder_filepath = "D:\\cowpea_project\\disease_project\\cowpea_disease\\cowpea-project\\models\\one_hot_encoder.pkl"

dir = path.Path(__file__)
sys.path.append(dir.parent.parent)

# load model
path_to_model = './helpers/disease_predictor_model.joblib'

with open(path_to_model, 'rb') as file:
    model = joblib.load(file)

# Function to preprocess data and make predictions
def preprocess_data(user_data, model_file, encoder_file=None):
    # Convert user data to DataFrame
    user_df = pd.DataFrame(user_data)

    # Feature Engineering
    user_df['PH_change'] = user_df['PH8'] - user_df['PH2']
    user_df['NLB_change'] = user_df['NLB8'] - user_df['NLB2']
    user_df['NLB_avg'] = (user_df['NLB2'] + user_df['NLB4'] + user_df['NLB6'] + user_df['NLB8']) / 4
    user_df['PH_avg'] = (user_df['PH2'] + user_df['PH4'] + user_df['PH6'] + user_df['PH8']) / 4

    # One-Hot Encoding for Sample column
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        # Load pre-fitted encoder
        #encoder = joblib.load(encoder_file)
        encoder.fit(pd.DataFrame({'Sample': ['A', 'B', 'C', 'D', 'E']}))  # Define all possible categories
        #joblib.dump(encoder, encoder_file)
        print("try")
        encoded_samples = encoder.transform(user_df[['Sample']])
        sample_encoded_cols = encoder.get_feature_names_out(['Sample'])
        encoded_sample_df = pd.DataFrame(encoded_samples, columns=sample_encoded_cols, index=user_df.index)
        
    except FileNotFoundError:
        # Fit and save the encoder if not already saved
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        encoder.fit(pd.DataFrame({'Sample': ['A', 'B', 'C', 'D', 'E']}))  # Define all possible categories
        joblib.dump(encoder, encoder_file)
        print("except")
        encoded_samples = encoder.transform(user_df[['Sample']])
        sample_encoded_cols = encoder.get_feature_names_out(['Sample'])
        encoded_sample_df = pd.DataFrame(encoded_samples, columns=sample_encoded_cols, index=user_df.index)

    # Add encoded columns and drop the original 'Sample' column
    user_df = pd.concat([user_df.drop(columns=['Sample']), encoded_sample_df], axis=1)

    # Scaling numeric features
    scaler = StandardScaler()
    numeric_cols = user_df.select_dtypes(include=['int64', 'float64']).columns
    user_df[numeric_cols] = scaler.fit_transform(user_df[numeric_cols])

    # Load Pre-trained Model
    #model = joblib.load(model_file)

    # Perform Inference
    predictions = model.predict(user_df)

    # Return Predictions
    return predictions[0]

# Streamlit App
def main():
    st.title("Cowpea Disease Predictor")
    image_path = "./cowpea_disease_predictor/cowpea_disease.jpeg"  # Replace with the actual path to your image
    try:
        image = Image.open(image_path)
        st.image(image, use_container_width=True)
    except FileNotFoundError:
        st.error("Image file not found. Please ensure the file path is correct.")
        st.write("Enter the values to predict the disease status.")

    # User Input for fields
    Rep = st.number_input("Rep", min_value=1, max_value=100, value=1)
    PH2 = st.number_input("PH2", min_value=0.0, max_value=10000.0, value=5.8)
    PH4 = st.number_input("PH4", min_value=0.0, max_value=10000.0, value=6.0)
    PH6 = st.number_input("PH6", min_value=0.0, max_value=10000.0, value=6.3)
    PH8 = st.number_input("PH8", min_value=0.0, max_value=10000.0, value=6.5)
    NLB2 = st.number_input("NLB2", min_value=0, max_value=100000, value=12)
    NLB4 = st.number_input("NLB4", min_value=0, max_value=100000, value=15)
    NLB6 = st.number_input("NLB6", min_value=0, max_value=100000, value=18)
    NLB8 = st.number_input("NLB8", min_value=0, max_value=100000, value=20)
    Sample = st.selectbox("Sample", options=['A', 'B', 'C', 'D', 'E'])

    # Create user data dictionary
    user_data = {
        'Rep': [Rep],
        'PH2': [PH2],
        'PH4': [PH4],
        'PH6': [PH6],
        'PH8': [PH8],
        'NLB2': [NLB2],
        'NLB4': [NLB4],
        'NLB6': [NLB6],
        'NLB8': [NLB8],
        'Sample': [Sample]
    }

    if st.button("Predict"):
        # Get the prediction
      try:
          prediction = preprocess_data(user_data, model_filepath, encoder_filepath)
          st.write(f"The predicted disease status is: {prediction}")
          user_data["SEEDKGHA"] = prediction
          save_to_db(input_data)
          st.success("Prediction saved to database!")
      except Exception as e:
            st.error(f"Error during inference: {e}")
    # Display stored predictions
    st.header("Stored Predictions")
    stored_data = fetch_all_data()
    if not stored_data.empty:
        st.dataframe(stored_data)
    else:
        st.info("No data stored yet.")      

if __name__ == "__main__":
    main()
