import streamlit as st
from streamlit_option_menu import option_menu
from keras.models import load_model
from keras import backend as K
import numpy as np
import pandas as pd
import pygame
pygame.mixer.init()
from sklearn.compose import ColumnTransformer
import time
from sklearn.preprocessing import OneHotEncoder, StandardScaler
ALARM_SOUND = './alerm.mp3'  # Path to your buzzer sound file
st.set_page_config(page_title = 'Attack Detection', page_icon = '🧊', layout = 'wide')
def play_buzzer():
    pygame.mixer.music.load(ALARM_SOUND)
    pygame.mixer.music.play()
    time.sleep(5)  
    pygame.mixer.music.stop()

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))
def sample_data():
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://thumbs.dreamstime.com/b/network-node-background-cyber-security-global-connection-internet-glowing-blue-network-node-background-cyber-security-global-348195338.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding: 50px 0;
        }
        .info-box {
            background-color: rgba(255, 255, 255, 0.8);
            padding: 20px;
            border-radius: 10px;
            box-shadow: 2px 2px 8px rgba(0, 0, 0, 0.2);
            max-width: 900px;
            margin: 0 auto;
            font-family: Arial, sans-serif;
            text-align: center;
        }
        .download-btn {
            padding: 10px 20px;
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            font-size: 16px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    
    # Information box with the dataset details
    st.markdown(
        """
        <div class="info-box">
            <p>The UNSW-NB 15 dataset created using the IXIA PerfectStorm tool, combines real normal activities and synthetic attack behaviors. 
            Captured using Tcpdump, the dataset includes nine types of attacks and uses Argus, Bro-IDS tools and twelve algorithms to generate 49 features. 
            The dataset has 2 million records and is divided into a training and testing set.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Button to download the sample CSV data
    # Provide the file for download
    st.write('')
    col1, col2, col3 = st.columns([4, 3, 4])
    with open("sample.csv", "rb") as f:
        col2.download_button(
            label="Download Sample Data",
            data=f,
            file_name="sample_data.csv",
            mime="text/csv", type='primary',
        )

    
def attack_detection():
    st.markdown(
        """
        <style>
        /* Apply background image to the main content area */
        .main {
            background-image: url('https://img.freepik.com/free-vector/cool-wavy-abstract-gradient_53876-93826.jpg');
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-color: rgba(255, 255, 255, 0.2);
            background-blend-mode: overlay;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    col1,col2,col3=st.columns([1,2,1])
    file=col2.file_uploader("Upload a CSV file", type=["csv"])
    try:
        if file is not None:
            model = load_model('LSTM_model.h5', custom_objects={'f1_m': f1_m, 'precision_m': precision_m, 'recall_m': recall_m})
            df_sample=pd.read_csv(file)
            col2.markdown("<h4 style='color: blue;'>Uploaded Data</h4>", unsafe_allow_html=True)
            col2.write(df_sample)
            df_sample = df_sample.replace("?", np.nan)
            numerical_columns = df_sample.select_dtypes(include=[np.number]).columns
            df_sample[numerical_columns] = df_sample[numerical_columns].fillna(df_sample[numerical_columns].mean())

            # Handle missing values for categorical columns (fill with the most frequent value)
            categorical_columns = df_sample.select_dtypes(exclude=[np.number]).columns
            for col in categorical_columns:
                df_sample[col].fillna(df_sample[col].mode()[0], inplace=True)

            # Check for missing values after handling them

            # Handle categorical features (proto, service, state)
            ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1, 2, 3])], remainder='passthrough')
            X_sample = np.array(ct.fit_transform(df_sample))

            # Check the shape after transformation

            # Scale the numerical features
            sc = StandardScaler()
            X_sample[:, 18:] = sc.fit_transform(X_sample[:, 18:])

            # Check the final shape after scaling

            # Padding if the number of features is less than expected
            expected_features = 56
            if X_sample.shape[1] < expected_features:
                padding = np.zeros((X_sample.shape[0], expected_features - X_sample.shape[1]))
                X_sample = np.hstack((X_sample, padding))  # Add padding columns with zero values
            # Ensure the shape matches the model's input (56 features)
            if X_sample.shape[1] != expected_features:
                raise ValueError(f"Mismatch in feature count: Expected {expected_features}, got {X_sample.shape[1]}")
            # Make predictions
            y_pred_sample = model.predict(X_sample)
            y_pred_sample = np.argmax(y_pred_sample, axis=1)  # Convert predictions to class labels
            #print class labels attack or normal
            class_labels = {0: 'Attack', 1: 'Normal'}
            y_pred_sample = [class_labels[i] for i in y_pred_sample]
            #print output
            col2.markdown("<h4 style='color: red;'>Attack Detection Results</h4>", unsafe_allow_html=True)
            if y_pred_sample[0]=='Attack':
                time.sleep(1)  # Small delay before playing the sound
                play_buzzer()
                col2.error("The uploaded data contains an threat.")
                #prevention measures
                col2.markdown("<h4 style='color: blue;'>Prevention Measures</h4>", unsafe_allow_html=True)
                col1,col2=col2.columns(2)
                col1.success("1. Use a firewall to block unauthorized access to your network.")
                col2.success("2. Use a VPN to encrypt your internet connection.")
                col1.success("3. Keep your software up to date to prevent vulnerabilities.")
                col2.success("4. Use strong passwords and two-factor authentication to secure your accounts.")
                col1.success("5. Be cautious of phishing emails and other social engineering attacks.")
                col2.success("6. Regularly monitor your network traffic for unusual activity.")
            else:
                col2.success("The uploaded data is normal.")
    except:
        col2.error("Please upload a correct CSV file in sample format to detect attacks.")

        

# Navigation menu for user dashboard
st.markdown(
    """
    <div style="text-align: center; padding: 1px; background-color: #fa6e9a; border-radius: 40px; border: 1.5px solid black; margin-bottom: 20px;">
        <p style="color: black; font-size: 30px;"><b>ML-Powered Firewall for Adaptive Threat Detection and Real-Time Attack Prevention</b></p>
    </div>
    """,
    unsafe_allow_html=True
)

selected = option_menu(
    menu_title=None,
    options=["Sample Data","Attack Detection"],
    icons=['database-fill','paperclip'], menu_icon="cast", default_index=0,
    orientation="horizontal",
styles={
        "nav-link-selected": {
            "background-color": "#ffc11c",  # Background color of the selected item
            "color": "black",
        },
        "nav-link": {
            "background-color": "#ffefc4",  # Background color of unselected items
            "color": "black",  # Text color of unselected items
        },
    },
)
if selected == "Sample Data":
    sample_data()
elif selected == "Attack Detection":
    attack_detection()