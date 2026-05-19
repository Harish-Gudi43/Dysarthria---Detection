import streamlit as st
import numpy as np
import librosa
import joblib
from tensorflow.keras.layers import Input, Conv1D, MaxPooling1D, BatchNormalization, Add, Activation, Reshape, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.models import Model, Sequential
import base64

# Define the CNN-1D base model
def cnn_base_model(input_shape):
    inputs = Input(shape=input_shape)
    
    # Convolutional layers
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=2)(x)
    x = BatchNormalization()(x)

    residual = Conv1D(128, kernel_size=1, padding='same')(x)
    x = Add()([x, residual])
    x = Activation('relu')(x)

    return Model(inputs, x)

# Define the full model
def dys_cnn_att_bilstm_model(input_shape):
    model = Sequential()
    cnn_model = cnn_base_model(input_shape)
    model.add(cnn_model)

    # Get the output shape from the CNN
    cnn_output_shape = cnn_model.output_shape[1:]  # Ignore batch size
    time_steps = cnn_output_shape[0]  # Number of time steps
    features = cnn_output_shape[1]  # Number of features
    
    model.add(Reshape((time_steps, features)))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    
    model.add(Bidirectional(LSTM(64)))
    model.add(Dropout(0.5))
    
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Load model architecture
input_shape = (13, 1)  
best_model = dys_cnn_att_bilstm_model(input_shape)

# Load weights 
try:
    best_model.load_weights('best_model.weights.h5')
except Exception as e:
    print(f"Error loading weights: {e}")

# Load  image function
def load_image(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

# Load  image
image_base64 = load_image("bg.webp")  

# Set background image 
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{image_base64});
        background-size: cover;
        background-position: center;
        color: white;    /* set text to white */
    }}
    h1, h2, h3, h4, h5, h6, .stFileUploader {{
        color: white;  /* set text to white */
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("Dysarthria Detection from Audio")

# Add a summary
st.write("""
This web application allows users to upload audio files for analysis to detect the presence of dysarthria, a motor speech disorder resulting from neurological conditions. 
Utilising advanced machine learning techniques, the app processes the audio input to extract relevant acoustic features and predict whether the speaker exhibits signs of dysarthria or not.
""")

# Add title to upload 
st.markdown("<h5 style='color:white;'>Upload an audio file</h5>", unsafe_allow_html=True)

# Upload file
uploaded_file = st.file_uploader("", type=["wav"])  

if uploaded_file is not None:
    # Save the uploaded file temporarily
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Preprocess the audio
    audio_data, sample_rate = librosa.load("temp.wav", sr=None)

    # Normalise and trim silence
    audio_data = audio_data / np.max(np.abs(audio_data))
    audio_data, _ = librosa.effects.trim(audio_data)

    # Resampling to 22.05 kHz
    target_sr = 22050
    audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=target_sr)

    # Extract MFCCs
    mfccs = librosa.feature.mfcc(y=audio_data, sr=target_sr, n_mfcc=13)

    # Scale the MFCCs
    scaler = joblib.load('scaler.pkl')
    mfccs_scaled = scaler.transform(mfccs.T)  # Scale the MFCCs

    # Ensure consistent number of time steps 
    if mfccs_scaled.shape[0] > 1:
        mfccs_scaled = mfccs_scaled[0:1, :]  # Keep only the first row

    # Reshape to (1, 13, 1)
    mfccs_scaled = mfccs_scaled.reshape(1, 13, 1)  # Reshape to (1, 13, 1)

    # Predict the class
    prediction = best_model.predict(mfccs_scaled)
    predicted_class = 'Dysarthria' if prediction[0][0] > 0.5 else 'Non-Dysarthria'

    # Display result
    st.subheader(f"Prediction: {predicted_class}")