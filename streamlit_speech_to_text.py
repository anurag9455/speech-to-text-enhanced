
import streamlit as st
import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import numpy as np
import soundfile as sf

# Load Wav2Vec2 model and processor
# processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
# model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")


processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Function for noise reduction using harmonic-percussive separation
def reduce_noise(audio_data, sr):
    harmonic, _ = librosa.effects.hpss(audio_data)
    return harmonic, sr

# Transcribe audio using Wav2Vec2
def transcribe_audio(audio_data, sr):
    # Resample if needed
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000
    
    # Convert audio to input values for Wav2Vec2
    input_values = processor(audio_data, sampling_rate=sr, return_tensors="pt").input_values
    
    # Perform transcription
    with torch.no_grad():
        logits = model(input_values).logits
    
    # Get predicted tokens
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # Decode transcription
    transcription = processor.batch_decode(predicted_ids)[0]
    
    # Calculate confidence score
    confidence = torch.max(torch.softmax(logits, dim=-1), dim=-1).values.mean().item()
    
    return transcription, confidence

# Streamlit app layout
st.title("Speech-to-Text with Ghost Reduction")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["wav"])

if audio_file is not None:
    st.audio(audio_file)
    
    # Load audio file using librosa
    audio_data, sr = librosa.load(audio_file, sr=None)
    
    # Reduce noise
    audio_data, sr = reduce_noise(audio_data, sr)
    
    # Transcribe audio and get confidence score
    transcription, confidence = transcribe_audio(audio_data, sr)
    
    # Display transcription and confidence score
    st.write("**Transcription:**", transcription)
    st.write("**Confidence Score:**", confidence)
