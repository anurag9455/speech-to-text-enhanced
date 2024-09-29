
import streamlit as st
import librosa
import torch
import noisereduce as nr
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, GPT2LMHeadModel, GPT2Tokenizer

# Load Wav2Vec2 model and processor for acoustic model transcription
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-large-960h")

# Load GPT-2 model and tokenizer for language model refinement
gpt_model = GPT2LMHeadModel.from_pretrained("gpt2")
gpt_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Function for noise reduction
def reduce_noise(audio_data, sr):
    return nr.reduce_noise(y=audio_data, sr=sr)

# Transcribe audio using Wav2Vec2
import numpy as np

def sanitize_audio(audio_data):
    # Ensure the audio data contains only finite values
    audio_data = np.nan_to_num(audio_data)  # Replace NaN and infinity with zeros
    return audio_data

# Transcribe audio using Wav2Vec2
def transcribe_audio(audio_data, sr):
    # Sanitize the audio data
    audio_data = sanitize_audio(audio_data)
    
    # Resample to 16kHz for Wav2Vec2 if needed
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
    
    return transcription, confidence, logits


# Language model refinement using GPT-2
def refine_transcription(transcription):
    inputs = gpt_tokenizer.encode(transcription, return_tensors="pt")
    outputs = gpt_model.generate(inputs, max_length=500)
    refined_transcription = gpt_tokenizer.decode(outputs[0], skip_special_tokens=True)
    return refined_transcription

# Confidence-based filtering
def filter_low_confidence(transcription, confidence, threshold=0.6):
    if confidence >= threshold:
        return transcription
    else:
        return "[Low confidence transcription removed]"

# Streamlit app layout
st.title("Enhanced Speech-to-Text App")

# Upload audio file
audio_file = st.file_uploader("Upload an audio file", type=["wav"])

if audio_file is not None:
    # Load and display the audio file
    audio_data, sr = librosa.load(audio_file, sr=None)
    st.audio(audio_file)

    # Apply noise reduction
    st.write("Applying noise reduction...")
    reduced_audio = reduce_noise(audio_data, sr)

    # Perform transcription with confidence scoring
    st.write("Performing transcription...")
    transcription, confidence, logits = transcribe_audio(reduced_audio, sr)
    
    # Filter out low-confidence parts
    filtered_transcription = filter_low_confidence(transcription, confidence)
    
    # Refine the transcription using GPT-2
    st.write("Refining transcription using language model...")
    refined_transcription = refine_transcription(filtered_transcription)
    
    # Display final transcription and confidence score
    st.write("**Final Transcription:**", refined_transcription)
    st.write("**Confidence Score:**", confidence)
