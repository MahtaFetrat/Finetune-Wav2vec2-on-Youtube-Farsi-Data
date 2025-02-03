import os
import csv
import pandas as pd
from vosk import Model as VoskModel
from vosk import KaldiRecognizer, SetLogLevel
from jiwer import cer
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import torch
import librosa
import torchaudio
import numpy as np
import speech_recognition as sr

# Function to transcribe audio to text using Google's Speech Recognition
def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        # Using Google's speech recognition API
        transcript = recognizer.recognize_google(audio, language="fa-IR")
        return transcript
    except Exception as e:
        print(f"Error transcribing {audio_path}: {e}")
        return None

# Function to calculate CER (Character Error Rate)
def calculate_cer(reference, hypothesis):
    return cer(reference, hypothesis)

# Function to process audio files and write to CSV
def process_audio_files(input_folder, output_csv):
    # Load existing results if any
    if os.path.exists(output_csv):
        processed_files = pd.read_csv(output_csv)
        processed_files_set = set(processed_files['filename'])
    else:
        processed_files_set = set()

    # List all wav files in the folder
    audio_files = [f for f in os.listdir(input_folder) if f.endswith('.wav')]
    processed_count = 0

    # Open CSV in append mode
    with open(output_csv, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        
        # If the CSV is empty, write headers
        if os.stat(output_csv).st_size == 0:
            writer.writerow(["filename", "google_transcript", "cer"])

        # Process each audio file
        for audio_file in audio_files:
            # Extract the numeric part from the filename, assuming it's in the format "train_[NUM].wav"
            if audio_file.startswith("train_") and audio_file.endswith(".wav"):
                try:
                    num = int(audio_file[6:-4])  # Extract the number between "train_" and ".wav"
                except ValueError:
                    print(f"Skipping invalid file name format: {audio_file}")
                    continue

                # # Only process files where num % 8 == 1
                if num % 8 != 0:
                    continue

                if audio_file in processed_files_set:
                    continue  # Skip already processed files

                audio_path = os.path.join(input_folder, audio_file)
                txt_file = audio_file.replace(".wav", ".txt")
                txt_path = os.path.join(input_folder, txt_file)

                # Read reference transcript from corresponding .txt file
                if os.path.exists(txt_path):
                    with open(txt_path, 'r') as file:
                        reference = file.read().strip()
                else:
                    print(f"Reference transcript for {audio_file} not found.")
                    continue

                # Transcribe the audio file
                transcript = transcribe_audio(audio_path)
                if transcript is None:
                    continue

                # Calculate CER (Character Error Rate)
                cer = calculate_cer(reference, transcript)

                # Write results to CSV
                writer.writerow([audio_file, transcript, cer])

                # Update processed count and status
                processed_count += 1
                print(f"Processed {processed_count}/{len(audio_files)} files")

# Define folder path and output CSV
input_folder = "/media/external_16TB_1/m_fetrat/speech-project/preprocessing_version/processed_data/train"  # Replace with your folder path
output_csv = "output_transcription_results_0_.csv"  # Path to your output CSV file

# Process the files
process_audio_files(input_folder, output_csv)
