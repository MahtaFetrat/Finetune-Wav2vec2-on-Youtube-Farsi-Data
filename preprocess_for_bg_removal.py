import os
import re
import subprocess
import shutil
import numpy as np
import speech_recognition as sr
from pydub import AudioSegment
from jiwer import cer
from datasets import load_from_disk

# Define paths
RAW_DATA_DIR = "/media/external_16TB_1/m_fetrat/speech-project/youtube_data"
OUTPUT_DIR = "/media/external_16TB_1/m_fetrat/speech-project/preprocessing_version/processed_data"

# Load dataset
train = load_from_disk(os.path.join(RAW_DATA_DIR, "train"))
test = load_from_disk(os.path.join(RAW_DATA_DIR, "test"))

# Remove unnecessary columns
train = train.remove_columns(["youtube_url", "title", "segment_id", "video_id"])
test = test.remove_columns(["youtube_url", "title", "segment_id", "video_id"])

# Define regex to clean transcriptions
chars_to_keep = "حخدذرزسشصضطظعغفقلمنهوپچژکی‌"
chars_to_ignore_regex = f'[^{chars_to_keep}\s]'

def remove_special_characters(text):
    return re.sub(chars_to_ignore_regex, '', text) + " "

# Function to remove background music from an audio array
def remove_background_music(audio_array, sampling_rate, log_idx):
    # Convert NumPy array to an audio segment
    audio = AudioSegment(
        np.int16(audio_array * 32767).tobytes(), 
        frame_rate=sampling_rate, 
        sample_width=2, 
        channels=1
    )

    # Save the audio segment to a temporary file
    temp_input_file = f"spleeter/temp_input_{log_idx}.wav"
    audio.export(temp_input_file, format="wav")

    temp_output_folder = f"spleeter/temp_output_{log_idx}"
    os.makedirs(temp_output_folder, exist_ok=True)

    while True:  # Handle potential CUDA out-of-memory issues
        subprocess.run(
            ['python3', '-m', 'spleeter', 'separate', temp_input_file, '-o', temp_output_folder, '-c', 'wav', '-b', '128k'],
        )

        try:
            # Load processed audio (vocals only)
            processed_audio = AudioSegment.from_file(f"{temp_output_folder}/temp_input_{log_idx}/vocals.wav", "wav")

            # Convert to mono if stereo
            if processed_audio.channels == 2:
                processed_audio = processed_audio.set_channels(1)
            
            break
        except FileNotFoundError:
            pass

   # Convert back to NumPy array
    processed_audio_array = np.array(processed_audio.get_array_of_samples(), dtype=np.float32) / 32767.0

    # Cleanup temporary files
    os.remove(temp_input_file)
    shutil.rmtree(temp_output_folder)

    return processed_audio_array


# Function to process audio and filter based on CER
def process_and_save(dataset, split_name):
    split_output_dir = os.path.join(OUTPUT_DIR, split_name)
    os.makedirs(split_output_dir, exist_ok=True)

    for idx, batch in enumerate(dataset):
        # if idx % 8 != 0: continue
        audio_id = f"{split_name}_{idx}"
        audio_path = os.path.join(split_output_dir, f"{audio_id}.wav")
        transcript_path = os.path.join(split_output_dir, f"{audio_id}.txt")

        # Save processed transcript
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(batch["transcription"])

        print(f"Saved {audio_id}")

# Process train and test sets
process_and_save(train, "train")
process_and_save(test, "test")

print("Preprocessing complete. Processed dataset saved.")
