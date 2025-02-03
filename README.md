# Finetune-Wav2vec2-on-Youtube-Farsi-Data

This repository contains the code and resources for fine-tuning the `wav2vec2` model for Automatic Speech Recognition (ASR) in Persian (Farsi). The model is fine-tuned on a chunked dataset of Persian audio from YouTube, and it achieves a Word Error Rate (WER) of **26%** on the test split.

## Model Details

- **Base Model**: [masoudmzb/wav2vec2-xlsr-multilingual-53-fa](https://huggingface.co/masoudmzb/wav2vec2-xlsr-multilingual-53-fa)
- **Fine-Tuned Model**: [Fine-tuned model weights]([INSERT_YOUR_MODEL_LINK_HERE](https://huggingface.co/MahtaFetrat/wav2vec2_finetuned_on_youtube_farsi_30))
- **Dataset**: [pourmand1376/asr-farsi-youtube-chunked-30-seconds](https://huggingface.co/datasets/pourmand1376/asr-farsi-youtube-chunked-30-seconds)
- **WER**: 26% (on the test split)

## Hugging Face Space

You can test the fine-tuned model directly in your browser using the Hugging Face Space:
[Link to Hugging Face Space](https://huggingface.co/spaces/baharbhz/persian_SR)

## Repository Structure

```
.
├── evaluate_model.ipynb          # Notebook for evaluating the model on test data
├── finetune.py                   # Script for fine-tuning the wav2vec2 model
├── preprocess_for_bg_removal.py  # Preprocessing script to remove background music using Spleeter
├── preprocess_for_getting_transcripts.py  # Preprocessing script to transcribe audio and calculate CER
├── create_clean_dataset.py       # Script to filter out chunks with high CER and create a clean dataset
├── inference.py                  # Script for inference (transcribing a given audio file) using the finetuned model
├── README.md                     # This file
```
---
## Team Members:
- Bahar Behzadipour
- Mahta Fetrat
---
