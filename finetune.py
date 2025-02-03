

from datasets import load_from_disk
train = load_from_disk('/media/external_16TB_1/m_fetrat/speech-project/youtube_data/train')
test = load_from_disk('/media/external_16TB_1/m_fetrat/speech-project/youtube_data/test')



train = train.remove_columns(["youtube_url", "title", "segment_id", "video_id"])
test = test.remove_columns(["youtube_url", "title", "segment_id", "video_id"])


from datasets import ClassLabel
import random
import pandas as pd
import re

# Define the regex to match any character not in the allowed set
chars_to_keep = "حابخدذرزسشصضطظعغفقلئآمتثجنهوپچژکی‌"
chars_to_ignore_regex = f'[^{chars_to_keep}\s]'

def remove_special_characters(batch):
    batch["transcription"] = re.sub(chars_to_ignore_regex, '', batch["transcription"]).lower() + " "
    return batch



train = train.map(remove_special_characters)
test = test.map(remove_special_characters)


def extract_all_chars(batch):
  all_text = " ".join(batch["transcription"])
  vocab = list(set(all_text))
  return {"vocab": [vocab], "all_text": [all_text]}


vocab_train = train.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=train.column_names)
vocab_test = test.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True, remove_columns=test.column_names)


vocab_list = list(set(vocab_train["vocab"][0]) | set(vocab_test["vocab"][0]))


vocab_dict = {v: k for k, v in enumerate(vocab_list)}
vocab_dict



vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]


vocab_dict["<unk>"] = len(vocab_dict)
vocab_dict["<pad>"] = len(vocab_dict)
len(vocab_dict)


import json


from transformers import Wav2Vec2CTCTokenizer

tokenizer = Wav2Vec2CTCTokenizer("./vocab.json", unk_token="<unk>", pad_token="<pad>", word_delimiter_token="|")


from transformers import Wav2Vec2FeatureExtractor

feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=True)


from transformers import Wav2Vec2Processor

processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)


from datasets import Audio

train = train.cast_column("audio", Audio(sampling_rate=16_000))
test = test.cast_column("audio", Audio(sampling_rate=16_000))


# import IPython.display as ipd
import numpy as np
import random


def prepare_dataset(batch):
    audio = batch["audio"]

    # batched output is "un-batched"
    batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]

    with processor.as_target_processor():
        batch["labels"] = processor(batch["transcription"]).input_ids
    return batch

train = train.map(prepare_dataset, remove_columns=train.column_names, num_proc=4)
test = test.map(prepare_dataset, remove_columns=test.column_names, num_proc=4)


import torch

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

@dataclass
class DataCollatorCTCWithPadding:

    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    max_length: Optional[int] = None
    max_length_labels: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    pad_to_multiple_of_labels: Optional[int] = None

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                max_length=self.max_length_labels,
                pad_to_multiple_of=self.pad_to_multiple_of_labels,
                return_tensors="pt",
            )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels

        return batch


data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)


# from datasets import load_metric
# wer_metric = load_metric("wer")
import evaluate
wer_metric = evaluate.load("wer")


def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)

    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

    pred_str = processor.batch_decode(pred_ids)
    # we do not want to group tokens when computing the metrics
    label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

    wer = wer_metric.compute(predictions=pred_str, references=label_str)

    return {"wer": wer}


from transformers import Wav2Vec2ForCTC
import torch
from torch.utils.data import DataLoader
from transformers import AdamW
from tqdm import tqdm
import numpy as np
import os
import re


# Create DataLoader for training and evaluation
train_dataloader = DataLoader(train, batch_size=16, collate_fn=data_collator, shuffle=True)
eval_dataloader = DataLoader(test, batch_size=16, collate_fn=data_collator)


# Path to checkpoint directory
checkpoint_dir = "/media/external_16TB_1/m_fetrat/speech-project/finetuned/checkpoint-25-200"

# Determine the starting epoch
if os.path.exists(checkpoint_dir):
    # Extract the epoch and step from the checkpoint name (e.g., checkpoint-4-1000)
    match = re.search(r"checkpoint-(\d+)-\d+", os.path.basename(checkpoint_dir))
    if match:
        start_epoch = int(match.group(1))  # Extract the epoch number
        print(f"Resuming training from checkpoint: {checkpoint_dir}, starting from epoch {start_epoch}.")
    else:
        raise ValueError(f"Invalid checkpoint format: {checkpoint_dir}")
    model = Wav2Vec2ForCTC.from_pretrained(checkpoint_dir)
else:
    print("No checkpoint found. Initializing new model.")
    start_epoch = 0
    model = Wav2Vec2ForCTC.from_pretrained(
        "/media/external_16TB_1/m_fetrat/speech-project/models/snapshots/4129748ad6295d2f73c155a5d0509a46f5e42f28",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )

# Ensure model is on the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Freeze feature extractor and enable gradient checkpointing
model.freeze_feature_extractor()
model.gradient_checkpointing_enable()

# Define optimizer
optimizer = AdamW(model.parameters(), lr=3e-4)

# Load optimizer state if resuming
if os.path.exists(checkpoint_dir):
    optimizer_state_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if os.path.exists(optimizer_state_path):
        optimizer.load_state_dict(torch.load(optimizer_state_path))
        print("Optimizer state loaded.")

# Training loop parameters
epochs = 30
eval_steps = 100
save_steps = 100
logging_steps = 10
output_dir = "./finetuned"

# Start training from the correct epoch
for epoch in range(start_epoch, epochs):
    model.train()
    running_loss = 0.0
    progress_bar = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")

    for step, batch in progress_bar:
        # Move batch to device
        input_values = batch["input_values"].to(device)
        labels = batch["labels"].to(device)

        # Forward pass
        outputs = model(input_values, labels=labels)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track loss
        running_loss += loss.item()
        if step % logging_steps == 0 and step > 0:
            progress_bar.set_postfix({"loss": running_loss / logging_steps})
            running_loss = 0.0

        # Save model periodically
        if step % save_steps == 0 and step > 0:
            checkpoint_path = f"{output_dir}/checkpoint-{epoch+1}-{step}"
            model.save_pretrained(checkpoint_path)
            torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

    # Evaluation loop
    model.eval()
    wer_scores = []
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(input_values, labels=labels)
            logits = outputs.logits

            # Decode predictions and labels
            pred_ids = torch.argmax(logits, dim=-1)
            pred_str = processor.batch_decode(pred_ids.cpu().numpy())
            label_str = processor.batch_decode(labels.cpu().numpy(), group_tokens=False)

            # Compute WER
            wer = wer_metric.compute(predictions=pred_str, references=label_str)
            wer_scores.append(wer)

    avg_wer = np.mean(wer_scores)
    print(f"Epoch {epoch+1} - Average WER: {avg_wer:.4f}")

    # Save model at the end of each epoch
    model.save_pretrained(f"{output_dir}/epoch-{epoch+1}")

