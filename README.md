# Transcription Using AI4Bharat Model

This repository demonstrates how to download audio files from a CSV dataset and transcribe them into regional Hindi using AI4Bharat's ASR (Automatic Speech Recognition) models. The process includes downloading `.wav` files from provided URLs and saving transcriptions back into the dataset.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Files Description](#files-description)

## Overview

The script performs the following steps:
1. **Downloads audio files** from URLs in a CSV file.
2. **Transcribes the audio** using the AI4Bharat IndicConformer model for Hindi.
3. **Saves transcriptions** into a new column called `transcript` in the CSV file.

## Requirements

Make sure you have the following:

- Python 3.x environment (like Kaggle or Colab).
- Required Python libraries:
  - `torch`
  - `nemo.collections.asr`
  - `pandas`
  - `requests`
  - `soundfile`

- AI4Bharat IndicConformer model for Hindi transcription.

## Installation

### Step 1: Clone AI4Bharat NeMo Repository

To get started, clone the AI4Bharat NeMo repository and switch to the correct branch:

```bash
git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo && git checkout nemo-v2 && bash reinstall.sh
```

### Step 2: Install Additional Dependencies
    Next, install any additional dependencies required for the project:
    pip install packaging
    pip install huggingface_hub==0.23.2
    pip install requests

### Step 3: Download the Pre-trained AI4Bharat ASR Model
Download the Hindi IndicConformer ASR model, which will be used for transcription:
      wget "https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_hi.nemo" -O checkpoint.nemo

## Files and Folders Description

1. **Bihar_combined_file.csv**  
   This is the input dataset containing URLs for the corresponding .wav audio files. Each entry in the CSV provides a link to an audio file that will be processed for transcription.

2. **checkpoint.nemo**  
   This file contains the pre-trained AI4Bharat IndicConformer model specifically designed for Hindi automatic speech recognition. It is essential for transcribing the audio data from the provided .wav files.

3. **transcribed_dataset.csv**  
   This is the output file where the transcriptions of the audio files will be stored. Each transcription corresponds to the audio file listed in the input dataset.

4. **Bihar_Dataset_wavFiles/**  
   This folder contains the downloaded .wav audio files corresponding to the URLs provided in the **Bihar_combined_file.csv**. Each audio file is named in accordance with its respective entry in the CSV file.


