# Bhashika : Dialect Based Text-to-Speech Model for Indic Languages

This report outlines two distinct processes aimed at enhancing the accessibility and usability of regional Hindi language content: the transcription of Hindi audio files using AI4Bharat's Automatic Speech Recognition (ASR) models and the generation of speech in a regional accent utilising the Toucan Text-to-Speech (TTS) model.

## Table of Contents

1. [Overview](#overview)
2. [Requirements](#requirements)
3. [Installation](#installation)
4. [Files Description](#files-description)
5. [Text-to-Speech Implementation](#text-to-speech-implementation)
6. [Using the Code](#using-the-code)

## Overview

The project consists of two main functionalities:

1. **Transcribes audio files** from URLs in a CSV file using the AI4Bharat IndicConformer model for Hindi.
2. **Generates speech audio** from text using the Toucan TTS model.

### Audio Transcription Process
- Downloads `.wav` files from URLs in the provided CSV dataset.
- Transcribes the audio using the AI4Bharat ASR model.
- Saves transcriptions into a new column called `transcript` in the CSV file.

### Text-to-Speech Generation
- Uses the Toucan TTS model to convert given text into speech, producing an audio file.

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
- Toucan TTS interface for text-to-speech generation.

## Installation

### Step 1: Clone AI4Bharat NeMo Repository

To get started, clone the AI4Bharat NeMo repository and switch to the correct branch:

```bash
git clone https://github.com/AI4Bharat/NeMo.git && cd NeMo && git checkout nemo-v2 && bash reinstall.sh

```
### Step 2: Install Additional Dependencies
Next, install any additional dependencies required for the project
pip install packaging
pip install huggingface_hub==0.23.2
pip install requests

### Step 3: Download the Pre-trained AI4Bharat ASR Model
Download the Hindi IndicConformer ASR model, which will be used for transcription:
wget "https://objectstore.e2enetworks.net/indic-asr-public/indicConformer/ai4b_indicConformer_hi.nemo" -O checkpoint.nemo

### Files Description
Bihar_combined_file.csv
This is the input dataset containing URLs for the corresponding .wav audio files. Each entry in the CSV provides a link to an audio file that will be processed for transcription.

checkpoint.nemo
This file contains the pre-trained AI4Bharat IndicConformer model specifically designed for Hindi automatic speech recognition. It is essential for transcribing the audio data from the provided .wav files.

transcribed_dataset.csv
This is the output file where the transcriptions of the audio files will be stored. Each transcription corresponds to the audio file listed in the input dataset.

Bihar_Dataset_wavFiles/
This folder contains the downloaded .wav audio files corresponding to the URLs provided in the Bihar_combined_file.csv. Each audio file is named in accordance with its respective entry in the CSV file.

audios/
This folder will be created to store generated audio files from the text-to-speech conversion.

### Text-to-Speech Implementation
The provided code uses the os library to manage directories and file paths. Here’s a brief overview:

os: This library provides a way of using operating system-dependent functionality like reading or writing to the file system. In the code, os.makedirs is used to create a directory if it does not already exist.

#### Code Functionality
read_texts: This function initializes the TTS model and reads a given text sentence, saving the output to a specified file.
dialect_generation: This function creates the output directory and calls read_texts to generate an audio file of the specified dialect.
The main section determines whether to use a GPU or CPU, prepares the speaker references, and calls dialect_generation.

### Using the Code
Clone the repository and set up the environment as described.
Prepare your input CSV file with audio URLs.
Run the transcription script to generate the transcriptions.
Use the provided TTS script to generate audio files from text.

## References:
### AI4Bharat - IndicTTS [[code link]](https://github.com/AI4Bharat/Indic-TTS)

### IMS-Toucan [[associated code and models]](https://github.com/DigitalPhonetics/IMS-Toucan)
```
@inproceedings{lux2021toucan,
  year         = 2021,
  title        = {{The IMS Toucan system for the Blizzard Challenge 2021}},
  author       = {Florian Lux and Julia Koch and Antje Schweitzer and Ngoc Thang Vu},
  booktitle    = {Blizzard Challenge Workshop},
  publisher    = {ISCA Speech Synthesis SIG}
}
```


