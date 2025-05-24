# Automated_Emotion_Analysis-_A_Telegram_Bot_powered_by_Speech_and_Image_Emotion_Recognition


This is a real-time emotion recognition system that detects user emotions from **speech** and **facial images**. It uses Convolutional Neural Networks (CNNs) for accurate emotion classification and responds through a Telegram bot.

---

## Features

* **Speech Emotion Recognition**: Detects emotions from audio input.
* **Facial Emotion Recognition**: Classifies emotions from facial images.
* **Telegram Bot Interface**: Users can interact by sending voice or image messages.
* **Real-Time Detection**: Processes inputs and responds instantly.
* **Deep Learning Models**: Uses CNN+LSTM for audio and only CNN for image-based recognition.

---

## Getting Started

### Requirements

* Python 3.8 or above
* pip
* Telegram bot token (create one using [@BotFather](https://t.me/BotFather))

### Setup

1. Clone the repository:

```bash
git clone https://github.com/bhramarisarkar/Automated_Emotion_Analysis-_A_Telegram_Bot_powered_by_Speech_and_Image_Emotion_Recognition.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Add your bot token in `config.py` or as an environment variable.

---

## Usage

1. Run the bot:

```bash
python main.py
```

2. Open Telegram and send a voice message or photo to the bot.
   It will reply based on the detected emotion.

---

## Project Structure

```
Automated_Emotion_Analysis-_A_Telegram_Bot_powered_by_Speech_and_Image_Emotion_Recognition/
├── Final_Audio_Code.ipynb/       # Audio emotion recognition training code
├── Final_Image_Code.ipynb/       # Facial emotion recognition training code
├── final_telegram_merged.py/     # Telegram bot logic
├── requirements.txt              # Dependency list
└── README.md
```

---

## Emotions Detected

* Happy
* Sad
* Angry
* Neutral
* Surprise
* Fear
* Disgust

---
## Datasets

We have downloaded the dataset directly from Kaggle

Following are the steps we used :
### Importing files from google colab
    from google.colab import files
   
    files.upload() 
### After this upload your own json file 
### Following steps are:
    import os
   
    import shutil 

   ### Make a Kaggle directory
    os.makedirs("/root/.kaggle", exist_ok=True)

   ### Move kaggle.json to the correct location
    shutil.move("kaggle.json", "/root/.kaggle/kaggle.json")

   ### Set permissions to prevent security warnings
    os.chmod("/root/.kaggle/kaggle.json", 600)
