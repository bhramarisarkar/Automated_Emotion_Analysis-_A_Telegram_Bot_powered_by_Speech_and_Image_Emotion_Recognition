import logging
import os
import tempfile
import numpy as np
import cv2
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
from pydub import AudioSegment

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Constants
TELEGRAM_TOKEN = "bot_token_here"
IMAGE_MODEL_PATH = "model_optimal.h5"  # Path to image emotion model
AUDIO_MODEL_PATH = "ser_model_real_world_v2.h5"  # Path to audio emotion model
MAX_FRAMES = 200  # For audio processing

# Load the image emotion model
try:
    # Try loading the optimal model first
    image_model = tf.keras.models.load_model(IMAGE_MODEL_PATH)
    print("Loaded optimal image model")
except:
    try:
        # Fall back to regular model if optimal isn't available
        image_model = tf.keras.models.load_model('model.weights.h5')
        print("Loaded regular image model")
    except Exception as e:
        print(f"Error loading image models: {e}")
        logger.error(f"Error loading image models: {e}")
        image_model = None

# Load the audio emotion model
try:
    audio_model = load_model(AUDIO_MODEL_PATH)
    print("Loaded audio emotion model")
except Exception as e:
    print(f"Error loading audio model: {e}")
    logger.error(f"Error loading audio model: {e}")
    audio_model = None

# Define image emotion labels
IMAGE_EMOTIONS = {
    0: 'Surprise',
    1: 'Disgust',
    2: 'Fear',
    3: 'Happiness',
    4: 'Neutral',
    5: 'Anger',
    6: 'Sadness'
}
#surprise -> disgust

# Define audio emotion labels
AUDIO_EMOTIONS = ['angry', 'happy', 'sad', 'neutral', 'fear', 'disgust', 'surprise']

# Emotion emojis for display
EMOTION_EMOJIS = {
    'angry': '😠',
    'anger': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😄',
    'happiness': '😄',
    'neutral': '😐',
    'sad': '😢',
    'sadness': '😢',
    'surprise': '😲'
}

# Face detection using OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# COMMAND HANDLERS
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    user = update.effective_user
    await update.message.reply_text(
        f"Hi {user.first_name}! I'm an emotion detection bot.\n\n"
        f"Send me a photo with a face, and I'll analyze the visual emotion.\n"
        f"Or send me a voice message or audio file, and I'll analyze the emotion in your voice!\n"
        f"Use /help for more information."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "📱 *Emotion Detection Bot Help* 📱\n\n"
        "*Image Analysis:*\n"
        "- Send a photo with a clear face\n"
        "- Make sure the face is clearly visible and well-lit\n\n"
        "*Audio Analysis:*\n"
        "- Send a voice message or audio file\n"
        "- Supported formats: voice messages, audio files\n"
        "- You can also upload .wav files as documents\n\n"
        "*Commands:*\n"
        "/start - Start the bot\n"
        "/help - Show this help message",
        parse_mode='Markdown'
    )


# IMAGE PROCESSING FUNCTIONS
def preprocess_image(image):
    """Preprocess the image for the model."""
    # Detect faces - use BGR image directly for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    if len(faces) == 0:
        return None

    # Process the largest face found
    x, y, w, h = max(faces, key=lambda rect: rect[2] * rect[3])

    # Extract face ROI from the original RGB image
    face_roi = image[y:y + h, x:x + w]

    # Resize to the input size expected by your model (e.g., 100x100)
    resized_face = cv2.resize(face_roi, (100, 100))

    # Normalize pixel values
    normalized_face = resized_face / 255.0

    # Reshape for model input (add batch dimension)
    return np.expand_dims(normalized_face, 0)


async def process_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process a photo and predict emotion."""
    if image_model is None:
        await update.message.reply_text(
            "Sorry, the image emotion model is not available. Please try with an audio sample instead.")
        return

    # Send processing message
    processing_msg = await update.message.reply_text("🖼️ Processing your image... Please wait.")

    # Get the photo file
    photo_file = await update.message.photo[-1].get_file()

    # Create a temporary file to save the photo
    photo_path = f"user_photo_{update.effective_user.id}.jpg"
    await photo_file.download_to_drive(photo_path)

    try:
        # Load and preprocess the image
        image = cv2.imread(photo_path)
        if image is None:
            await update.message.reply_text("I couldn't read the image. Please try again with a different photo.")
            await processing_msg.delete()
            return

        processed_image = preprocess_image(image)

        if processed_image is None:
            await update.message.reply_text(
                "I couldn't detect any faces in the image. Please try again with a clearer photo.")
            await processing_msg.delete()
            return

        # Add logging to see the processed image shape
        logger.info(f"Processed image shape: {processed_image.shape}")

        # Predict emotion
        prediction = image_model.predict(processed_image)
        emotion_idx = np.argmax(prediction)

        # Add logging to see raw prediction values
        logger.info(f"Raw prediction: {prediction}")
        logger.info(f"Predicted emotion index: {emotion_idx}")

        # Get the emotion label
        emotion = IMAGE_EMOTIONS.get(emotion_idx, f"Unknown emotion (index {emotion_idx})")
        confidence = float(prediction[0][emotion_idx]) * 100

        # Get emoji for emotion
        emoji = EMOTION_EMOJIS.get(emotion.lower(), '🤔')

        # Send the result
        await update.message.reply_text(
            f"{emoji} Visual emotion detected: *{emotion}*\n"
            f"Confidence: {confidence:.2f}%\n\n"
            f"Remember that my predictions are just estimates, and your actual feelings matter most! 😊",
            parse_mode='Markdown'
        )

    except Exception as e:
        logger.error(f"Error processing image: {e}")
        await update.message.reply_text(
            f"Sorry, I encountered an error while processing your image: {str(e)}\nPlease try again with another photo."
        )
    finally:
        # Clean up
        if os.path.exists(photo_path):
            os.remove(photo_path)
        # Delete processing message
        await processing_msg.delete()


# AUDIO PROCESSING FUNCTIONS
def extract_features(file_path):
    """Extract audio features for the model."""
    y, sr = librosa.load(file_path, sr=16000)

    # Extract Features
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40)
    contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    # Set a fixed length for all features
    def pad_or_truncate(feature):
        if feature.shape[1] > MAX_FRAMES:
            return feature[:, :MAX_FRAMES]  # Truncate
        else:
            return np.pad(feature, ((0, 0), (0, MAX_FRAMES - feature.shape[1])), mode='constant')

    mfcc = pad_or_truncate(mfcc)
    mfcc_delta = pad_or_truncate(mfcc_delta)
    mfcc_delta2 = pad_or_truncate(mfcc_delta2)
    mel_spec = pad_or_truncate(mel_spec)
    contrast = pad_or_truncate(contrast)

    feature_stack = np.vstack((mfcc, mfcc_delta, mfcc_delta2, mel_spec, contrast))
    return feature_stack


def predict_emotion(audio_path):
    """Predict emotion from audio file."""
    try:
        # Extract features
        features = extract_features(audio_path)

        # Reshape for model input (add batch and channel dimensions)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        features = np.expand_dims(features, axis=-1)  # Add channel dimension

        # Make prediction
        prediction = audio_model.predict(features)[0]

        # Get the predicted emotion and confidence
        emotion_idx = np.argmax(prediction)
        confidence = prediction[emotion_idx] * 100

        return AUDIO_EMOTIONS[emotion_idx], confidence

    except Exception as e:
        logger.error(f"Error in audio prediction: {str(e)}")
        return None, None


def convert_to_wav(input_file, output_file):
    """Convert any audio format to WAV if needed."""
    try:
        audio = AudioSegment.from_file(input_file)
        audio.export(output_file, format="wav")
        return True
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        return False


def is_wav_file(file_path):
    """Check if file is WAV."""
    try:
        if file_path.lower().endswith('.wav'):
            return True
        return False
    except:
        return False


async def process_audio(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process voice message or audio file and predict emotion."""
    if audio_model is None:
        await update.message.reply_text(
            "Sorry, the audio emotion model is not available. Please try with an image instead.")
        return

    # Send processing message
    processing_msg = await update.message.reply_text("🎧 Processing your audio... Please wait.")

    try:
        # Get audio file from message
        if update.message.voice:
            audio_file = await update.message.voice.get_file()
            original_format = "ogg"
        elif update.message.audio:
            audio_file = await update.message.audio.get_file()
            original_format = os.path.splitext(update.message.audio.file_name)[1][
                              1:] if update.message.audio.file_name else "mp3"
        else:
            await update.message.reply_text("❌ Unsupported file format. Please send a voice message or audio file.")
            await processing_msg.delete()
            return

        # Create temporary file for the original audio
        with tempfile.NamedTemporaryFile(suffix=f'.{original_format}', delete=False) as temp_file:
            original_path = temp_file.name

        # Download the audio to temporary file
        await audio_file.download_to_drive(custom_path=original_path)

        # Process path for analysis
        if is_wav_file(original_path):
            analysis_path = original_path
        else:
            # Convert to WAV if not already in WAV format
            await update.message.reply_text("🔄 Converting audio format...")
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as wav_file:
                wav_path = wav_file.name

            if convert_to_wav(original_path, wav_path):
                analysis_path = wav_path
            else:
                await update.message.reply_text("❌ Error converting your audio file. Please try a different format.")
                os.unlink(original_path)
                await processing_msg.delete()
                return

        # Process audio and get prediction
        await update.message.reply_text("🔍 Analyzing emotions in your audio...")
        emotion, confidence = predict_emotion(analysis_path)

        # Remove temporary files
        if original_path != analysis_path and os.path.exists(analysis_path):
            os.unlink(analysis_path)
        os.unlink(original_path)

        # Send prediction result
        if emotion:
            emoji = EMOTION_EMOJIS.get(emotion, '🤔')

            await update.message.reply_text(
                f"{emoji} Audible emotion detected: *{emotion.upper()}*\n"
                f"Confidence: {confidence:.2f}%",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("❌ Sorry, I couldn't detect any emotion in this audio.")

        # Delete processing message
        await processing_msg.delete()

    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        await update.message.reply_text(f"❌ Error processing your audio: {str(e)}")
        await processing_msg.delete()


async def process_document(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process document files (for .wav uploads)"""
    if audio_model is None:
        await update.message.reply_text(
            "Sorry, the audio emotion model is not available. Please try with an image instead.")
        return

    # Check if document is a WAV file
    if not update.message.document.file_name.lower().endswith('.wav'):
        await update.message.reply_text("❌ Only WAV files are supported for document uploads. Please send a WAV file.")
        return

    # Send processing message
    processing_msg = await update.message.reply_text("🎧 Processing your WAV file... Please wait.")

    try:
        # Get document file
        doc_file = await update.message.document.get_file()

        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            temp_path = temp_file.name

        # Download the document to temporary file
        await doc_file.download_to_drive(custom_path=temp_path)

        # Process audio and get prediction
        await update.message.reply_text("🔍 Analyzing emotions in your audio...")
        emotion, confidence = predict_emotion(temp_path)

        # Remove temporary file
        os.unlink(temp_path)

        # Send prediction result
        if emotion:
            emoji = EMOTION_EMOJIS.get(emotion, '🤔')

            await update.message.reply_text(
                f"{emoji} Audible emotion detected: *{emotion.upper()}*\n"
                f"Confidence: {confidence:.2f}%",
                parse_mode='Markdown'
            )
        else:
            await update.message.reply_text("❌ Sorry, I couldn't detect any emotion in this audio.")

        # Delete processing message
        await processing_msg.delete()

    except Exception as e:
        logger.error(f"Error processing WAV file: {str(e)}")
        await update.message.reply_text(f"❌ Error processing your WAV file: {str(e)}")
        await processing_msg.delete()


def main() -> None:
    """Start the bot."""
    print("🤖 Starting Multi-Modal Emotion Recognition Bot...")

    # Create the Application and pass it your bot's token
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Add command handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))

    # Add message handlers
    application.add_handler(MessageHandler(filters.PHOTO, process_photo))
    application.add_handler(MessageHandler(filters.VOICE | filters.AUDIO, process_audio))
    application.add_handler(MessageHandler(filters.Document.ALL, process_document))

    # Run the bot until the user presses Ctrl-C
    application.run_polling()


if __name__ == "__main__":
    main()
