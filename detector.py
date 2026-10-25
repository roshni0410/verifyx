import os
import cv2
import numpy as np
import tensorflow as tf

MODEL_PATH = os.path.join("model", "deepfake_model.keras")

# Load trained AI model
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = (224, 224)


def predict_image(image):
    """
    Class mapping:
    fake = 0
    real = 1
    """

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, IMG_SIZE)

    image_normalized = image_resized.astype("float32") / 255.0
    image_input = np.expand_dims(image_normalized, axis=0)

    prediction = model.predict(image_input, verbose=0)[0][0]

    # Model output = probability of REAL
    real_probability = float(prediction)
    fake_probability = 1.0 - real_probability

    real_percentage = real_probability * 100
    manipulated_percentage = fake_probability * 100

    if real_probability >= 0.5:
        result = "Real"
        confidence = real_percentage
    else:
        result = "Manipulated / AI Generated"
        confidence = manipulated_percentage

    return {
        "result": result,
        "confidence": round(confidence, 2),
        "real_percentage": round(real_percentage, 2),
        "manipulated_percentage": round(manipulated_percentage, 2),
        "ai_generated_percentage": round(manipulated_percentage, 2)
    }


def analyze_image(file_path):

    image = cv2.imread(file_path)

    if image is None:
        return {
            "type": "Image",
            "media_type": "Image",
            "result": "Error",
            "confidence": 0,
            "real_percentage": 0,
            "manipulated_percentage": 0,
            "ai_generated_percentage": 0,
            "frames_analyzed": 0,
            "analysis": "Unable to read the uploaded image."
        }

    prediction = predict_image(image)

    if prediction["result"] == "Real":
        analysis = (
            "The AI model classified this image as REAL. "
            "The model estimates that the image is more likely to be authentic "
            "than manipulated."
        )
    else:
        analysis = (
            "The AI model classified this image as MANIPULATED / AI GENERATED. "
            "The model estimates that the image contains characteristics "
            "associated with manipulated or AI-generated content."
        )

    return {
        "type": "Image",
        "media_type": "Image",

        "result": prediction["result"],
        "confidence": prediction["confidence"],

        "real_percentage": prediction["real_percentage"],
        "manipulated_percentage": prediction["manipulated_percentage"],
        "ai_generated_percentage": prediction["ai_generated_percentage"],

        "frames_analyzed": 1,
        "analysis": analysis
    }


def analyze_video(file_path):

    video = cv2.VideoCapture(file_path)

    if not video.isOpened():
        return {
            "type": "Video",
            "media_type": "Video",
            "result": "Error",
            "confidence": 0,
            "real_percentage": 0,
            "manipulated_percentage": 0,
            "ai_generated_percentage": 0,
            "frames_analyzed": 0,
            "analysis": "Unable to open the uploaded video."
        }

    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames <= 0:
        video.release()

        return {
            "type": "Video",
            "media_type": "Video",
            "result": "Error",
            "confidence": 0,
            "real_percentage": 0,
            "manipulated_percentage": 0,
            "ai_generated_percentage": 0,
            "frames_analyzed": 0,
            "analysis": "No frames were found in the video."
        }

    # Analyze maximum 10 frames
    number_of_frames = min(10, total_frames)

    frame_positions = np.linspace(
        0,
        total_frames - 1,
        number_of_frames,
        dtype=int
    )

    real_probabilities = []

    for position in frame_positions:

        video.set(cv2.CAP_PROP_POS_FRAMES, int(position))

        success, frame = video.read()

        if not success:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_rgb, IMG_SIZE)

        frame_normalized = frame_resized.astype("float32") / 255.0
        frame_input = np.expand_dims(frame_normalized, axis=0)

        prediction = model.predict(
            frame_input,
            verbose=0
        )[0][0]

        real_probabilities.append(float(prediction))

    video.release()

    if not real_probabilities:
        return {
            "type": "Video",
            "media_type": "Video",
            "result": "Error",
            "confidence": 0,
            "real_percentage": 0,
            "manipulated_percentage": 0,
            "ai_generated_percentage": 0,
            "frames_analyzed": 0,
            "analysis": "Unable to analyze video frames."
        }

    # Average prediction from analyzed frames
    average_real_probability = float(
        np.mean(real_probabilities)
    )

    average_fake_probability = 1.0 - average_real_probability

    real_percentage = average_real_probability * 100
    manipulated_percentage = average_fake_probability * 100

    if average_real_probability >= 0.5:
        result = "Real"
        confidence = real_percentage

        analysis = (
            "The AI model classified this video as REAL based on "
            "the analyzed video frames."
        )

    else:
        result = "Manipulated / AI Generated"
        confidence = manipulated_percentage

        analysis = (
            "The AI model classified this video as MANIPULATED / "
            "AI GENERATED based on the analyzed video frames."
        )

    return {
        "type": "Video",
        "media_type": "Video",

        "result": result,
        "confidence": round(float(confidence), 2),

        "real_percentage": round(float(real_percentage), 2),
        "manipulated_percentage": round(
            float(manipulated_percentage), 2
        ),
        "ai_generated_percentage": round(
            float(manipulated_percentage), 2
        ),

        "frames_analyzed": len(real_probabilities),
        "analysis": analysis
    }


def analyze_media(file_path):

    if not os.path.exists(file_path):

        return {
            "type": "Unknown",
            "media_type": "Unknown",
            "result": "Error",
            "confidence": 0,
            "real_percentage": 0,
            "manipulated_percentage": 0,
            "ai_generated_percentage": 0,
            "frames_analyzed": 0,
            "analysis": "File not found."
        }

    extension = os.path.splitext(file_path)[1].lower()

    image_extensions = [
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    ]

    video_extensions = [
        ".mp4",
        ".avi",
        ".mov",
        ".mkv",
        ".webm"
    ]

    if extension in image_extensions:

        return analyze_image(file_path)

    elif extension in video_extensions:

        return analyze_video(file_path)

    else:

        return {
            "type": "Unknown",
            "media_type": "Unknown",
            "result": "Unsupported File",
            "confidence": 0,
            "real_percentage": 0,
            "manipulated_percentage": 0,
            "ai_generated_percentage": 0,
            "frames_analyzed": 0,
            "analysis": "This file type is not supported."
        }
