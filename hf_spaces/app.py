import os
import random

import cv2
import gradio as gr
import numpy as np
from PIL import Image
from transformers import pipeline, AutoTokenizer, TFAutoModelForSequenceClassification, AutoImageProcessor, TFAutoModelForImageClassification


TEXT_MODEL_NAME = os.getenv("TEXT_MODEL_NAME", "duygucakir/emotion-analysis-distilbert")
FACE_MODEL_NAME = os.getenv("FACE_MODEL_NAME", "mo-thecreator/vit-Facial-Expression-Recognition")


random_texts = [
    "I am so happy with the results today!",
    "The session was somewhat boring.",
    "I am confused by the topics discussed.",
    "Very energetic and positive environment!",
    "I don't think this was useful at all.",
]


def load_text_pipeline():
    try:
        tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        # Load text model (assuming it's a native TF model, 'from_tf=True' is not needed here)
        model = TFAutoModelForSequenceClassification.from_pretrained(TEXT_MODEL_NAME)
        return pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
    except Exception as e:
        print(f"Warning: Could not load custom text model '{TEXT_MODEL_NAME}'. Reason: {e}")
        # Fallback to a generic sentiment analysis model if the custom one fails
        return pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


def load_face_pipeline():
    try:
        # Let pipeline handle loading model and processor automatically
        return pipeline('image-classification', model=FACE_MODEL_NAME)
    except Exception as e:
        print(f"Warning: Could not load face model '{FACE_MODEL_NAME}'. Reason: {e}")
        # Fallback to a generic image classification model if the custom one fails
        return pipeline("image-classification", model="google/vit-base-patch16-224")


sentiment_pipe = load_text_pipeline()
face_pipe = load_face_pipeline()

face_cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(face_cascade_path)

if face_cascade.empty():
    raise RuntimeError(f"Could not load Haar cascade from: {face_cascade_path}")


def normalize_label(label):
    return str(label).strip().lower().replace("label_", "label_")


def is_positive_text(label):
    label = normalize_label(label)
    return label in {"positive", "pos", "joy", "love", "happy", "happiness", "label_1", "label_2"}


def is_negative_text(label):
    label = normalize_label(label)
    return label in {"negative", "neg", "sadness", "anger", "fear", "disgust", "label_0", "label_3", "label_4"}


def is_positive_face(label):
    label = normalize_label(label)
    return label in {"happy", "happiness", "neutral", "surprise"}


def is_negative_face(label):
    label = normalize_label(label)
    return label in {"sad", "sadness", "angry", "anger", "fear", "disgust"}


def finish_processing(results, faces, img_cv):
    total = len(results)

    if total == 0:
        summary = "No results to aggregate."
    else:
        pos_sentiment = sum(1 for r in results if is_positive_text(r["sentiment"]))
        happy_neutral_expr = sum(1 for r in results if is_positive_face(r["emotion"]))

        aligned = 0
        mismatch = 0
        uncertain = 0

        for r in results:
            s = r["sentiment"]
            e = r["emotion"]

            if (is_positive_text(s) and is_positive_face(e)) or (is_negative_text(s) and is_negative_face(e)):
                aligned += 1
            elif (is_positive_text(s) and is_negative_face(e)) or (is_negative_text(s) and is_positive_face(e)):
                mismatch += 1
            else:
                uncertain += 1

        metric_rows = [
            (
                "Positive sentiment",
                f"{pos_sentiment / total * 100:.0f}%",
                "Percentage of entered text comments classified as positive by the text sentiment model.",
            ),
            (
                "Happy / neutral expression",
                f"{happy_neutral_expr / total * 100:.0f}%",
                "Percentage of detected faces classified as happy, neutral, or similarly non-negative by the facial expression model.",
            ),
            (
                "Text-face aligned",
                f"{aligned / total * 100:.0f}%",
                "Cases where the text sentiment and facial expression are interpreted in the same general direction, for example positive text with happy/neutral face.",
            ),
            (
                "Possible mismatch",
                f"{mismatch / total * 100:.0f}%",
                "Cases where the text and face labels point in opposite directions, for example positive text with sad/angry expression or negative text with happy/neutral expression.",
            ),
            (
                "Uncertain",
                f"{uncertain / total * 100:.0f}%",
                "Cases where one or both model labels could not be confidently mapped to positive or negative categories.",
            ),
        ]

        rows = "\n".join(
            f"| {metric} | {value} | {meaning} |"
            for metric, value, meaning in metric_rows
        )

        summary = f"""
### Aggregated Results ({total} faces processed)
| Metric | Value | Meaning |
| --- | --- | --- |
{rows}
**Note:** These values are model-based approximations for classroom demonstration. They should not be interpreted as reliable psychological or behavioral assessment.
"""

    return (
        faces,
        len(faces),
        img_cv,
        results,
        None,
        "**All faces processed. See results below.**",
        "",
        gr.update(interactive=False),
        summary,
    )


def display_face(idx, faces, img_cv, results):
    if idx >= len(faces):
        return finish_processing(results, faces, img_cv)

    x, y, w, h = faces[idx]
    display_img = img_cv.copy()
    cv2.rectangle(display_img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    display_pil = Image.fromarray(cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB))

    status = f"**Processing Face {idx + 1} of {len(faces)}**"
    random_text = random.choice(random_texts)

    return (
        faces,
        idx,
        img_cv,
        results,
        display_pil,
        status,
        random_text,
        gr.update(interactive=True),
        "",
    )


def start_processing(image):
    if image is None:
        return (
            [],
            0,
            None,
            [],
            None,
            "No image uploaded.",
            "",
            gr.update(interactive=False),
            "",
        )

    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(detected) == 0:
        return (
            [],
            0,
            None,
            [],
            None,
            "No faces detected.",
            "",
            gr.update(interactive=False),
            "",
        )

    # Convert NumPy integers to Python integers for safer Gradio State handling.
    faces = [tuple(map(int, box)) for box in detected]
    return display_face(0, faces, img_cv, [])


def process_next(feedback_text, faces, idx, img_cv, results):
    if img_cv is None or idx >= len(faces):
        return finish_processing(results, faces, img_cv)

    x, y, w, h = faces[idx]
    face_crop = img_cv[y : y + h, x : x + w]
    face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

    try:
        emotion_result = face_pipe(face_pil)[0]["label"].lower()
    except Exception as e:
        print(f"Face classification failed: {e}")
        emotion_result = "unknown"

    try:
        text = feedback_text if feedback_text and feedback_text.strip() else "No comment."
        sentiment_result = sentiment_pipe(text)[0]["label"].lower()
    except Exception as e:
        print(f"Text classification failed: {e}")
        sentiment_result = "unknown"

    results = list(results)
    results.append({"emotion": emotion_result, "sentiment": sentiment_result})

    return display_face(idx + 1, faces, img_cv, results)


with gr.Blocks() as demo:
    gr.Markdown("## Multimodal Workshop Feedback App")
    gr.Markdown(
        """
**Disclaimer:** This demo is for educational purposes only. Facial expression analysis does not determine a person’s true emotional state.
Uploaded images and text inputs should not be used for psychological, medical, legal, educational assessment, recruitment, security, or other high-stakes decisions.
**Önemli Uyarı:** Gönüllü katılımcılar fotoğraf ve kısa metin girdisi sağlar. Uygulama her kişi için facial expression label ve text sentiment label üretir. Sonuçlar yalnızca demo sonunda aggregate edilir; bireysel sonuçlar değerlendirme veya karar verme amacıyla kullanılmaz.
"""
    )

    faces_state = gr.State([])
    index_state = gr.State(0)
    img_state = gr.State(None)
    results_state = gr.State([])

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(type="pil", label="Upload Group Photo")
            upload_btn = gr.Button("Detect Faces & Start")

        with gr.Column():
            face_image = gr.Image(type="pil", label="Current Detected Face")
            status_text = gr.Markdown("**Please upload an image and click Detect Faces.**")
            feedback_input = gr.Textbox(label="Short feedback text for the detected face", interactive=True)
            next_btn = gr.Button("Next Face", interactive=False)

    results_output = gr.Markdown(label="Aggregated Results")

    upload_btn.click(
        start_processing,
        inputs=[image_input],
        outputs=[
            faces_state,
            index_state,
            img_state,
            results_state,
            face_image,
            status_text,
            feedback_input,
            next_btn,
            results_output,
        ],
    )

    next_btn.click(
        process_next,
        inputs=[feedback_input, faces_state, index_state, img_state, results_state],
        outputs=[
            faces_state,
            index_state,
            img_state,
            results_state,
            face_image,
            status_text,
            feedback_input,
            next_btn,
            results_output,
        ],
    )


if __name__ == "__main__":
    demo.launch()
