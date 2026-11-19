"""Plant Leaf Disease Detection dashboard for Streamlit Community Cloud.

Model is hosted as a public GitHub Release asset, not in the Git repository.
The EfficientNetB0/Keras model from this project expects 224x224 RGB images
with values on the original 0..255 scale (no manual /255 normalization).
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import requests
import streamlit as st
import tensorflow as tf

st.set_page_config(
    page_title="Plant Disease Detection AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# IMPORTANT: Replace this text with the public "Copy link address" of the
# final_plant_disease_model.keras asset in your GitHub Release.
MODEL_URL = "PASTE_YOUR_GITHUB_RELEASE_MODEL_URL_HERE"

MODEL_PATH = Path("/tmp/plant_disease_final.keras")
CLASS_NAMES_FILE = Path(__file__).with_name("class_names.json")
IMAGE_SIZE = (224, 224)

with CLASS_NAMES_FILE.open("r", encoding="utf-8") as f:
    CLASS_NAMES = json.load(f)

assert CLASS_NAMES == [
    "Anthracnose", "BacterialSpot", "Curl", "Healthy", "RingSpot"
], "Unexpected model class order. Check class_names.json."

DISPLAY_NAMES = {
    "Anthracnose": "Anthracnose",
    "BacterialSpot": "Bacterial Spot",
    "Curl": "Curl",
    "Healthy": "Healthy Leaf",
    "RingSpot": "Ring Spot",
}

st.markdown(
    """
    <style>
      .block-container {padding-top: 2rem; max-width: 1150px;}
      .hero {padding: 1.2rem 1.5rem; border-radius: 14px;
             background: linear-gradient(110deg,#12462e,#26734b); color: white;
             margin-bottom: 1.25rem;}
      .hero h1 {color: white; margin-bottom: .3rem; font-size: 2.0rem;}
      .hero p {color: #e6f5eb; margin: 0;}
      .note {font-size: .9rem; color: #56655c;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="AI মডেল লোড হচ্ছে...")
def load_ai_model(model_url: str):
    """Download public Release asset once per server session and cache the model."""
    if not model_url.startswith("https://"):
        raise ValueError("সঠিক HTTPS GitHub Release Model URL বসাও।")

    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size < 1024 * 1024:
        temp_path = MODEL_PATH.with_suffix(".partial")
        with requests.get(model_url, stream=True, timeout=(20, 180)) as response:
            response.raise_for_status()
            with temp_path.open("wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
        if temp_path.stat().st_size < 1024 * 1024:
            temp_path.unlink(missing_ok=True)
            raise ValueError("ডাউনলোড করা Model ফাইলটি সঠিক নয়। URL পরীক্ষা করো।")
        os.replace(temp_path, MODEL_PATH)

    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def classify_image(model, img: Image.Image) -> np.ndarray:
    # This trained EfficientNetB0 contains its own preprocessing.
    # Never divide the image by 255 here.
    img = img.convert("RGB").resize(IMAGE_SIZE, Image.Resampling.BILINEAR)
    arr = np.asarray(img, dtype=np.float32)[None, ...]
    probabilities = model.predict(arr, verbose=0)[0]
    if probabilities.shape[0] != len(CLASS_NAMES):
        raise ValueError("Model output size does not match class_names.json")
    return probabilities


st.markdown(
    '<div class="hero"><h1>🌿 Plant Disease Detection AI</h1>'
    '<p>পাতার ছবি আপলোড করে পাঁচটি Class-এর মধ্যে সম্ভাব্য রোগ শনাক্ত করো</p></div>',
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("📊 Model Information")
    st.metric("Final Test Accuracy", "68.21%")
    st.caption("৩২৪টি আলাদা Test Image-এ মূল্যায়িত ফলাফল")
    st.write("**Model:** Fine-tuned EfficientNetB0")
    st.write("**Input:** 224 × 224 RGB")
    st.write("**Classes:** 5")
    st.write("**Available classes:**")
    for name in CLASS_NAMES:
        st.write("•", DISPLAY_NAMES[name])
    st.warning(
        "এটি পরীক্ষামূলক AI। বিশেষ করে Curl ও Bacterial Spot-এর ক্ষেত্রে ভুল হতে পারে। "
        "কৃষি বিশেষজ্ঞের পরামর্শের বিকল্প নয়।"
    )

st.subheader("📤 পাতার ছবি আপলোড করো")
st.caption("JPG, JPEG, PNG অথবা WEBP ফাইল সমর্থিত")
uploaded_file = st.file_uploader(
    "পাতার একটি পরিষ্কার ছবি নির্বাচন করো",
    type=["jpg", "jpeg", "png", "webp"],
)

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file is not None:
    try:
        original_image = Image.open(uploaded_file)
        original_image.load()
        display_image = original_image.convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError):
        st.error("ছবিটি খোলা যাচ্ছে না। অন্য JPG বা PNG ছবি দাও।")
        st.stop()

    left, right = st.columns([1, 1], gap="large")
    with left:
        st.image(display_image, caption="Uploaded Leaf Image", use_container_width=True)
    with right:
        st.write("**বিশ্লেষণের জন্য প্রস্তুত**")
        st.write(f"ছবির মাপ: {display_image.width} × {display_image.height} pixels")
        analyze = st.button("🔍 Detect Disease", type="primary", use_container_width=True)

    if analyze:
        if MODEL_URL == "PASTE_YOUR_GITHUB_RELEASE_MODEL_URL_HERE":
            st.error(
                "app.py ফাইলে MODEL_URL পরিবর্তন করে GitHub Release-এর "
                "Model Download Link বসাতে হবে।"
            )
            st.stop()

        try:
            model = load_ai_model(MODEL_URL)
            probs = classify_image(model, display_image)
        except Exception as exc:
            st.error("Model Load বা Prediction করতে সমস্যা হয়েছে।")
            st.code(f"{type(exc).__name__}: {exc}")
            st.stop()

        best_idx = int(np.argmax(probs))
        best_class = CLASS_NAMES[best_idx]
        best_score = float(probs[best_idx])

        st.divider()
        st.subheader("🧠 AI Prediction Result")
        a, b = st.columns(2)
        with a:
            st.metric("Predicted Class", DISPLAY_NAMES[best_class])
        with b:
            st.metric("Model Confidence", f"{best_score * 100:.1f}%")

        if best_score < 0.60:
            st.warning("Model-এর Confidence কম। ভালো আলোতে নতুন ছবি দিয়ে পরীক্ষা করো।")
        elif best_class == "Healthy":
            st.info("Model এই ছবিকে Healthy Class হিসেবে শনাক্ত করেছে।")
        else:
            st.info("এটি সম্ভাব্য Class, নিশ্চিত রোগ নির্ণয় নয়।")

        result_df = pd.DataFrame({
            "Class": [DISPLAY_NAMES[name] for name in CLASS_NAMES],
            "Confidence (%)": [round(float(p) * 100, 2) for p in probs],
        }).sort_values("Confidence (%)", ascending=False)
        st.write("**পাঁচটি Class-এর Prediction Score**")
        st.bar_chart(result_df.set_index("Class"), horizontal=True)
        st.dataframe(result_df, hide_index=True, use_container_width=True)

        st.session_state.history.append({
            "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Image": uploaded_file.name,
            "Predicted Class": best_class,
            "Confidence (%)": round(best_score * 100, 2),
        })

st.divider()
st.subheader("🗂️ এই সেশনের Prediction History")
if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df.iloc[::-1], hide_index=True, use_container_width=True)
    st.download_button(
        "⬇️ Download History (CSV)",
        data=history_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="plant_disease_history.csv",
        mime="text/csv",
    )
else:
    st.caption("এখনো কোনো Prediction করা হয়নি।")

st.caption(
    "নোট: History শুধু বর্তমান ব্রাউজার সেশনে রাখা হয়। "
    "এটি স্থায়ী ডাটাবেস নয়। Prediction Confidence একটি Model Score, "
    "নিশ্চিত রোগ হওয়ার সম্ভাবনা নয়।"
)
