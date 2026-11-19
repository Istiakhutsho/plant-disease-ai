# Plant Disease Detection AI — Streamlit Starter

This repository contains Streamlit code for the user's trained EfficientNetB0 model.

## Files
- `app.py`: Streamlit Dashboard
- `requirements.txt`: Python packages
- `class_names.json`: the exact class order the model was trained on

## Publish the 29.47 MB Model with GitHub Releases
1. In Colab's Google Drive mount, download the model file:
   `MyDrive/Plant_Disease_AI/final_plant_disease_model.keras`
2. Create a public GitHub repository, e.g., `plant-disease-ai`.
3. Upload `app.py`, `requirements.txt`, and `class_names.json` to the repository.
4. Open **Releases → Draft a new release**.
5. Create tag **v1.0** and attach `final_plant_disease_model.keras` under **Attach binaries**.
6. Publish release. Right-click its model attachment and copy the URL.
7. In `app.py` replace `PASTE_YOUR_GITHUB_RELEASE_MODEL_URL_HERE` with that exact URL.
8. Commit the updated `app.py`.
9. Visit https://share.streamlit.io, connect GitHub, choose the repository, branch `main`, and `app.py`; choose Python 3.13 if the option is offered.

Do not attempt to upload the model file with GitHub's ordinary browser *Add file → Upload files* function: that interface has a 25 MiB per-file limit. Release assets have a much larger limit.

## Model facts
- TensorFlow version during training: 2.21.0
- Input: 224 x 224 RGB image, values 0..255 (EfficientNetB0 handles normalization internally)
- Validation accuracy: 73.46%
- Final untouched test accuracy: 68.21% on 324 test images
- Classes: Anthracnose, BacterialSpot, Curl, Healthy, RingSpot

**Important:** This is an experimental model. In particular, Curl and BacterialSpot frequently get confused. The prediction is not a definitive disease diagnosis.

## Operational notes
- A public GitHub release makes the trained model downloadable by others.
- Streamlit Community Cloud has finite memory/CPU quotas. If a 2.21.0 TensorFlow CPU installation exceeds limits, consider a smaller inference format/model or another host.
- The image is processed in RAM. CSV history is stored only for the current Streamlit session, not persistently.
