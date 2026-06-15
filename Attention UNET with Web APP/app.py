from flask import (
    Flask,
    render_template,
    request
)

import os
import cv2
import numpy as np

from model import load_model
from inference import predict_image
from utils import create_overlay

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
MASK_FOLDER = "static/masks"
OVERLAY_FOLDER = "static/overlays"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MASK_FOLDER, exist_ok=True)
os.makedirs(OVERLAY_FOLDER, exist_ok=True)

MODEL = load_model(
    "weights/best_attunet.pth"
)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["image"]

    filepath = os.path.join(
        UPLOAD_FOLDER,
        file.filename
    )

    file.save(filepath)

    result = predict_image(
        MODEL,
        filepath
    )

    mask_path = os.path.join(
        MASK_FOLDER,
        file.filename.replace(
            ".tif",
            "_mask.png"
        )
    )

    overlay_path = os.path.join(
        OVERLAY_FOLDER,
        file.filename.replace(
            ".tif",
            "_overlay.png"
        )
    )

    cv2.imwrite(
        mask_path,
        result["mask"] * 255
    )

    overlay = create_overlay(
        result["original"],
        result["mask"]
    )

    cv2.imwrite(
        overlay_path,
        cv2.cvtColor(
            overlay,
            cv2.COLOR_RGB2BGR
        )
    )

    return render_template(
        "result.html",
        prediction=result["prediction"],
        confidence=result["confidence"],
        tumor_area=result["tumor_area"],
        tumor_pixels=result["tumor_pixels"],
        image=file.filename,
        mask=os.path.basename(mask_path),
        overlay=os.path.basename(overlay_path)
    )


if __name__ == "__main__":
    app.run(
        debug=True
    )