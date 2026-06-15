import torch
import numpy as np

from PIL import Image

import torchvision.transforms.functional as TF

THRESHOLD = 0.35


def predict_image(model, image_path):

    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    resized = image.resize((256, 256))

    tensor = TF.to_tensor(resized)
    tensor = tensor.unsqueeze(0)

    with torch.no_grad():

        logits = model(tensor)

        probs = torch.sigmoid(logits)

        probs = probs.squeeze().cpu().numpy()

    mask = (probs > THRESHOLD).astype(np.uint8)

    tumor_pixels = int(mask.sum())

    total_pixels = mask.shape[0] * mask.shape[1]

    tumor_area = (
        tumor_pixels / total_pixels
    ) * 100

    confidence = float(probs.max())

    prediction = (
        "Tumor Detected"
        if tumor_pixels > 500
        else "No Tumor Detected"
    )

    return {
        "original": original,
        "mask": mask,
        "confidence": round(confidence * 100, 2),
        "tumor_area": round(tumor_area, 2),
        "prediction": prediction,
        "tumor_pixels": tumor_pixels,
    }