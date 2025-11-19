import cv2
import numpy as np
import os
from tensorflow.keras.models import load_model

IMG_SIZE = 128

# -------------------
# Create output folder
# -------------------
output_folder = r"C:\Users\KIIT\Desktop\Agri-Land-Detection\new_model\outputs"
os.makedirs(output_folder, exist_ok=True)

# -------------------
# Load model
# -------------------
model = load_model(r"C:\Users\KIIT\Desktop\Agri-Land-Detection\new_model\agri_unet.h5")

# -------------------
# Predict mask
# -------------------
def predict_mask(img_path):
    img = cv2.imread(img_path)

    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    original = img.copy()

    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    inp = img_resized / 255.0
    inp = np.expand_dims(inp, axis=0)

    pred = model.predict(inp)[0]
    mask = (pred > 0.5).astype(np.uint8) * 255
    mask = mask.squeeze()

    return original, mask

# -------------------
# Extract & draw farmland boundaries
# -------------------
def draw_boundaries(original, mask):
    # Clean noise
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    overlay = original.copy()
    for c in contours:
        cv2.drawContours(overlay, [c], -1, (0, 255, 0), 1)

    return overlay

# -------------------
# RUN
# -------------------
input_image = r"C:\Users\KIIT\Desktop\Agri-Land-Detection\dataset\predict_images\img_5.png"

orig, mask = predict_mask(input_image)
overlay = draw_boundaries(orig, mask)

cv2.imwrite(os.path.join(output_folder, "original.png"), orig)
cv2.imwrite(os.path.join(output_folder, "mask.png"), mask)
cv2.imwrite(os.path.join(output_folder, "overlay.png"), overlay)

print("✅ Overlay saved!")
