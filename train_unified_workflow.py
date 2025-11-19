import os
import shutil
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# --- Configuration ---
IMG_SIZE = 128
val_ratio = 0.2
output_base = "dataset_split"

# 🛑 INPUT PATHS (Source data)
SRC_IMG_DIR = r"C:\Users\KIIT\Desktop\Agri-Land-Detection\dataset\images1"
SRC_MASK_DIR = r"C:\Users\KIIT\Desktop\Agri-Land-Detection\dataset\masks1"

# 🟢 OUTPUT PATHS (New folders created by the script)
TRAIN_IMG_DIR = os.path.join(output_base, "train/images")
TRAIN_MASK_DIR = os.path.join(output_base, "train/masks")
VAL_IMG_DIR = os.path.join(output_base, "val/images")
VAL_MASK_DIR = os.path.join(output_base, "val/masks")


# ----------------------------------------------
# 1. EXPLICIT FILE SYSTEM DATA SPLITTING
# ----------------------------------------------

def split_dataset_files():
    """Shuffles the dataset and copies files into train/val folders."""
    print("--- Starting File System Split (80/20) ---")

    # Create directories
    for d in [TRAIN_IMG_DIR, TRAIN_MASK_DIR, VAL_IMG_DIR, VAL_MASK_DIR]:
        os.makedirs(d, exist_ok=True)

    # List all image files
    images = sorted([f for f in os.listdir(SRC_IMG_DIR) if f.endswith(".png")])
    random.shuffle(images)

    # Calculate split count
    val_count = int(len(images) * val_ratio)
    val_set = images[:val_count]
    train_set = images[val_count:]

    print(f"Total images: {len(images)}")
    print(f"Train Set Size: {len(train_set)}")
    print(f"Validation Set Size: {len(val_set)}")

    def copy_pairs(file_list, img_out, mask_out):
        for img_file in file_list:
            mask_file = img_file.replace("img_", "mask_")
            src_img = os.path.join(SRC_IMG_DIR, img_file)
            src_mask = os.path.join(SRC_MASK_DIR, mask_file)
            if os.path.exists(src_mask):
                shutil.copy(src_img, os.path.join(img_out, img_file))
                shutil.copy(src_mask, os.path.join(mask_out, mask_file))

    copy_pairs(train_set, TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    copy_pairs(val_set, VAL_IMG_DIR, VAL_MASK_DIR)
    print("✅ Train/Val file split complete!")


# ----------------------------------------------
# 2. MODIFIED DATA LOADING FOR SPLIT FOLDERS
# ----------------------------------------------

def load_split_dataset(img_folder, mask_folder):
    """Loads images and masks from specific train/val folders into memory."""
    X, Y = [], []
    files = sorted(os.listdir(img_folder))

    for f in files:
        img_path = os.path.join(img_folder, f)
        img = load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
        img = img_to_array(img) / 255.0

        mask_name = f.replace("img", "mask")
        mask_path = os.path.join(mask_folder, mask_name)
        mask = load_img(mask_path, color_mode="grayscale", target_size=(IMG_SIZE, IMG_SIZE))
        mask = img_to_array(mask) / 255.0

        X.append(img)
        Y.append(mask)

    return np.array(X), np.array(Y)


# ----------------------------------------------
# 3. U-NET Model Definition (Unchanged)
# ----------------------------------------------

def unet_model(input_size=(IMG_SIZE, IMG_SIZE, 3)):
    inputs = Input(input_size)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(16, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(32, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(64, 3, activation='relu', padding='same')(c3)
    u1 = UpSampling2D()(c3)
    u1 = concatenate([u1, c2])
    c4 = Conv2D(32, 3, activation='relu', padding='same')(u1)
    c4 = Conv2D(32, 3, activation='relu', padding='same')(c4)
    u2 = UpSampling2D()(c4)
    u2 = concatenate([u2, c1])
    c5 = Conv2D(16, 3, activation='relu', padding='same')(u2)
    c5 = Conv2D(16, 3, activation='relu', padding='same')(c5)
    outputs = Conv2D(1, 1, activation='sigmoid')(c5)
    return Model(inputs, outputs)


# ----------------------------------------------
# 4. MAIN EXECUTION BLOCK (Split, Load, Train, Evaluate)
# ----------------------------------------------

if __name__ == '__main__':
    # 1. Perform File System Split
    split_dataset_files()

    # 2. Load Data from Split Folders
    print("\n--- Loading Data from Split Folders ---")
    X_train, Y_train = load_split_dataset(TRAIN_IMG_DIR, TRAIN_MASK_DIR)
    X_val, Y_val = load_split_dataset(VAL_IMG_DIR, VAL_MASK_DIR)

    # 3. Model Training
    model = unet_model()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("\n--- Starting Model Training ---")
    history = model.fit(
        X_train, Y_train,
        batch_size=8,
        epochs=20,
        validation_data=(X_val, Y_val)
    )

    model.save("agri_unet.h5")
    print("\n✅ Training Complete — Model Saved!")

    # --------------------------------------------------
    # 5. ADDED: PLOT TRAINING HISTORY CURVES
    # --------------------------------------------------

    # Loss Curve Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_curve.png')

    # Accuracy Curve Plot
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_curve.png')
    print("✅ Training History Curves (loss_curve.png, accuracy_curve.png) saved!")

    # --------------------------------------------------
    # 6. MODEL EVALUATION AND ROC CURVE
    # --------------------------------------------------

    print("\n--- Starting Final Model Evaluation on Validation Set ---")

    # Make Predictions
    Y_pred_prob = model.predict(X_val)

    # Flatten data for sklearn metrics
    Y_true_flat = Y_val.flatten()
    Y_prob_flat = Y_pred_prob.flatten()
    Y_pred_mask = (Y_pred_prob > 0.5).astype(np.uint8)
    Y_pred_flat = Y_pred_mask.flatten()

    # Calculate Metrics
    accuracy = (Y_true_flat == Y_pred_flat).mean() * 100
    f1 = f1_score(Y_true_flat, Y_pred_flat)
    recall = recall_score(Y_true_flat, Y_pred_flat)
    precision = precision_score(Y_true_flat, Y_pred_flat)
    auc = roc_auc_score(Y_true_flat, Y_prob_flat)

    print(f"\n--- Final Validation Performance ---")
    print(f"Pixel Accuracy: {accuracy:.2f}%")
    print(f"F1-Score: {f1:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"AUC: {auc:.4f}")

    # Calculate and plot Confusion Matrix
    cm = confusion_matrix(Y_true_flat, Y_pred_flat)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Non-Land', 'Agri-Land'])
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix (Pixel Level)")
    plt.savefig("confusion_matrix.png")

    # ADDED: ROC Curve Plot
    fpr, tpr, thresholds = roc_curve(Y_true_flat, Y_prob_flat)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Chance')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('roc_curve.png')
    print("✅ ROC Curve (roc_curve.png) saved!")