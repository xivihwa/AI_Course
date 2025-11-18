import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_CACHE_DISABLE'] = '1'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['XLA_FLAGS'] = '--xla_gpu_autotune_level=0'
import warnings
warnings.filterwarnings('ignore')
import sys
from contextlib import redirect_stderr, redirect_stdout
import time
with open(os.devnull, 'w') as devnull:
    with redirect_stderr(devnull):
        with redirect_stdout(devnull):
            import tensorflow as tf
from google.protobuf import message as _message
try:
    from google import protobuf
except Exception:
    pass
tf.get_logger().setLevel('ERROR')
try:
    import absl.logging as absl_logging
    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras import layers, models, Input
from tensorflow.keras.applications import Xception
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import cv2
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

DATASET_DIR = "/kaggle/input/lab6-data/dataset"
VIDEO_PATH = "/kaggle/input/lab6-video/Coca-Cola _ Holidays Are Coming.mp4"
WORK_DIR = "/kaggle/working/dataset"
AUGMENTED_DIR = "/kaggle/working/augmented_dataset"
MODEL_PATH = "/kaggle/working/best_model.h5"
TARGET_SIZE = (299, 299)
BATCH_SIZE = 16
EPOCHS_FROZEN = 15
EPOCHS_FINE_TUNE = 10

print("\n[INFO] Checking and cleaning dataset...")
if os.path.exists(WORK_DIR):
    shutil.rmtree(WORK_DIR)
shutil.copytree(DATASET_DIR, WORK_DIR)
DATASET_DIR = WORK_DIR

corrupted_count = 0
for root, dirs, files in os.walk(DATASET_DIR):
    for file in files:
        fp = os.path.join(root, file)
        try:
            with Image.open(fp) as im:
                im.convert("RGB")
        except Exception:
            try:
                os.remove(fp)
                corrupted_count += 1
            except Exception:
                pass

print(f"[INFO] Dataset cleaned. Removed {corrupted_count} corrupted images.\n")

print("[INFO] Analyzing class distribution...")
positive_dir = os.path.join(DATASET_DIR, "positive")
negative_dir = os.path.join(DATASET_DIR, "negative")
positive_count = len([f for f in os.listdir(positive_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
negative_count = len([f for f in os.listdir(negative_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"Positive class: {positive_count} images")
print(f"Negative class: {negative_count} images")
ratio_text = f"{positive_count / negative_count:.2f}" if negative_count > 0 else "inf"
print(f"Ratio: {ratio_text}\n")

print("[INFO] Performing data augmentation...")
if os.path.exists(AUGMENTED_DIR):
    shutil.rmtree(AUGMENTED_DIR)
os.makedirs(os.path.join(AUGMENTED_DIR, "positive"), exist_ok=True)
os.makedirs(os.path.join(AUGMENTED_DIR, "negative"), exist_ok=True)
shutil.copytree(os.path.join(DATASET_DIR, "positive"), os.path.join(AUGMENTED_DIR, "positive"), dirs_exist_ok=True)
shutil.copytree(os.path.join(DATASET_DIR, "negative"), os.path.join(AUGMENTED_DIR, "negative"), dirs_exist_ok=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2],
    fill_mode='nearest'
)

if positive_count == 0 or negative_count == 0:
    print("[ERROR] One of the classes is empty. Exiting.")
    raise SystemExit

if positive_count < negative_count:
    target_class = "positive"
    augment_count = (negative_count - positive_count) // positive_count if positive_count > 0 else 2
else:
    target_class = "negative"
    augment_count = (positive_count - negative_count) // negative_count if negative_count > 0 else 2
augment_count = max(2, min(augment_count, 5))

target_dir = os.path.join(DATASET_DIR, target_class)
output_dir = os.path.join(AUGMENTED_DIR, target_class)

for filename in os.listdir(target_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue
    img_path = os.path.join(target_dir, filename)
    try:
        img = load_img(img_path, target_size=TARGET_SIZE)
        img = img.convert('RGB')
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        i = 0
        base_name = os.path.splitext(filename)[0]
        for batch in datagen.flow(x, batch_size=1):
            aug_filename = f"{base_name}_aug_{i}.jpg"
            aug_path = os.path.join(output_dir, aug_filename)
            img_aug = tf.keras.preprocessing.image.array_to_img(batch[0])
            img_aug.save(aug_path)
            i += 1
            if i >= augment_count:
                break
    except Exception:
        pass

DATASET_DIR = AUGMENTED_DIR

positive_count_aug = len([f for f in os.listdir(os.path.join(DATASET_DIR, "positive")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
negative_count_aug = len([f for f in os.listdir(os.path.join(DATASET_DIR, "negative")) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
print(f"[INFO] After augmentation:")
print(f"Positive class: {positive_count_aug} images")
print(f"Negative class: {negative_count_aug} images\n")

print("[INFO] Preparing data...")
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)
val_datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='training',
    shuffle=True
)
validation_generator = val_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    subset='validation',
    shuffle=False
)
print(f"Training images: {train_generator.samples}")
print(f"Validation images: {validation_generator.samples}\n")

class_weights = compute_class_weight(
    'balanced',
    classes=np.unique(train_generator.classes),
    y=train_generator.classes
)
class_weight_dict = dict(enumerate(class_weights))
print(f"[INFO] Class weights: {class_weight_dict}\n")

print("[INFO] Creating Xception model...")
input_shape = TARGET_SIZE + (3,)
base_model = Xception(include_top=False, weights='imagenet', input_shape=input_shape)
base_model.trainable = False
x = base_model.output
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
output = layers.Dense(1, activation='sigmoid')(x)
model = models.Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
total_params = model.count_params()
trainable_params = int(np.sum([np.prod(w.shape.as_list()) for w in model.trainable_weights])) if model.trainable_weights else 0
print(f"[INFO] Total parameters: {total_params:,}")
print(f"[INFO] Trainable parameters: {trainable_params:,}\n")

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(MODEL_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

print("\n" + "=" * 60)
print("[INFO] PHASE 1: Training with frozen base model")
print("=" * 60 + "\n")

history_frozen = model.fit(
    train_generator,
    epochs=EPOCHS_FROZEN,
    validation_data=validation_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

print("\n" + "=" * 60)
print("[INFO] PHASE 2: Fine-tuning (unfreezing last layers)")
print("=" * 60 + "\n")

base_model.trainable = True
fine_tune_at = max(0, len(base_model.layers) - 20)
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
trainable_params_after = int(np.sum([np.prod(w.shape.as_list()) for w in model.trainable_weights])) if model.trainable_weights else 0
print(f"[INFO] Trainable parameters after unfreezing: {trainable_params_after:,}\n")

history_fine = model.fit(
    train_generator,
    epochs=EPOCHS_FINE_TUNE,
    validation_data=validation_generator,
    initial_epoch=len(history_frozen.history['loss']),
    class_weight=class_weight_dict,
    callbacks=callbacks,
    verbose=1
)

history_combined = {}
for key in history_frozen.history.keys():
    history_combined[key] = history_frozen.history[key] + history_fine.history.get(key, [])

print("\n[INFO] Visualizing results...")
plt.figure(figsize=(16, 5))

plt.subplot(1, 3, 1)
plt.plot(history_combined.get('accuracy', []), label="Training Accuracy", linewidth=2)
plt.plot(history_combined.get('val_accuracy', []), label="Validation Accuracy", linewidth=2)
plt.axvline(x=EPOCHS_FROZEN, color='red', linestyle='--', label='Fine-tuning Start')
plt.title("Accuracy over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 2)
plt.plot(history_combined.get('loss', []), label="Training Loss", linewidth=2)
plt.plot(history_combined.get('val_loss', []), label="Validation Loss", linewidth=2)
plt.axvline(x=EPOCHS_FROZEN, color='red', linestyle='--', label='Fine-tuning Start')
plt.title("Loss over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 3, 3)
plt.plot(history_combined.get('auc', []), label="Training AUC", linewidth=2)
plt.plot(history_combined.get('val_auc', []), label="Validation AUC", linewidth=2)
plt.axvline(x=EPOCHS_FROZEN, color='red', linestyle='--', label='Fine-tuning Start')
plt.title("AUC over Epochs", fontsize=14, fontweight='bold')
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('/kaggle/working/training_history.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("[INFO] MODEL EVALUATION ON VALIDATION SET")
print("=" * 60 + "\n")

if os.path.exists(MODEL_PATH):
    model.load_weights(MODEL_PATH)

Y_pred = model.predict(validation_generator, verbose=1)
y_pred = (Y_pred > 0.5).astype(int).flatten()
cm = confusion_matrix(validation_generator.classes, y_pred)
print("\nConfusion Matrix:")
print(cm)

tn, fp, fn, tp = cm.ravel()
accuracy = (tp + tn) / (tp + tn + fp + fn)
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n{'=' * 50}")
print(f"Accuracy:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")
print(f"{'=' * 50}\n")

print("\nClassification Report:")
print(classification_report(validation_generator.classes, y_pred, target_names=['Negative', 'Positive']))

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"], cbar_kws={'label': 'Count'})
plt.ylabel("Actual", fontsize=12, fontweight='bold')
plt.xlabel("Predicted", fontsize=12, fontweight='bold')
plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n[INFO] Analyzing random images from validation set...")
val_images = []
val_labels = []
for fname, label in zip(validation_generator.filenames, validation_generator.classes):
    img_path = os.path.join(DATASET_DIR, fname)
    val_images.append(img_path)
    val_labels.append(label)
samples = random.sample(list(zip(val_images, val_labels)), min(10, len(val_images)))

plt.figure(figsize=(20, 8))
for i, (img_path, true_label) in enumerate(samples):
    try:
        img = load_img(img_path, target_size=TARGET_SIZE)
        img = img.convert('RGB')
    except Exception:
        continue
    img_array = img_to_array(img) / 255.0
    prediction = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0]
    pred_label = 1 if prediction > 0.5 else 0
    is_correct = pred_label == true_label
    border_color = 'green' if is_correct else 'red'
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.axis("off")
    title_text = f"True: {'Positive' if true_label == 1 else 'Negative'}\n"
    title_text += f"Pred: {'Positive' if pred_label == 1 else 'Negative'}\n"
    title_text += f"Conf: {prediction:.3f}"
    plt.title(title_text, color=border_color, fontweight='bold')
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(3)
plt.tight_layout()
plt.savefig('/kaggle/working/sample_predictions.png', dpi=300, bbox_inches='tight')
plt.show()

def smooth_predictions(predictions, window_size=7, threshold=0.5):
    smoothed = []
    for i in range(len(predictions)):
        window = predictions[max(0, i - window_size // 2):i + window_size // 2 + 1]
        avg = np.mean(window) if len(window) > 0 else 0
        smoothed.append(1 if avg > threshold else 0)
    return smoothed

def process_video_frame_by_frame(video_path, model, target_size):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Failed to open video!")
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_predictions = []
    start_time = time.time()
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        img_array = img_to_array(Image.fromarray(img).convert('RGB')) / 255.0
        pred = model.predict(np.expand_dims(img_array, axis=0), verbose=0)[0][0]
        frame_predictions.append(pred)
    cap.release()
    total_time = time.time() - start_time
    return frame_predictions, total_time

def process_video_batch(video_path, model, target_size, batch_size=32, skip_frames=3):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[ERROR] Failed to open video!")
        return None, None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_predictions = []
    frame_batch = []
    start_time = time.time()
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % skip_frames != 0:
            continue
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        frame_batch.append(np.array(Image.fromarray(img).convert('RGB')) / 255.0)
        if len(frame_batch) == batch_size:
            predictions = model.predict(np.array(frame_batch), verbose=0)
            frame_predictions.extend(predictions.flatten().tolist())
            frame_batch = []
    if len(frame_batch) > 0:
        predictions = model.predict(np.array(frame_batch), verbose=0)
        frame_predictions.extend(predictions.flatten().tolist())
    cap.release()
    total_time = time.time() - start_time
    return frame_predictions, total_time

print("\n" + "=" * 60)
print("[INFO] INVESTIGATING VIDEO PROCESSING SPEED AND ACCURACY")
print("=" * 60 + "\n")

frame_preds_frame_by_frame, time_frame_by_frame = process_video_frame_by_frame(VIDEO_PATH, model, TARGET_SIZE)
frame_preds_batch, time_batch = process_video_batch(VIDEO_PATH, model, TARGET_SIZE, batch_size=32, skip_frames=3)
if frame_preds_batch is None or frame_preds_frame_by_frame is None:
    print("[INFO] Video processing skipped due to missing video or read error.")
else:
    smoothed_batch_preds = smooth_predictions(frame_preds_batch, window_size=7, threshold=0.5)
    print(f"Processing time (frame by frame): {time_frame_by_frame:.2f} sec")
    print(f"Processing time (batches): {time_batch:.2f} sec")
    speedup = time_frame_by_frame / time_batch if time_batch and time_batch > 0 else 0
    print(f"Speedup: {speedup:.2f}x")
    plt.figure(figsize=(14, 6))
    plt.plot(frame_preds_batch, label="Original predictions (batches)", alpha=0.7)
    plt.plot(smoothed_batch_preds, label="Post-processed predictions", linewidth=2)
    plt.title("Comparison of predictions with and without post-processing")
    plt.xlabel("Frames")
    plt.ylabel("Probability")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()
    print("\n[INFO] Investigation completed.")
    print(f"Video processing time frame by frame: {time_frame_by_frame:.2f} sec")
    print(f"Video processing time in batches: {time_batch:.2f} sec")
    print(f"Speedup: {speedup:.2f}x")
