import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, Dropout,
    BatchNormalization, Input, GlobalAveragePooling2D
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_recall_fscore_support
)
import seaborn as sns

extract_path = "/kaggle/input/animals10dataset/raw-img"  
img_size = (227, 227)
BATCH_SIZE = 32
EPOCHS = 15

TRAIN_VARIANT = True

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    extract_path,
    target_size=img_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=SEED
)

val_generator = datagen.flow_from_directory(
    extract_path,
    target_size=img_size,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False,   
    seed=SEED
)

class_names = list(train_generator.class_indices.keys())
NUM_CLASSES = len(class_names)
print("Classes:", class_names)

def build_alexnet(input_shape=(227,227,3), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(96, (11, 11), strides=4, activation='relu'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        Conv2D(256, (5, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        Conv2D(384, (3, 3), activation='relu', padding='same'),
        Conv2D(384, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        Flatten(),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(4096, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def build_alexnet_variant(input_shape=(227,227,3), num_classes=10):
    model = Sequential([
        Input(shape=input_shape),
        Conv2D(96, (11, 11), strides=4, activation='relu'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        Conv2D(256, (5, 5), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        Conv2D(384, (3, 3), activation='relu', padding='same'),
        Conv2D(384, (3, 3), activation='relu', padding='same'),
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((3, 3), strides=2),

        GlobalAveragePooling2D(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

def compile_and_train(model, train_gen, val_gen, epochs=EPOCHS, learning_rate=1e-4):
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1)
    ]
    start = time.time()
    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=2
    )
    end = time.time()
    total_time = end - start
    time_per_epoch = total_time / len(history.history['loss'])
    return history, total_time, time_per_epoch

def evaluate_model(model, val_gen):
    steps = val_gen.samples // val_gen.batch_size + int(val_gen.samples % val_gen.batch_size != 0)
    preds = model.predict(val_gen, steps=steps, verbose=0)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes  
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted', zero_division=0)
    report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    return {
        "accuracy": acc,
        "precision_weighted": precision,
        "recall_weighted": recall,
        "f1_weighted": f1,
        "report": report,
        "confusion_matrix": cm,
        "y_true": y_true,
        "y_pred": y_pred
    }

def plot_history(history, title_suffix=""):
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    losses = history.history.get('loss', [])
    val_losses = history.history.get('val_loss', [])
    epochs_range = range(len(acc))

    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, acc, label='train acc')
    plt.plot(epochs_range, val_acc, label='val acc')
    plt.legend()
    plt.title('Accuracy ' + title_suffix)

    plt.subplot(1,2,2)
    plt.plot(epochs_range, losses, label='train loss')
    plt.plot(epochs_range, val_losses, label='val loss')
    plt.legend()
    plt.title('Loss ' + title_suffix)
    plt.show()

def plot_confusion_matrix(cm, class_names, title="Confusion matrix"):
    plt.figure(figsize=(10,8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.show()

def test_on_validation_samples(model, extract_path, class_names, num_samples=10):
    """
    Test model on random images from validation set.
    These images were not used during training.
    """
    print(f"\n=== Testing on {num_samples} random images ===")
    
    all_images = []
    for class_name in os.listdir(extract_path):
        class_dir = os.path.join(extract_path, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                all_images.append((img_path, class_name))
    
    selected_images = random.sample(all_images, min(num_samples, len(all_images)))
    
    results = []
    correct = 0
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for idx, (img_path, true_class) in enumerate(selected_images):
        try:
            img = image.load_img(img_path, target_size=img_size)
            arr = image.img_to_array(img) / 255.0
            arr = np.expand_dims(arr, axis=0)
            
            pred = model.predict(arr, verbose=0)
            predicted_class = class_names[np.argmax(pred)]
            confidence = np.max(pred) * 100
            
            is_correct = (predicted_class == true_class)
            if is_correct:
                correct += 1
            
            results.append({
                'image': os.path.basename(img_path),
                'true_class': true_class,
                'predicted_class': predicted_class,
                'confidence': confidence,
                'correct': is_correct
            })
            
            axes[idx].imshow(img)
            axes[idx].axis('off')
            color = 'green' if is_correct else 'red'
            axes[idx].set_title(
                f"True: {true_class}\nPred: {predicted_class}\n{confidence:.1f}%",
                color=color,
                fontsize=10
            )
            
        except Exception as ex:
            print(f"Error processing {img_path}: {ex}")
    
    plt.tight_layout()
    plt.show()
    
    accuracy = correct / len(selected_images) * 100
    print(f"\nAccuracy on test images: {accuracy:.2f}% ({correct}/{len(selected_images)})")
    
    df_results = pd.DataFrame(results)
    print("\nDetailed results:")
    print(df_results.to_string(index=False))
    
    return results, accuracy

def save_random_predictions_to_csv(model, extract_path, num_images=1024, out_csv="/kaggle/working/classification_results.csv"):
    all_images = []
    for class_name in os.listdir(extract_path):
        class_dir = os.path.join(extract_path, class_name)
        if os.path.isdir(class_dir):
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                all_images.append((img_path, class_name))
    selected_images = random.sample(all_images, min(num_images, len(all_images)))
    results = []
    batch_size = 128
    
    def load_batch(image_data):
        images, paths, true_classes = [], [], []
        for img_path, true_class in image_data:
            img = image.load_img(img_path, target_size=img_size)
            img_array = image.img_to_array(img) / 255.0
            images.append(img_array)
            paths.append(img_path)
            true_classes.append(true_class)
        return np.array(images), paths, true_classes

    for i in range(0, len(selected_images), batch_size):
        batch_data = selected_images[i:i+batch_size]
        batch_images, batch_paths, batch_true_classes = load_batch(batch_data)
        preds = model.predict(batch_images, verbose=0)
        pred_classes = np.argmax(preds, axis=1)
        for j in range(len(batch_paths)):
            results.append([batch_paths[j], batch_true_classes[j], class_names[pred_classes[j]]])
    
    df = pd.DataFrame(results, columns=["path", "true_class", "predicted_class"])
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"Saved predictions to {out_csv}")

print("\n=== BUILD & TRAIN ORIGINAL ALEXNET ===")
orig_model = build_alexnet(input_shape=(227,227,3), num_classes=NUM_CLASSES)
print(orig_model.summary())

history_orig, total_time_orig, time_per_epoch_orig = compile_and_train(orig_model, train_generator, val_generator, epochs=EPOCHS)
print(f"Total training time (original): {total_time_orig:.1f}s, time/epoch ≈ {time_per_epoch_orig:.1f}s")
plot_history(history_orig, title_suffix="(original)")

eval_orig = evaluate_model(orig_model, val_generator)
print("=== Original AlexNet Metrics ===")
print(f"Accuracy: {eval_orig['accuracy']:.4f}")
print(f"Precision (weighted): {eval_orig['precision_weighted']:.4f}")
print(f"Recall (weighted): {eval_orig['recall_weighted']:.4f}")
print(f"F1 (weighted): {eval_orig['f1_weighted']:.4f}")
print("\nClassification Report:\n", eval_orig['report'])
plot_confusion_matrix(eval_orig['confusion_matrix'], class_names, title="Confusion Matrix (original)")

save_random_predictions_to_csv(orig_model, extract_path, num_images=1024)

print("\n=== TEST ORIGINAL MODEL ===")
test_results_orig, test_acc_orig = test_on_validation_samples(orig_model, extract_path, class_names, num_samples=10)

if TRAIN_VARIANT:
    print("\n=== BUILD & TRAIN OPTIMIZED ALEXNET ===")
    var_model = build_alexnet_variant(input_shape=(227,227,3), num_classes=NUM_CLASSES)
    print(var_model.summary())
    history_var, total_time_var, time_per_epoch_var = compile_and_train(var_model, train_generator, val_generator, epochs=EPOCHS)
    print(f"Total training time (variant): {total_time_var:.1f}s, time/epoch ≈ {time_per_epoch_var:.1f}s")
    plot_history(history_var, title_suffix="(variant)")

    eval_var = evaluate_model(var_model, val_generator)
    print("=== Optimized AlexNet Metrics ===")
    print(f"Accuracy: {eval_var['accuracy']:.4f}")
    print(f"Precision (weighted): {eval_var['precision_weighted']:.4f}")
    print(f"Recall (weighted): {eval_var['recall_weighted']:.4f}")
    print(f"F1 (weighted): {eval_var['f1_weighted']:.4f}")
    print("\nClassification Report:\n", eval_var['report'])
    plot_confusion_matrix(eval_var['confusion_matrix'], class_names, title="Confusion Matrix (variant)")

    print("\n=== TEST OPTIMIZED MODEL ===")
    test_results_var, test_acc_var = test_on_validation_samples(var_model, extract_path, class_names, num_samples=10)

    orig_params = orig_model.count_params()
    var_params = var_model.count_params()
    print("\n=== MODEL COMPARISON ===")
    print(f"Original parameters: {orig_params:,}")
    print(f"Variant parameters:  {var_params:,}")
    print(f"Parameter reduction: {orig_params - var_params:,} ({100.0*(orig_params-var_params)/orig_params:.2f}%)")
    print(f"Time/epoch original ≈ {time_per_epoch_orig:.1f}s; variant ≈ {time_per_epoch_var:.1f}s")
    print(f"Speedup: {time_per_epoch_orig/time_per_epoch_var:.2f}x")
    print(f"\nValidation Accuracy - original: {eval_orig['accuracy']:.4f}; variant: {eval_var['accuracy']:.4f}")
    print(f"Test Accuracy - original: {test_acc_orig:.2f}%; variant: {test_acc_var:.2f}%")
    print(f"F1 original: {eval_orig['f1_weighted']:.4f}; variant: {eval_var['f1_weighted']:.4f}")

print("\n=== FINISHED ===")