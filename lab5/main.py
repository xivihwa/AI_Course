import os

import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, concatenate, BatchNormalization, Activation
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from IPython.display import Image as IPImage, display

tf.get_logger().setLevel('ERROR')
tf.autograph.set_verbosity(0)

CONFIG = {
    'train_path': "/kaggle/input/lab5-data/dataset/train",
    'test_path': "/kaggle/input/lab5-data/dataset/test",
    'image_size': (299, 299),
    'batch_size': 16,
    'epochs': 15,
    'learning_rate': 0.0001,
    'dropout_rate': 0.5,
    'validation_split': 0.2,
    'model_file': "inception_binary_model.keras",
    'classes_file': "class_mapping.json",
    'metrics_file': "training_metrics.json"
}

def prepare_data_generators(train_directory, config):
    data_augmentation = ImageDataGenerator(
        rescale=1.0/255.0,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest',
        validation_split=config['validation_split']
    )
    training_generator = data_augmentation.flow_from_directory(
        train_directory,
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode='binary',
        subset='training',
        shuffle=True,
        seed=42
    )
    val_datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=config['validation_split'])
    validation_generator = val_datagen.flow_from_directory(
        train_directory,
        target_size=config['image_size'],
        batch_size=config['batch_size'],
        class_mode='binary',
        subset='validation',
        shuffle=False
    )
    with open(config['classes_file'], "w", encoding="utf-8") as file:
        json.dump(training_generator.class_indices, file, ensure_ascii=False, indent=2)
    print(f"Classes: {training_generator.class_indices}")
    print(f"Training images: {training_generator.samples}")
    print(f"Validation images: {validation_generator.samples}")
    return training_generator, validation_generator

def conv2d_bn(x, filters, kernel_size, strides=1, padding='same', activation='relu', name=None):
    x = Conv2D(filters, kernel_size, strides=strides, padding=padding, use_bias=False, name=name)(x)
    x = BatchNormalization()(x)
    if activation:
        x = Activation(activation)(x)
    return x

def inception_block(x, filters_1x1, filters_3x3_reduce, filters_3x3, filters_5x5_reduce, filters_5x5, filters_pool, name=None):
    branch1x1 = conv2d_bn(x, filters_1x1, (1, 1), name=f'{name}_1x1' if name else None)
    branch3x3 = conv2d_bn(x, filters_3x3_reduce, (1, 1), name=f'{name}_3x3_reduce' if name else None)
    branch3x3 = conv2d_bn(branch3x3, filters_3x3, (3, 3), name=f'{name}_3x3' if name else None)
    branch5x5 = conv2d_bn(x, filters_5x5_reduce, (1, 1), name=f'{name}_5x5_reduce' if name else None)
    branch5x5 = conv2d_bn(branch5x5, filters_5x5, (5, 5), name=f'{name}_5x5' if name else None)
    branch_pool = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, filters_pool, (1, 1), name=f'{name}_pool' if name else None)
    x = concatenate([branch1x1, branch3x3, branch5x5, branch_pool], axis=-1, name=name)
    return x

def create_improved_inception(input_shape=(299, 299, 3), dropout=0.5):
    inputs = Input(shape=input_shape)
    x = conv2d_bn(inputs, 32, (3, 3), strides=2, name='conv1')
    x = conv2d_bn(x, 32, (3, 3), name='conv2')
    x = conv2d_bn(x, 64, (3, 3), name='conv3')
    x = MaxPooling2D((3, 3), strides=2, padding='same', name='pool1')(x)
    x = conv2d_bn(x, 80, (1, 1), name='conv4')
    x = conv2d_bn(x, 192, (3, 3), name='conv5')
    x = MaxPooling2D((3, 3), strides=2, padding='same', name='pool2')(x)
    x = inception_block(x, 64, 96, 128, 16, 32, 32, name='inception_3a')
    x = inception_block(x, 128, 128, 192, 32, 96, 64, name='inception_3b')
    x = MaxPooling2D((3, 3), strides=2, padding='same', name='pool3')(x)
    x = inception_block(x, 192, 96, 208, 16, 48, 64, name='inception_4a')
    x = inception_block(x, 160, 112, 224, 24, 64, 64, name='inception_4b')
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(dropout, name='dropout1')(x)
    x = Dense(256, activation='relu', name='fc1')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout * 0.5, name='dropout2')(x)
    x = Dense(128, activation='relu', name='fc2')(x)
    x = Dropout(dropout * 0.3, name='dropout3')(x)
    outputs = Dense(1, activation='sigmoid', name='predictions')(x)
    model = Model(inputs=inputs, outputs=outputs, name='ImprovedInception_Binary')
    return model

def train_model(model, train_gen, val_gen, config):
    model.compile(
        optimizer=Adam(learning_rate=config['learning_rate']),
        loss='binary_crossentropy',
        metrics=['accuracy', keras.metrics.Precision(name='precision'), keras.metrics.Recall(name='recall'), keras.metrics.AUC(name='auc')]
    )
    class_counts = Counter(train_gen.classes)
    max_count = max(class_counts.values())
    weights = {cls_id: float(max_count / count) for cls_id, count in class_counts.items()}
    print(f"Class weights: {weights}")
    training_callbacks = [
        EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint(config['model_file'], save_best_only=True, monitor='val_loss', mode='min', verbose=1)
    ]
    print("Training started...")
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config['epochs'],
        class_weight=weights,
        callbacks=training_callbacks,
        verbose=1
    )
    return history

def plot_training_history(history):
    if not hasattr(history, "history") or not history.history:
        print("No training history available")
        return
    plt.close('all')
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Training History', fontsize=16, fontweight='bold')
    if 'accuracy' in history.history:
        axes[0, 0].plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
        axes[0, 0].plot(history.history.get('val_accuracy', []), label='Val Accuracy', linewidth=2)
        axes[0, 0].set_title('Model Accuracy', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
    if 'loss' in history.history:
        axes[0, 1].plot(history.history['loss'], label='Train Loss', linewidth=2)
        axes[0, 1].plot(history.history.get('val_loss', []), label='Val Loss', linewidth=2)
        axes[0, 1].set_title('Model Loss', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
    if 'precision' in history.history:
        axes[0, 2].plot(history.history['precision'], label='Train Precision', linewidth=2)
        axes[0, 2].plot(history.history.get('val_precision', []), label='Val Precision', linewidth=2)
        axes[0, 2].set_title('Model Precision', fontsize=14, fontweight='bold')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Precision')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
    if 'recall' in history.history:
        axes[1, 0].plot(history.history['recall'], label='Train Recall', linewidth=2)
        axes[1, 0].plot(history.history.get('val_recall', []), label='Val Recall', linewidth=2)
        axes[1, 0].set_title('Model Recall', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Recall')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
    if 'auc' in history.history:
        axes[1, 1].plot(history.history['auc'], label='Train AUC', linewidth=2)
        axes[1, 1].plot(history.history.get('val_auc', []), label='Val AUC', linewidth=2)
        axes[1, 1].set_title('Model AUC', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('AUC')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    if 'loss' in history.history and 'val_loss' in history.history:
        epochs = range(1, len(history.history['loss']) + 1)
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
        axes[1, 2].plot(epochs, train_loss, label='Train Loss', linewidth=2)
        axes[1, 2].plot(epochs, val_loss, label='Val Loss', linewidth=2)
        if len(train_loss) > 1:
            overfitting_gap = [val - train for train, val in zip(train_loss, val_loss)]
            axes[1, 2].fill_between(epochs, train_loss, val_loss, where=[gap > 0 for gap in overfitting_gap], alpha=0.2)
        axes[1, 2].set_title('Overfitting Analysis', fontsize=14, fontweight='bold')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Loss')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=200, bbox_inches='tight')
    print("✓ Training history plot saved: training_history.png")
    try:
        display(IPImage('training_history.png'))
    except Exception as e:
        print("Could not display training history image:", e)
    plt.close()

def evaluate_and_visualize(model, val_gen, config):
    print("Model evaluation...")
    val_gen.reset()
    predictions = model.predict(val_gen, verbose=1).ravel()
    predicted_classes = (predictions > 0.5).astype(int)
    true_classes = val_gen.classes
    metrics = {
        'accuracy': accuracy_score(true_classes, predicted_classes),
        'precision': precision_score(true_classes, predicted_classes, zero_division=0),
        'recall': recall_score(true_classes, predicted_classes, zero_division=0),
        'f1_score': f1_score(true_classes, predicted_classes, zero_division=0)
    }
    print("="*50)
    print("EVALUATION RESULTS:")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print("="*50)
    class_names = list(val_gen.class_indices.keys())
    print("Classification Report:")
    print(classification_report(true_classes, predicted_classes, target_names=class_names, zero_division=0))
    plt.close('all')
    conf_matrix = confusion_matrix(true_classes, predicted_classes)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Count'})
    plt.xlabel("Predicted Class", fontsize=12)
    plt.ylabel("True Class", fontsize=12)
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=200, bbox_inches='tight')
    print("✓ Confusion matrix saved: confusion_matrix.png")
    try:
        display(IPImage('confusion_matrix.png'))
    except Exception as e:
        print("Could not display confusion matrix image:", e)
    plt.close()
    metrics_serializable = {k: float(v) for k, v in metrics.items()}
    with open(config['metrics_file'], "w") as file:
        json.dump(metrics_serializable, file, indent=2)
    return metrics

def test_on_new_images(model, test_path, config):
    if not os.path.exists(test_path):
        print(f"Test folder not found: {test_path}")
        return
    print("Testing on new images...")
    test_datagen = ImageDataGenerator(rescale=1.0/255.0)
    test_gen = test_datagen.flow_from_directory(
        test_path,
        target_size=config['image_size'],
        batch_size=1,
        class_mode=None,
        shuffle=False
    )
    predictions = model.predict(test_gen, verbose=1).ravel()
    predicted_classes = (predictions > 0.5).astype(int)
    with open(config['classes_file'], "r", encoding="utf-8") as file:
        class_mapping = json.load(file)
    inverted = {v: k for k, v in class_mapping.items()}
    correct = 0
    total = 0
    print(f"{'File':<60} {'Prediction':<20} {'Probability':<12} {'Status'}")
    print("="*110)
    for idx, filename in enumerate(test_gen.filenames):
        predicted_label = inverted.get(int(predicted_classes[idx]), str(predicted_classes[idx]))
        probability = float(predictions[idx])
        true_label = filename.split('/')[0]
        is_correct = predicted_label == true_label
        status = "✓ Correct" if is_correct else "✗ Wrong"
        if is_correct:
            correct += 1
        total += 1
        print(f"{filename:<60} {predicted_label:<20} {probability:.3f}        {status}")
    test_accuracy = correct / total if total > 0 else 0
    print("="*110)
    print(f"Test accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"Correct predictions: {correct}/{total}")

def print_model_summary(model):
    print("="*50)
    print("MODEL INFORMATION:")
    print("="*50)
    total_params = model.count_params()
    trainable_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {non_trainable_params:,}")
    print(f"Number of layers: {len(model.layers)}")
    print("="*50)

def main():
    if not os.path.exists(CONFIG['train_path']):
        raise FileNotFoundError(f"Training dataset not found: {CONFIG['train_path']}")
    print("[1/6] Preparing data...")
    train_gen, val_gen = prepare_data_generators(CONFIG['train_path'], CONFIG)
    print("[2/6] Creating improved Inception model...")
    model = create_improved_inception(input_shape=(CONFIG['image_size'][0], CONFIG['image_size'][1], 3), dropout=CONFIG['dropout_rate'])
    print_model_summary(model)
    print("[3/6] Training model...")
    history = train_model(model, train_gen, val_gen, CONFIG)
    print("[4/6] Visualizing training history...")
    plot_training_history(history)
    print("[5/6] Evaluating model...")
    metrics = evaluate_and_visualize(model, val_gen, CONFIG)
    print("[6/6] Testing on new images...")
    test_on_new_images(model, CONFIG['test_path'], CONFIG)
    model.save(CONFIG['model_file'])
    print("="*50)
    print("WORK COMPLETED SUCCESSFULLY!")
    print("="*50)
    print(f"Model saved: {CONFIG['model_file']}")
    print(f"Metrics saved: {CONFIG['metrics_file']}")
    print("="*50)

if __name__ == "__main__":
    main()
