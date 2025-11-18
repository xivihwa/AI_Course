import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.get_logger().setLevel('ERROR')

train_path = "/kaggle/input/lab7-data/yelp_review_polarity_csv/yelp_review_polarity_csv/train.csv"
test_path = "/kaggle/input/lab7-data/yelp_review_polarity_csv/yelp_review_polarity_csv/test.csv"

train_df = pd.read_csv(train_path, header=None, names=["label", "text"])
test_df = pd.read_csv(test_path, header=None, names=["label", "text"])

print("\n=== DATA INSPECTION ===")
print("First 3 rows of training dataset:")
print(train_df.head(3))
print(f"Unique labels: {train_df['label'].unique()}")
print(f"Training samples: {train_df.shape[0]}, Testing samples: {test_df.shape[0]}")

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

vocab_size = 10000
max_sequence_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

x_train = tokenizer.texts_to_sequences(train_df["text"])
x_test = tokenizer.texts_to_sequences(test_df["text"])

x_train = pad_sequences(x_train, maxlen=max_sequence_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_sequence_length, padding='post', truncating='post')

y_train = np.array(train_df["label"]) - 1
y_test = np.array(test_df["label"]) - 1

print("\nData preprocessing complete.")
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n=== MODEL SUMMARY ===")
model.summary()

print("\nTraining LSTM model...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

print("\nEvaluating model on test data...")
y_pred = (model.predict(x_test) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== MODEL EVALUATION RESULTS ===")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n=== SENTIMENT ANALYSIS TESTING ===")

sample_reviews = [
    "This restaurant was amazing! The food was delicious and the service was great.",
    "Terrible experience, I will never come back again.",
    "The movie was just okay, nothing special.",
    "Absolutely loved this place, 10/10!",
    "Worst food ever. Very disappointed."
]

sample_sequences = tokenizer.texts_to_sequences(sample_reviews)
sample_sequences = pad_sequences(sample_sequences, maxlen=max_sequence_length, padding='post')

predictions = model.predict(sample_sequences, verbose=0)
for i, (review, prob) in enumerate(zip(sample_reviews, predictions), 1):
    sentiment = "Positive" if prob > 0.5 else "Negative"
    confidence = prob[0] if prob > 0.5 else 1 - prob[0]
    print(f"{i}. Review: {review}")
    print(f"   Sentiment: {sentiment} (Confidence: {confidence:.2%})")

print("\n=== WORK COMPLETED ===")
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging

logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("absl").setLevel(logging.ERROR)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

tf.get_logger().setLevel('ERROR')

train_path = "/kaggle/input/lab7-data/yelp_review_polarity_csv/yelp_review_polarity_csv/train.csv"
test_path = "/kaggle/input/lab7-data/yelp_review_polarity_csv/yelp_review_polarity_csv/test.csv"

train_df = pd.read_csv(train_path, header=None, names=["label", "text"])
test_df = pd.read_csv(test_path, header=None, names=["label", "text"])

print("\n=== DATA INSPECTION ===")
print("First 3 rows of training dataset:")
print(train_df.head(3))
print(f"Unique labels: {train_df['label'].unique()}")
print(f"Training samples: {train_df.shape[0]}, Testing samples: {test_df.shape[0]}")

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)

vocab_size = 10000
max_sequence_length = 200

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_df["text"])

x_train = tokenizer.texts_to_sequences(train_df["text"])
x_test = tokenizer.texts_to_sequences(test_df["text"])

x_train = pad_sequences(x_train, maxlen=max_sequence_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_sequence_length, padding='post', truncating='post')

y_train = np.array(train_df["label"]) - 1
y_test = np.array(test_df["label"]) - 1

print("\nData preprocessing complete.")
print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

model = Sequential([
    Embedding(vocab_size, 128),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print("\n=== MODEL SUMMARY ===")
model.summary()

print("\nTraining LSTM model...")
history = model.fit(
    x_train, y_train,
    epochs=5,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

print("\nEvaluating model on test data...")
y_pred = (model.predict(x_test) > 0.5).astype("int32")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n=== MODEL EVALUATION RESULTS ===")
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-score:  {f1:.4f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Negative', 'Positive'],
    yticklabels=['Negative', 'Positive']
)
plt.title("Confusion Matrix", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.show()

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy', linewidth=2)
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss', linewidth=2)
plt.plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

print("\n=== SENTIMENT ANALYSIS TESTING ===")

sample_reviews = [
    "This restaurant was amazing! The food was delicious and the service was great.",
    "Terrible experience, I will never come back again.",
    "The movie was just okay, nothing special.",
    "Absolutely loved this place, 10/10!",
    "Worst food ever. Very disappointed."
]

sample_sequences = tokenizer.texts_to_sequences(sample_reviews)
sample_sequences = pad_sequences(sample_sequences, maxlen=max_sequence_length, padding='post')

predictions = model.predict(sample_sequences, verbose=0)
for i, (review, prob) in enumerate(zip(sample_reviews, predictions), 1):
    sentiment = "Positive" if prob > 0.5 else "Negative"
    confidence = prob[0] if prob > 0.5 else 1 - prob[0]
    print(f"{i}. Review: {review}")
    print(f"   Sentiment: {sentiment} (Confidence: {confidence:.2%})")

print("\n=== WORK COMPLETED ===")
