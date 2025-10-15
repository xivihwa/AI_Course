import numpy as np
import pandas as pd
from keras import Model, Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, Input, Add
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def generate_data(n_samples=1000):
    X = np.random.uniform(0, 10, (n_samples, 2))
    Y = X[:, 0]**3 + X[:, 1]**3
    return X, Y

scaler_X = MinMaxScaler()
scaler_Y = MinMaxScaler()

X, Y = generate_data()
Y = Y.reshape(-1, 1)

X_scaled = scaler_X.fit_transform(X)
Y_scaled = scaler_Y.fit_transform(Y)

X_train, X_test = X_scaled[:800], X_scaled[800:]
Y_train, Y_test = Y_scaled[:800], Y_scaled[800:]

def train_and_evaluate(model, X_train, Y_train, X_test, Y_test, epochs=200, batch_size=10):
    model.compile(optimizer='adam', loss='mse')
    history = model.fit(
        X_train, Y_train,
        epochs=epochs,
        batch_size=batch_size,
        verbose=0,
        validation_data=(X_test, Y_test)
    )
    Y_pred = model.predict(X_test)

    Y_test_orig = scaler_Y.inverse_transform(Y_test)
    Y_pred_orig = scaler_Y.inverse_transform(Y_pred)

    error = np.mean(np.abs((Y_test_orig - Y_pred_orig.flatten()) / Y_test_orig))
    r2 = r2_score(Y_test_orig, Y_pred_orig)
    return history, error, Y_pred_orig, r2, Y_test_orig

inputs = Input(shape=(2,))
hidden1 = Dense(20, activation='relu')(inputs)
output = Dense(1)(hidden1)
output_cascade = Dense(1)(inputs)
final_output = Add()([output, output_cascade])
cascade_model_1 = Model(inputs=inputs, outputs=final_output)

inputs_2 = Input(shape=(2,))
hidden1_2 = Dense(10, activation='relu')(inputs_2)
hidden2_2 = Dense(10, activation='relu')(hidden1_2)
output_2 = Dense(1)(hidden2_2)
output_cascade_2 = Dense(1)(inputs_2)
final_output_2 = Add()([output_2, output_cascade_2])
cascade_model_2 = Model(inputs=inputs_2, outputs=final_output_2)

models = {
    "FeedForward (10 нейронів)": Sequential([
        Input(shape=(2,)),
        Dense(10, activation='tanh'),
        Dense(1)
    ]),
    "FeedForward (20 нейронів)": Sequential([
        Input(shape=(2,)),
        Dense(20, activation='tanh'),
        Dense(1)
    ]),
    "FeedForward (3x32 нейрони)": Sequential([
        Input(shape=(2,)),
        Dense(32, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(32, activation='tanh'),
        Dense(1)
    ]),
    "Cascade (20 нейронів)": cascade_model_1,
    "Cascade (2x10 нейронів)": cascade_model_2,
    "Elman (15 нейронів)": Sequential([
        Input(shape=(2, 1)),
        SimpleRNN(15, activation='tanh'),
        Dense(1)
    ]),
    "Elman (3x5 нейронів)": Sequential([
        Input(shape=(2, 1)),
        SimpleRNN(5, activation='tanh', return_sequences=True),
        SimpleRNN(5, activation='tanh', return_sequences=True),
        SimpleRNN(5, activation='tanh'),
        Dense(1)
    ])
}

errors = {}
predictions = {}
r2_scores = {}
plt.figure(figsize=(10, 6))

for name, model in models.items():
    print(f"Навчання {name}...")
    if "Elman" in name:
        X_train_rnn = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        X_test_rnn = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        history, error, Y_pred, r2, Y_test_orig = train_and_evaluate(model, X_train_rnn, Y_train, X_test_rnn, Y_test)
    else:
        history, error, Y_pred, r2, Y_test_orig = train_and_evaluate(model, X_train, Y_train, X_test, Y_test)
    errors[name] = error
    predictions[name] = Y_pred
    r2_scores[name] = r2
    plt.plot(history.history['loss'], label=name)

plt.title("Зміна MSE під час навчання")
plt.xlabel("Епохи")
plt.ylabel("MSE")
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
for name, Y_pred in predictions.items():
    plt.scatter(Y_test_orig, Y_pred, label=name, alpha=0.5)

plt.plot(Y_test_orig, Y_test_orig, 'k--', label="Ідеальна лінія")
plt.xlabel("Реальні значення")
plt.ylabel("Передбачені значення")
plt.legend()
plt.title("Порівняння передбачень різних моделей")
plt.show()

results = pd.DataFrame({
    "Model": list(errors.keys()),
    "Relative Error": list(errors.values()),
    "R2 Score": list(r2_scores.values())
})

print("\nПідсумкові результати:")
print(results.sort_values("R2 Score", ascending=False))
