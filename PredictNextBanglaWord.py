import numpy as np
import PySimpleGUI as sg
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load the dataset and preprocess it
with open("bangla dataset.txt", "r", encoding="utf-8") as file:
    text = file.read()

# Tokenize the text into words
words = text.split()

# Create sequences of words
seq_length = 5  # Adjust this based on your preference
sequences = []
for i in range(len(words) - seq_length):
    sequences.append(words[i:i+seq_length])

# Create word-to-index and index-to-word mappings
word_to_index = {word: idx for idx, word in enumerate(set(words))}
index_to_word = {idx: word for word, idx in word_to_index.items()}

# Encode words as numerical values (one-hot encoding or embeddings)
vocab_size = len(word_to_index)
X = np.zeros((len(sequences), seq_length, vocab_size), dtype=bool)
y = np.zeros((len(sequences), vocab_size), dtype=bool)

for i, sequence in enumerate(sequences):
    for t, word in enumerate(sequence):
        X[i, t, word_to_index[word]] = 1
    y[i, word_to_index[sequences[i][-1]]] = 1

# Build and train the RNN model
model = Sequential([
    LSTM(128, input_shape=(seq_length, vocab_size)),
    Dense(vocab_size, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(X, y, epochs=100, batch_size=128)

# Define the PySimpleGUI layout
layout = [
    [sg.Text("Enter a text sequence:"), sg.InputText(key="-INPUT-")],
    [sg.Button("Generate Suggestions"), sg.Button("Exit")],
    [sg.Text("Suggestions:")],
    [sg.Listbox([], size=(80,15), key="-SUGGESTIONS-", enable_events=True)]
]

window = sg.Window("Next Bengali Word Suggestion", layout)

def generate_suggestions(input_text, num_suggestions=5):
    input_sequence = input_text.split()[-seq_length:]
    encoded_input = np.zeros((1, seq_length, vocab_size), dtype=bool)

    for t, word in enumerate(input_sequence):
        if word in word_to_index:
            encoded_input[0, t, word_to_index[word]] = 1
        else:
            # Handle the case where the word is not in the dataset
            # You can skip it or use a default value as needed
            pass

    predictions = model.predict(encoded_input)[0]
    top_indices = np.argsort(predictions)[-num_suggestions:][::-1]
    suggestions = [index_to_word[idx] for idx in top_indices]

    return suggestions

while True:
    event, values = window.read()

    if event == sg.WIN_CLOSED or event == "Exit":
        break

    if event == "Generate Suggestions":
        input_text = values["-INPUT-"]
        suggestions = generate_suggestions(input_text)
        window["-SUGGESTIONS-"].update(suggestions)

window.close()
