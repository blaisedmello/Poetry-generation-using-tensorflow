import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import os
import string

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *

# Reading Data from the folders and populating the corpus list

root_dir = "./forms"
corpus = []
corpus_size = 10000
done = False

print("Loading Poems in corpus..\n")
for dirname, _, filenames in os.walk(root_dir):
    if done: 
        break
    #print(f"Loading {dirname}")
    for filename in filenames:
        if done: 
            break

        file_path = os.path.join(dirname, filename)

        if filename.endswith(".txt"):
            try:
                with open(os.path.join(dirname, filename), "r") as file:
                    #print(file)
                    txt = file.read()
                    for line in txt.split("\n"):
                        if done: 
                            break
                        corpus.append(line)
                        if len(corpus) == corpus_size:
                            done = True
                            break
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

# print(len(corpus))
                
# Preprocessing

# Stop word removal
def remove_punc(S):
    return S.translate(str.maketrans('', '', string.punctuation))

corpus = [remove_punc(s.lower().strip()) for s in corpus]

# print(corpus[:10])

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)

vocab_size = len(tokenizer.word_index) + 1
# print(f"Vocabulary size: {vocab_size}")

# Generate n-grams
n_grams = []
max_sequence_len = 0

for sentence in corpus:
    tokens = tokenizer.texts_to_sequences([sentence])[0]
    for i in range(1,len(tokens) + 1):
        n_gram = tokens[:i + 1]
        n_grams.append(n_gram)
        if len(n_gram) > max_sequence_len:
            max_sequence_len = len(n_gram)

# print(f"Number of N-grams: {len(n_grams)}")
# print(f"Max sequence length: {max_sequence_len}")
            
# for n in n_grams[:10]:
#     print(n)
            
# Padding the N-Grams
padded_n_grams = np.array(pad_sequences(n_grams, maxlen = 100, padding = "pre", truncating = "pre"))
# print(padded_n_grams.shape)

# for seq in padded_n_grams[:3]:
#     print(seq)

X = padded_n_grams[:, :-1]
y = padded_n_grams[:, -1]

# print(f"X: {X.shape}")
# print(f"Y: {y.shape}")

# One hot encode y
y = tf.keras.utils.to_categorical(y, num_classes = vocab_size)
# print(f"y: {y.shape}")

# Modelling
model = tf.keras.Sequential([
    Embedding(vocab_size, 300, input_length = 99),
    LSTM(150),
    Dense(vocab_size, activation = "softmax")
])

model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
model.summary()

model.fit(
    X, y, epochs = 100, batch_size = 128,
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor = "loss", patience = 20)
    ]
)

hist = model.history.history

plt.figure(figsize=(10, 8))

plt.subplot(2, 1, 1)
plt.title("Loss")
plt.plot(hist["loss"])
plt.ylabel("Loss")
plt.grid(True)

plt.subplot(2, 1, 2)
plt.title("Accuracy")
plt.plot(hist["accuracy"], color="orange")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)

plt.show()

def generate(seed_text, next_words):
    for _ in range(next_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=99, padding="pre")
        predicted = np.argmax(model.predict(token_list, verbose=0))
        output_word = ""
        for word, index in tokenizer.word_index.items():
            if index == predicted:
                output_word = word
                break
        seed_text += " " + output_word
    print(seed_text)  

generate("Hello there", 200)

generate("In a town of Athy one Jeremy Lanigan", 200)