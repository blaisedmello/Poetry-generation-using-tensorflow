import numpy as np
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import random
import os

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import *

root_dir = "./forms"
corpus = []
corpus_size = 10000
done = False

print("Loading Poems in corpus..\n")
for dirname, _, filenames in os.walk(root_dir):
    if done: 
        break
    print(f"Loading {dirname}")
    for filename in filenames:
        if done: 
            break

        file_path = os.path.join(dirname, filename)

        if filename.endswith(".txt"):
            try:
                with open(os.path.join(dirname, filename), "r") as file:
                    print(file)
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

print(len(corpus))