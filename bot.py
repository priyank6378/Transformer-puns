import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import string
import random
import re
import csv
import pickle
import os
from transformer import *

data = []
with open("jokes.csv") as f:
    x = csv.reader(f)
    f = 0;
    for i in x:
        if f:
            data.append([
                i[1], 
                '[start] ' + i[2] + ' [end]'
            ])
        f = 1;
data = np.array(data )
print(data[:2])

MAX_TOKENS = 20000
MAX_SEQUENCE_LENGTH = 100
EMBEDDING_DIM = 128
DENSE_DIM = 512
BATCH_SIZE = 16
NUM_HEADS = 8
EPOCHS = 10

strip_chars = string.punctuation + "¿"
strip_chars = strip_chars.replace("[", "") 
strip_chars = strip_chars.replace("]", "")

def custom_standardization(s):
    lowercase = tf.strings.lower(s)
    return tf.strings.regex_replace(lowercase, f'[{re.escape(strip_chars)}]', '')

question_vectorizer = keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_SEQUENCE_LENGTH,
    standardize=custom_standardization,
)

answer_vectorizer = keras.layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=MAX_SEQUENCE_LENGTH + 1,
    standardize=custom_standardization,
)

question_data = data[:, 0]
answer_data = data[:, 1]
question_vectorizer.adapt(question_data)
answer_vectorizer.adapt(answer_data)


def prepare_dataset(questions, answers):
    tokenized_questions = question_vectorizer(questions)
    tokenized_answers = answer_vectorizer(answers)
    x = {"question": tokenized_questions, "answer": tokenized_answers[:, :-1]}
    y = tokenized_answers[: , 1:]
    return (x, y)

full_data_ds = tf.data.Dataset.from_tensor_slices((question_data, answer_data))
full_data_ds = full_data_ds.batch(BATCH_SIZE)
full_data_ds = full_data_ds.shuffle(buffer_size=4096)
full_data_ds = full_data_ds.map(prepare_dataset, num_parallel_calls=8)
full_data_ds = full_data_ds.prefetch(16).cache()   

# remove from memory
del data
del question_data
del answer_data

encoder_inputs = keras.Input(shape=(None,), dtype='int64', name='question')
x = PositionalEmbedding(MAX_SEQUENCE_LENGTH, MAX_TOKENS, EMBEDDING_DIM)(encoder_inputs)
encoder_outputs = TransformerEncoder(EMBEDDING_DIM, DENSE_DIM, NUM_HEADS)(x)

decoder_inputs = keras.Input(shape=(None,), dtype='int64', name='answer')
x = PositionalEmbedding(MAX_SEQUENCE_LENGTH, MAX_TOKENS, EMBEDDING_DIM)(decoder_inputs)
x = TransformerDecoder(EMBEDDING_DIM, DENSE_DIM, NUM_HEADS)(x, encoder_outputs)
x = layers.Dropout(0.5)(x)
decoder_outputs = layers.Dense(MAX_TOKENS, activation='softmax')(x)

model = keras.Model(
    inputs = [encoder_inputs, decoder_inputs],
    outputs = decoder_outputs
)

model.load_weights("training_checkpoint/puns_bot.keras")

target_dict = dict([(i, val) for i, val in enumerate(answer_vectorizer.get_vocabulary())])

def generate_text(question):
    question = question_vectorizer([question])
    output = "[start]"
    answer = ""
    for i in range(MAX_SEQUENCE_LENGTH):
        x = answer_vectorizer([output])[:, :-1]
        x = model([question, x])
        x = x[:, i, :]
        x = tf.argmax(x, axis=-1)
        x = target_dict[x.numpy()[0]]
        output += " " + x
        if x == "[end]":
            break
        answer += " " + x
    return answer

os.system("clear")
while (1):    
    question = input("Enter question (Leave empty to exit): ")
    if (question == ""):
        break
    print(generate_text(question))
    print()

