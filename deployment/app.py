import numpy as np
import tensorflow as tf
import streamlit as st
import numpy as np
import re
import pickle
import tensorflow as tf
import random
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, concatenate, MaxPool1D, Dropout, GlobalMaxPool1D, BatchNormalization, Input, Flatten, Conv1D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2, l1, l1_l2

tf.config.experimental.allow_growth = True

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)  # suppress warnings

#defining required variables
pad_type = 'post'
trunc_type = 'post'
maxlen1 = 78
maxlen2 = 57
vocab_size_1 = 17997
vocab_size_2 = 30076
emb_dim = 300

# loading required files

with open('tokenizer1.pickle', 'rb') as handle:
    tokenizer1 = pickle.load(handle)

with open('tokenizer2.pickle', 'rb') as handle:
    tokenizer2 = pickle.load(handle)

# Function to preprocess sentences
def decontracted(phrase):
    # specific
    phrase = re.sub(r"won't", "will not", phrase)
    phrase = re.sub(r"can\'t", "can not", phrase)

    # general
    phrase = re.sub(r"n\'t", " not", phrase)
    phrase = re.sub(r"\'re", " are", phrase)
    phrase = re.sub(r"\'s", " is", phrase)
    phrase = re.sub(r"\'d", " would", phrase)
    phrase = re.sub(r"\'ll", " will", phrase)
    phrase = re.sub(r"\'t", " not", phrase)
    phrase = re.sub(r"\'ve", " have", phrase)
    phrase = re.sub(r"\'m", " am", phrase)
    return phrase

def preprocess_text(text_data):
    preprocessed_text = []
    for sentence in text_data:
        if type(sentence) != str:
            sent = ' '
        else:
            sent = decontracted(sentence)
            sent = sent.replace('\\r', ' ')
            sent = sent.replace('\\n', ' ')
            sent = sent.replace('\\"', ' ')
            sent = re.sub('[^A-Za-z0-9]+', ' ', sent)
        # https://gist.github.com/sebleier/554280
        sent = ' '.join(e for e in sent.split())
        preprocessed_text.append(sent.lower().strip())
    return preprocessed_text


def get_model():
    # custom self attention class
    class Attention_Custom(tf.keras.layers.Layer):

        def __init__(self, att_units, inp_shape):
            super().__init__()
            self.att_units = att_units
            self.inp_shape = inp_shape
            self.num_dim = self.inp_shape[-1]  ##dimensions per time step
            self.ts = self.inp_shape[-2]  ##number of timesteps
            self.w = self.add_weight(shape=(self.num_dim, self.att_units), initializer='normal', trainable=True)
            self.b = self.add_weight(shape=(self.ts, self.att_units), initializer='zero', trainable=True)
            self.v = self.add_weight(shape=(self.att_units, self.num_dim), initializer='normal', trainable=True)
            self.v_b = self.add_weight(shape=(self.ts, self.num_dim))

        def call(self, x):
            score = K.dot((K.tanh(K.dot(x, self.w) + self.b)), self.v) + self.v_b
            weights = K.softmax(score, axis=1)
            weighted_states = x * weights
            output = K.sum(weighted_states, axis=1)

            return weights, output

    # defining the model

    # getting sentence 1 representations
    input1 = Input(shape=(maxlen1,))  # for sentence1 embeddings
    emb1 = Embedding(input_dim=vocab_size_1, output_dim=emb_dim, input_length=maxlen1, trainable=False)(input1)

    lstm1 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, kernel_regularizer=l2(1e-5)),
                          merge_mode='concat')(emb1)
    att_1 = Attention_Custom(512, lstm1.shape)
    att_weights_1, context_1 = att_1.call(lstm1)
    s1_dense = Dense(units=300, activation='relu')(context_1)

    # getting sentence2 representations
    input2 = Input(shape=(maxlen2,))  # for sentence2 embeddings
    emb2 = Embedding(input_dim=vocab_size_2, output_dim=emb_dim, input_length=maxlen2, trainable=False)(input2)

    lstm2 = Bidirectional(LSTM(256, return_sequences=True, dropout=0.3, kernel_regularizer=l2(1e-5)),
                          merge_mode='concat')(emb2)
    att_2 = Attention_Custom(512, lstm2.shape)
    att_weights_2, context_2 = att_2.call(lstm2)
    s2_dense = Dense(units=300, activation='relu')(context_2)

    # merging sentence1 and sentence2 representations
    merged_representations = concatenate([s1_dense, s2_dense])
    # point-wise product of sentence representations
    prod = tf.math.multiply(s1_dense, s2_dense)
    # point-wise absolute difference of sentence representations
    subtracted = tf.keras.layers.Subtract()([s1_dense, s2_dense])
    abs_diff = tf.math.abs(subtracted)

    # concatenating
    concatenated = concatenate([merged_representations, prod, abs_diff])

    # MLP and threeway classifier
    dense1 = Dense(units=512, activation='relu')(concatenated)
    dense2 = Dense(units=128, activation='relu')(dense1)
    output = Dense(units=3, activation='softmax')(dense2)
    model = Model(inputs=[input1, input2], outputs=output)

    return model



#Headings for Web Application
st.title("Natural Language Inferencing")

#Textbox for text user is entering
#st.subheader("Enter the first sentence.")
sentence1 = st.text_input('Enter the first sentence:') #text is stored in this variable

#st.subheader("Enter the second sentence.")
sentence2 = st.text_input('Enter the second sentence:') #text is stored in this variable


def predict(sentence1, sentence2):
    # basic preprocessing
    preprocessed = preprocess_text([sentence1, sentence2])
    preprocessed_sentence1 = [preprocessed[0]]
    preprocessed_sentence2 = [preprocessed[1]]

    # Encode sentence 1 into sequence of token ids
    sentence1_sequence = tokenizer1.texts_to_sequences(preprocessed_sentence1)

    # Padding the encoded sentence1
    padded_sentence_1 = pad_sequences(sentence1_sequence, padding=pad_type, truncating=trunc_type, maxlen=maxlen1)

    # Encode sentence2 into sequence of token ids
    sentence2_sequence = tokenizer2.texts_to_sequences(preprocessed_sentence2)

    # Padding the the encoded sentence2
    padded_sentence_2 = pad_sequences(sentence2_sequence, padding=pad_type, truncating=trunc_type, maxlen=maxlen2)

    # defining the model architecture
    model = get_model()

    # loading the weights into the model
    model.load_weights("final_model.h5")

    # making prediction
    pred_proba = model.predict([padded_sentence_1, padded_sentence_2])

    label = pred_proba.argmax()

    if label == 0:
        prediction = "neutral"
    elif label == 1:
        prediction = "contradiction"
    elif label == 2:
        prediction = "entailment"

    return prediction

pred = predict(sentence1, sentence2)

st.write("The prediction is:", pred)

