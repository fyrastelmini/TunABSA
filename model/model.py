import numpy as np
import pandas as pd
import ast
from google.colab import drive
import transformers as tr
import tensorflow as tf
from tqdm import tqdm

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense,Softmax,Flatten, Attention, Embedding, SpatialDropout1D, Bidirectional, GRU, Dropout, Concatenate, GlobalAveragePooling1D, GlobalMaxPooling1D, Layer,LayerNormalization, Masking
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras import layers, models, preprocessing
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras_self_attention import SeqSelfAttention
from tensorflow.keras.losses import SparseCategoricalCrossentropy, KLDivergence, Hinge, SquaredHinge,MeanSquaredError
from tensorflow.keras import metrics
tokenizer = tr.BertTokenizer.from_pretrained("ziedsb19/tunbert_zied")

class BiGRU_pretrain:
    def __init__(self, vocab_size=vocab_size, embedding_dim=32, gru_units=16):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.model = None
    
    def build_model(self):
        initializer=kernel_initializer=tf.keras.initializers.GlorotNormal()
        input = Input(shape=(None,))
        x = Embedding(self.vocab_size, self.embedding_dim, input_length=None)(input)
        x = Dropout(0.2)(x)
        x = Bidirectional(GRU(self.gru_units, return_sequences=True))(x)
        subject_output = Dense(12, activation='sigmoid',kernel_initializer=initializer)(x)
        polarized_output = Dense(12, activation='sigmoid',kernel_initializer=initializer)(x)
        subject_output = Dense(1, activation='sigmoid', name='subject_output',kernel_initializer=initializer)(subject_output)
        polarized_output = Dense(1, activation='sigmoid', name='polarized_output',kernel_initializer=initializer)(polarized_output)
        self.model = Model(inputs=input, outputs=[subject_output, polarized_output])
    
    def compile_model(self):
        adam = Adam(learning_rate=0.001)
        jaccard_similarity = metrics.BinaryAccuracy(threshold=0.5)
        self.model.compile(loss="binary_crossentropy", optimizer=adam, metrics=[jaccard_similarity])
    
    def train_model(self, X_train, Y, batch_size=32, epochs=10, validation_split=0.2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',patience=3,restore_best_weights=True)
        self.model.fit(X_train, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,callbacks=[early_stopping])
    
    def predict(self, X):
        return self.model.predict(X)

class BiGRU_attention:
    def __init__(self, vocab_size=vocab_size, embedding_dim=32, gru_units=16):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.gru_units = gru_units
        self.model = None
    
    def build_model(self):
        input = Input(shape=(None,))
        x = Embedding(self.vocab_size, self.embedding_dim, input_length=None)(input)
        x = Dropout(0.2)(x)
        x = Bidirectional(GRU(self.gru_units, return_sequences=True))(x)
        x = Dropout(0.2)(x)
        x = SeqSelfAttention()(x)
        x = LayerNormalization()(x)
        x = Dropout(0.2)(x)
        output = Dense(1, activation='sigmoid', name='output',kernel_initializer=tf.keras.initializers.GlorotNormal())(x)
        self.model = Model(inputs=input, outputs=output)
    
    def compile_model(self,lr=0.0001):
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.model.compile(loss='binary_crossentropy', optimizer=optimizer,  metrics=["accuracy"])
    
    def train_model(self, X_train, Y, batch_size=32, epochs=10, validation_split=0.2):
        early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',patience=5,restore_best_weights=True)
        self.model.fit(X_train, Y, batch_size=batch_size, epochs=epochs, validation_split=validation_split,callbacks=[early_stopping])
    
    def predict(self, X):
        return self.model.predict(X)
