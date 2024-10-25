# Import Library Python
import numpy as np
import pandas as pd
import tensorflow as tf
from flask import Flask, request, render_template
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from PyPDF2 import PdfReader
from nltk.corpus import stopwords
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l2
import re
import os
import nltk

# Initialize Flask app
app = Flask(__name__)

# Download stopwords
nltk.download('stopwords')

clean_df = pd.read_csv('clean_postings.csv')

# Initialize vectorizer and calculate TF-IDF vectors
vectorizer = TfidfVectorizer(min_df=10, max_df=0.8, sublinear_tf=True, use_idf=True)
X = vectorizer.fit_transform(clean_df['combined'])

# Reduce dimensions with Truncated SVD
svd = TruncatedSVD(n_components=500)
X_reduced = svd.fit_transform(X)

# Dummy target variable (replace with actual labels if available)
y = np.random.randint(0, 2, X_reduced.shape[0])

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# TensorFlow-based neural network model
def create_model(input_dim):
    model = Sequential([
        Dense(256, input_dim=input_dim, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(128, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.4),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Create and train the model
model = create_model(X_train.shape[1])
early_stopping = EarlyStopping(monitor='val_loss', patience=5)
model.fit(X_train, y_train, epochs=20, batch_size=64, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)