# Read files 

import pandas as pd 
import os 
os.chdir("C://Users//mandy//Desktop//wine-recommender//")

data = pd.read_csv("winemag-data_first150k.csv", index_col = False)

# Part 1: Using text description to predict points 

# The range of points is within the proper range, but shifted to the right of the spectrum
# The two columns we are interested in (points and description) don't have missing (yay!)

data.describe()

# According to the Wine Spectator's 100-Point Scale, we have the following points segmentation:
# 95-100 = Classic: a great wine
# 90-94 = Outstanding
# 85-89 = Very good
# 80-84 = Good
# 75-79 = Mediocre
# 50-74 = Not recommended 

# Create a new variable to represent the rating class based on the points 

def points_to_rating(points):
    if points in range(80,85):
        return 0
    elif points in range(85,90):
        return 1
    elif points in range(90,95):
        return 2
    elif points in range(95,101):
        return 3

data["rating"] = data["points"].apply(points_to_rating)

# Cleaning the wine descriptions

from nltk.corpus import stopwords
stopwords = set(stopwords.words('english'))

def clean_description(desc):
    desc = desc.lower().split()
    desc = [token for token in desc if token not in stopwords and token.isalpha()]
    return " ".join(desc)

data["cleaned_description"] = data["description"].apply(clean_description)

# One hot encoding the rating labels
# Ratings are not balanced!

import numpy as np 

def onehot(arr, num_class):
    return np.eye(num_class)[np.array(arr.astype(int)).reshape(-1)]

y = onehot(data["rating"],4)

# Split between train and validation (test is in another file)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(data["cleaned_description"], y, test_size = 0.1)

# Vectorize the text 
# I will opt for TF-IDF as there seems to be a common lingo in wine ratings

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features = 500)
X_train = vectorizer.fit_transform(X_train).toarray()
X_val = vectorizer.transform(X_val).toarray()

# Build the neural network
# To think about: there should be a distinction between mistakening two similar grade wines vs an outstanding and mediocre wine

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv1D, GlobalMaxPooling1D
from keras.callbacks import EarlyStopping, ModelCheckpoint, Callback

file_path = "C://Users//mandy//Desktop//wine-recommender/best_model.h5"

model = Sequential()
model.add(Dense(256, input_dim=500, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(200, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(4, activation="softmax")) # number of classes go here
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])

# utilizing early stopping 
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
early = EarlyStopping(monitor='val_loss', mode='min', patience=3)
callback = [checkpoint, early] 

model.fit(X_train, y_train, batch_size = 128, epochs = 20, validation_data=(X_val, y_val), callbacks = callback)

# See validation results

predictions = model.predict(X_val)
predictions = [np.argmax(x) for x in predictions]
actual_labels = [np.argmax(x) for x in y_val]

from sklearn.metrics import confusion_matrix 

matrix = confusion_matrix(actual_labels, predictions)