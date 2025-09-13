Sentiment Analysis on Amazon Reviews using RNN, LSTM, and GRU
This project performs sentiment analysis on the Amazon Reviews dataset using three different types of Recurrent Neural Networks (RNNs): a simple RNN, Long Short-Term Memory (LSTM), and Gated Recurrent Unit (GRU). The goal is to classify reviews as either positive or negative and compare the performance of these deep learning models.

Table of Contents

Project Description
Dataset
Methodology
Model Architectures
Results
How to Use
Dependencies

Project Description
The notebook walks through the complete process of building, training, and evaluating sentiment analysis models. It starts with data loading and preprocessing, followed by the implementation of three distinct neural network architectures. Each model's performance is evaluated using standard classification metrics, and a prediction system is built to test the best-performing model on new, unseen text.

Dataset
The project utilizes the Amazon Reviews Dataset. Due to the large size of the original dataset, a subset of the data is used for training and testing to facilitate faster model training:

Training Data: 3,000 samples
Test Data: 1,000 samples
The labels are binary:
0: Negative sentiment
1: Positive sentiment

Methodology
The workflow is divided into the following key steps:
Data Loading: Subsets of the training and test data are loaded from .bz2 compressed files.

Data Preparation:
Labels (__label__1 or __label__2) and review texts are extracted from each line.
Labels are converted to a binary format (0 for negative, 1 for positive).

Text Cleaning:
Text is converted to lowercase.
Non-alphabetic characters are removed.
Common English stopwords are removed using nltk.
Words are stemmed to their root form using the PorterStemmer.

Tokenization & Padding:
The cleaned text is converted into sequences of integers using Tokenizer. The vocabulary size is limited to the top 1000 words.
All sequences are padded to a uniform length of 100 (max_sequence_length).
Model Training: Three different models (SimpleRNN, Bidirectional LSTM, and GRU) are built, compiled, and trained. Early stopping is used to prevent overfitting.
Model Evaluation: Each model's performance on the test set is evaluated using a confusion matrix and a classification report (precision, recall, F1-score).
Prediction System: A function predict_sentiment is created to perform sentiment analysis on new text inputs using the trained model and tokenizer.

Model Architectures
All models use an Embedding layer with an output dimension of 300. The optimizer is Adam with a learning rate of 5×10 
−5
 , and the loss function is binary_crossentropy.

1. Simple RNN Model
Python

model = Sequential([
    Embedding(input_dim=1000, output_dim=300, input_length=100),
    SimpleRNN(128, return_sequences=True, dropout=0.3),
    LayerNormalization(),
    SimpleRNN(128, dropout=0.3),
    LayerNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
2. Bidirectional LSTM Model
Python

model = Sequential([
    Embedding(input_dim=1000, output_dim=300, input_length=100),
    Bidirectional(LSTM(128, return_sequences=True, recurrent_dropout=0.2)),
    LayerNormalization(),
    Dropout(0.3),
    Bidirectional(LSTM(128, recurrent_dropout=0.2)),  
    LayerNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
3. GRU Model
Python

model = Sequential([
    Embedding(input_dim=1000, output_dim=300, input_length=100),
    GRU(128, return_sequences=True, dropout=0.3, recurrent_dropout=0.2),
    LayerNormalization(),
    GRU(128, dropout=0.3, recurrent_dropout=0.2),
    LayerNormalization(),
    Dropout(0.3),
    Dense(64, activation="relu"),
    Dropout(0.2),
    Dense(1, activation='sigmoid')
])
Results
The performance of the three models on the test set is summarized below. The Bidirectional LSTM model achieved the highest accuracy.

Model	Accuracy	F1-Score (Class 0)	F1-Score (Class 1)
Simple RNN	74%	0.77	0.71
Bidirectional LSTM	75%	0.75	0.76
GRU	67%	0.71	0.62

Export to Sheets
The LSTM model was saved as lstm_model.h5 for future use.

How to Use
To run this project, follow these steps:

Clone the repository:

Bash

git clone <repository-url>
cd <repository-directory>
Install dependencies:
Make sure you have Python installed, then install the required libraries.

Bash

pip install -r requirements.txt
Download NLTK Stopwords:
Run the following in a Python interpreter:

Python

import nltk
nltk.download('stopwords')
Run the Jupyter Notebook:
Launch Jupyter Notebook and open sentiment-analysis-on-social-media-posts-with-lstm-rnn-and-gru.ipynb.

Bash

jupyter notebook
Use the Prediction System:
The notebook includes a section to make predictions on new sentences. The trained LSTM model (lstm_model.h5) and tokenizer (tokenizer.pkl) can be loaded to predict the sentiment of any text input.

Dependencies
The project requires the following Python libraries:

numpy
pandas
matplotlib
scikit-learn
tensorflow==2.12.0
nltk
bz2file

You can install them using pip. A requirements.txt file should be included in the repository.
