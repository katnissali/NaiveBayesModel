import math
import os
import sys
from collections import defaultdict, Counter
import time

import pandas as pd
from sklearn.model_selection import train_test_split

class NaiveBayes:
    def __init__(self, data_col, classifier_col):
        self.word_probabilities = {}
        self.class_probabilities = {}
        self.vocabulary = set()
        self.classes = set()
        self.data_col = data_col
        self.classifier_col = classifier_col

    def tokenize(self, text):
        return text.lower().split()

    @staticmethod

    def load_csv(file):
        if not os.path.exists(file):
            sys.exit(file, " does not exist.")

        return pd.read_csv(file)

    # read csv, drop all rows without a classifier, handle NaN messages, tokenize messages
    def prep_df(self, file):
        df = self.load_csv(file)
        df = df.dropna(subset=self.classifier_col)
        df[self.data_col] = df[self.data_col].fillna('').apply(self.tokenize)
        return df

    def train(self, training_file):

        print("Starting training process...")

        df = self.prep_df(training_file)

        messages = df[self.data_col]
        sentiments = df[self.classifier_col]

        self.classes.update(sentiments)
        class_counts = sentiments.value_counts().to_dict()

        word_counts = defaultdict(Counter)
        for sentiment, tokens in zip(sentiments, messages):
            word_counts[sentiment].update(tokens)
            self.vocabulary.update(tokens)

        total_examples = len(df)

        # calculate word probabilities
        for sentiment in self.classes:
            count = class_counts[sentiment]
            self.class_probabilities[sentiment] = math.log(count / total_examples)

            total_words_in_class = sum(word_counts[sentiment].values()) + len(word_counts[sentiment])

            self.word_probabilities[sentiment] = {}
            for word, count in word_counts[sentiment].items():
                self.word_probabilities[sentiment][word] = math.log((count + 1) / total_words_in_class)

        print(f"Training completed. Classes: {self.classes}, Vocabulary size: {len(self.vocabulary)}")

    def predict(self, tokens):
        class_scores = {sentiment: self.class_probabilities[sentiment] for sentiment in self.classes}

        print(f"\nPredicting sentiment for tokens: {tokens}")

        # calculate class scores based on token probabilities
        log_epsilon = math.log(1e-10)  # calculate once before looping
        for sentiment in self.classes:
            for token in tokens:
                # get probability of token: if exists use it, else default to epsilon (smoothing)
                class_scores[sentiment] += self.word_probabilities[sentiment].get(token, log_epsilon)

        print(f"Class scores: {class_scores}")

        predicted_class = max(class_scores, key=class_scores.get)
        print(f"Predicted sentiment: {predicted_class}")
        return predicted_class

    def evaluate(self, validation_file):
        print("\nStarting evaluation process...")

        if not os.path.exists(validation_file):
            print(validation_file, " does not exist.")

        df = self.prep_df(validation_file)
        rows = len(df)

        df['predicted_class'] = df[self.data_col].apply(self.predict)
        df['result'] = (df['predicted_class'] == df[self.classifier_col])
        correct_predictions = df['result'].sum()  # true=1, false=0

        accuracy = correct_predictions / rows
        print(f"\nEvaluation completed. Accuracy: {accuracy:.2%}")
        return accuracy

    @staticmethod
    def split_data(file_path, test_size=0.5, random_state=42, index=False):
        # load data
        data = NaiveBayes.load_csv(file_path)

        # Split the data into training and validation sets (80% train, 20% validation)
        train_data, valid_data = train_test_split(data, test_size=test_size, random_state=random_state)

        # Save the split data (optional)
        train_data.to_csv('training_data.csv', index=index)
        valid_data.to_csv('validation_data.csv', index=index)

        print("Training and validation data split completed.")
        return train_data, valid_data

    @staticmethod
    def run(training_path, data_col, classifier_col, validation_path=None):

        if validation_path is None:
            NaiveBayes.split_data(file_path=training_path, test_size=0.025)
            training_path = "training_data.csv"
            validation_path = "validation_data.csv"

        total_time = time.time()
        analyzer = NaiveBayes(data_col, classifier_col)

        train_time = time.time()
        analyzer.train(f'{training_path}')
        train_time = time.time() - train_time

        evaluate_time = time.time()
        accuracy = analyzer.evaluate(f'{validation_path}')
        evaluate_time = time.time() - evaluate_time

        total_time = time.time() - total_time

        print("Results:")

        print(f"    Validation Accuracy: {accuracy:.2%}")
        print(f"    Training time: {train_time:.2}")
        print(f"    Evaluation time: {evaluate_time:.2}")
        print(f"    Total time: {total_time:.2}")
