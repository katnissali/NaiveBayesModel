import csv
import math
import os
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
        # lowercase, split by " "
        return text.lower().split()

    def train(self, training_file):
        word_counts = defaultdict(Counter)
        class_counts = Counter()

        print("Starting training process...")

        # open training file and process data
        if not os.path.exists(training_file):
            print(training_file, " does not exist.")

        with open(training_file, 'r') as file:
            reader = csv.reader(file)

            headers = next(reader)  # First row is the header

            # check training file empty
            if headers is None:
                raise ValueError("Training file is empty")

            class_idx = headers.index(self.classifier_col)
            data_idx = headers.index(self.data_col)

            for row in reader:
                sentiment = row[class_idx]
                data = row[data_idx]

                self.classes.add(sentiment)
                class_counts[sentiment] += 1

                tokens = self.tokenize(data)
                self.vocabulary.update(tokens)
                word_counts[sentiment].update(tokens)

        vocab_len = len(self.vocabulary)

        total_examples = sum(class_counts.values())
        print(f"Class probabilities before: {self.class_probabilities}")

        self.class_probabilities = {
            sentiment: math.log(count / total_examples)
            for sentiment, count in class_counts.items()
        }

        print(f"Class probabilities: {self.class_probabilities}")


        # calculate word probabilities
        for sentiment in self.classes:
            total_words_in_class = sum(word_counts[sentiment].values()) + vocab_len
            self.word_probabilities[sentiment] = {
                word: math.log((count + 1) / total_words_in_class)
                for word, count in word_counts[sentiment].items()
            }

        print(f"Training completed. Classes: {self.classes}, Vocabulary size: {vocab_len}")

    def predict(self, data):
        tokens = self.tokenize(data)
        class_scores = {sentiment: self.class_probabilities[sentiment] for sentiment in self.classes}

        print(f"\nPredicting sentiment for data: {data}")

        # calculate class scores based on token probabilities
        log_epsilon = math.log(1e-10)
        for sentiment in self.classes:
            for token in tokens:
                # get porbability of token: if exists use it, else default to epsilon
                # smoothing
                class_scores[sentiment] += self.word_probabilities[sentiment].get(token, -log_epsilon)

        print(f"Class scores: {class_scores}")

        # use min instead of max bc log of a positive fraction flips the sign
        predicted_class = min(class_scores, key=class_scores.get)
        print(f"Predicted sentiment: {predicted_class}")
        return predicted_class

    def evaluate(self, validation_file):
        correct_predictions = 0
        total_predictions = 0

        print("\nStarting evaluation process...")

        rows = 0


        if not os.path.exists(validation_file):
            print(validation_file, " does not exist.")

        # open validation file and process data
        with open(validation_file, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                rows += 1
                actual_sentiment = row[self.classifier_col]
                data = row[self.data_col]

                predicted_sentiment = self.predict(data)
                if predicted_sentiment == actual_sentiment:
                    correct_predictions += 1
                total_predictions += 1

                print(f"Actual: {actual_sentiment}, Predicted: {predicted_sentiment}")

        accuracy = correct_predictions / total_predictions
        print(f"\nEvaluation completed. Accuracy: {accuracy:.2%}")
        print("rows: ", rows)
        return accuracy

    @staticmethod
    def split_data(file_path, test_size=0.5, random_state=42, index=False):
        data = None

        # Load your dataset
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        data = pd.read_csv(file_path)

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

        print("start time: ", time.time())
        total_time = time.time()
        analyzer = NaiveBayes(data_col, classifier_col)

        train_time = time.time()
        analyzer.train(f'{training_path}')
        train_time = time.time()-train_time

        evaluate_time = time.time()
        accuracy = analyzer.evaluate(f'{validation_path}')
        evaluate_time = time.time()-evaluate_time

        total_time = time.time()-total_time

        print("Results:")

        print(f"    Validation Accuracy: {accuracy:.2%}")
        print(f"    Training time: {train_time:.2}")
        print(f"    Evaluation time: {evaluate_time:.2}")
        print(f"    Total time: {total_time:.2}")


