import csv
import math
import sys
from collections import defaultdict, Counter

class NaiveBayesSentimentAnalyzer:
    def __init__(self):
        self.word_probabilities = {}
        self.class_probabilities = {}
        self.vocabulary = set()
        self.classes = set()
        self.epsilon = 1e-10

    def tokenize(self, text):
        return text.lower().split()

    def train(self, training_file):
        word_counts = defaultdict(Counter)
        class_counts = Counter()

        print("Starting training process...")

        # Open the training file and process each row
        with open(f'{training_file}', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                print("row: ", row, "\n")
                sentiment = row['sentiment']
                message = row['message']

                print(f"Processing row: Sentiment={sentiment}, Message={message[:30]}...")

                # Update class counts and vocabulary
                self.classes.add(sentiment)
                class_counts[sentiment] += 1

                tokens = self.tokenize(message)

                for token in tokens:
                    word_counts[sentiment][token] += 1
                    self.vocabulary.add(token)

        # Calculate class probabilities
        total_examples = sum(class_counts.values())
        print(f"Class probabilities before: {self.class_probabilities}")

        self.class_probabilities = {
            sentiment: math.log(count / total_examples)
            for sentiment, count in class_counts.items()
        }

        print(f"Class probabilities: {self.class_probabilities}")

        # Calculate word probabilities for each class
        for sentiment in self.classes:
            total_words_in_class = sum(word_counts[sentiment].values()) + len(self.vocabulary)
            self.word_probabilities[sentiment] = {
                word: math.log(self.epsilon + ((count + 1) / total_words_in_class))
                for word, count in word_counts[sentiment].items()
            }

        print(f"Training completed. Classes: {self.classes}, Vocabulary size: {len(self.vocabulary)}")

    def predict(self, message):
        tokens = self.tokenize(message)
        class_scores = {sentiment: self.class_probabilities[sentiment] for sentiment in self.classes}

        print(f"\nPredicting sentiment for message: {message}")

        # Calculate the score for each class based on token probabilities
        for sentiment in self.classes:
            for token in tokens:

                if token in self.word_probabilities[sentiment]:
                    class_scores[sentiment] += self.word_probabilities[sentiment][token]
                else:
                    # Handle unseen words with smoothing
                    score = 1 / (sum(self.word_probabilities[sentiment].values()) + len(self.vocabulary))
                    score = max(score, self.epsilon)
                    try:
                        class_scores[sentiment] += math.log(1 / score)
                    except:
                        print(f"Error: class score for message: {message} is {score}, skipping.")

        print(f"Class scores: {class_scores}")
        # Return the class with the highest score
        predicted_class = min(class_scores, key=class_scores.get)
        print(f"Predicted sentiment: {predicted_class}")
        return predicted_class

    def evaluate(self, validation_file):
        correct_predictions = 0
        total_predictions = 0

        print("\nStarting evaluation process...")

        # Open the validation file and process each row
        with open(f'{validation_file}', 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                actual_sentiment = row['sentiment']
                message = row['message']

                predicted_sentiment = self.predict(message)
                if predicted_sentiment == actual_sentiment:
                    correct_predictions += 1
                total_predictions += 1

                print(f"Actual: {actual_sentiment}, Predicted: {predicted_sentiment}")

        accuracy = correct_predictions / total_predictions
        print(f"\nEvaluation completed. Accuracy: {accuracy:.2%}")
        return accuracy

# Example usage
if len(sys.argv) < 2:
    print("Run with training set filename and validation set filename.")
else:
    training_path = sys.argv[1]
    validation_path = sys.argv[2]

    analyzer = NaiveBayesSentimentAnalyzer()
    analyzer.train(f'{training_path}')
    accuracy = analyzer.evaluate(f'{validation_path}')
    print(f"Validation Accuracy: {accuracy:.2%}")