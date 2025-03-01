import sys
from naive_bayes import NaiveBayes

if len(sys.argv) == 5:
    data_col = sys.argv[3]
    classifier_col = sys.argv[4]
    NaiveBayes.run(sys.argv[1], data_col, classifier_col, validation_path=sys.argv[2])
else:
    print("Run with arguments: <training csv path> <validation csv path> <data column header> <classifier column header>")