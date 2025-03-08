import sys
from naive_bayes import NaiveBayes


if len(sys.argv) == 5:
    # for debugging, todo remove in final product
    training_path = sys.argv[1]
    validation_path = sys.argv[2]
    data_col = sys.argv[3]
    classifier_col = sys.argv[4]

    NaiveBayes.run(training_path, data_col, classifier_col, validation_path)

else:
    print("Run with arguments: <training csv path> <validation csv path> <data column header> <classifier column header>")