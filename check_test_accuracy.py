import numpy as np
from NeuralNetwork3 import read_label_file

if __name__ == "__main__":
    output_filename = "test_predictions.csv"
    answer_filename = "test_label.csv"

    targets = read_label_file(answer_filename)
    predicts = read_label_file(output_filename)

    num_correct = 0
    for i in range(len(targets)):
        if targets[i] == predicts[i]:
            num_correct += 1
    
    test_accuracy = num_correct/float(len(targets))
    print("Test accuracy is {}".format(test_accuracy))