# MNIST-fullyconnected
Fully connected neural network with 3 hidden layers, relu activation and batch gradient descent. Able to achieve 96% accuracy with 10000 training data.

To run the code. You need to store training data and label in file called train_label.csv and train_image.csv in the same format as test_label.csv and test_image.csv The size of the train data can be up to the user. 

Run command:

  python3 Three_layer.py
  
The program will take in train_image.csv, train_label.csv, and test_image.csv and output a file test_predictions.csv.
To check on test accuracy, run:
  python3 check_test_accuracy.py
An accuracy will be outputed according to test_predictions.csv and test_label.csv
