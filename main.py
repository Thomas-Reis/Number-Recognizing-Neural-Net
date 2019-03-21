import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random

import data_handler
import visual_evaluate

# set the number of neurons per layer
# 45 inputs for each pixel (do not touch unless image sizes change)
n_input = 45
# 5 because the example has 5 on hidden layer 1
n_hidden1 = 5
# 10 for output as classifications
n_output = 10

# Neural Network Variables
learning_rate = 0.001
n_iterations = 100
n_epochs = 100
dropout = 0.5

# Setting the initial values for the connections/inputs for the neural network to each layer
initial_w1 = 0.01
initial_w2 = 0.01
bias1 = 0.001
bias2 = 0.001

# Directories for sample
test_directory = 'Test_Numbers/'
train_directory = 'Train_Numbers/'
# Whether or not you want the visual evaluation afterwards
display_training_complete = True
display_test_complete = True

# the default label for classification (no classification), 10 values for each digit
default_label = [0] * 10

# Loads the training, testing, and validation images, as well as their labels
train_images, train_labels = data_handler.load_data(train_directory, default_label)
test_images, test_labels = data_handler.load_data(test_directory, default_label)

# Number of training examples we have
n_train = len(train_images)
n_test = len(test_images)

# Initializing the tensorflow variables for the inputs and outputs as placeholders
X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
# Setting a variable placeholder for the dropout rate
keep_prob = tf.placeholder(tf.float32)

# sets the initial weights, w1 = input to hidden layer 1, out = hidden layer 1 to output
weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=initial_w1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden1, n_output], stddev=initial_w2)),
}

# setting low biases as the inputs should not change TOO much
biases = {
    'b1': tf.Variable(tf.constant(bias1, shape=[n_hidden1])),
    'out': tf.Variable(tf.constant(bias2, shape=[n_output]))
}

# initializes the layers with the given weights, inputs, and biases
# tf.add sums up the contents, as a layer should do (sum(weight * input))
# tf.matmul = matrix multiplication (input vector x Weights vector)
layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
output_layer = tf.add(tf.matmul(layer_1, weights['out']), biases['out'])

# Attempts to reduce the mean between the actual labels and output layer labels,
# Y is like a variable name, the actual labels are assigned later when we call run
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
# Optimizes the training process using the Adam optimization algorithm
# attempts to minimize the cross entropy (difference between predicted label and actual label)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Determines if the output layer's value is the same as the label value assigned to Y (the actual label during the run)
# During the run it'll be true if correct, false if incorrect
correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
# Casts the predictions to a float and calculates the average accuracy overall during the run
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init_variables = tf.global_variables_initializer()
# Initializes the neural network as an object
neural_network = tf.Session()
# Runs the neural network with the global variables required from tensorflow
neural_network.run(tf.global_variables_initializer())


loss_plot = []
accuracy_plot = []
for j in range(n_epochs):
    # train on the entire set of training images and labels, with a dropout rate to avoid over fitting
    neural_network.run(train_step, feed_dict={X: train_images, Y: train_labels, keep_prob: dropout})
    # Gets the Loss and Accuracy on the set as of the current epoch (this does not train it, just evaluates)
    # Runs using all neurons and uses ALL results from the layers for evaluation
    epoch_loss, epoch_accuracy = \
        neural_network.run([cross_entropy, accuracy], feed_dict={X: train_images, Y: train_labels, keep_prob: 1.0})
    loss_plot.append(epoch_loss)
    accuracy_plot.append(epoch_accuracy)
    print("Epoch ", str(j), "    | Loss =", str(epoch_loss), "    | Accuracy =", str(epoch_accuracy))

# Prints a newline to separate output from runs with resulting test outputs
print()
# Displays the Mean Squared Errors per Epoch
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel("Mean Squared Error (Loss)")
plt.title("Epochs vs Mean Squared Error")
plt.show()

# Displays the Accuracy per Epoch
plt.plot(accuracy_plot)
plt.xlabel('Epochs')
plt.ylabel("Accuracy")
plt.title("Epochs vs Accuracy")
plt.show()

if display_training_complete:
    visual_evaluate.evaluate(train_images, train_labels, neural_network, output_layer, X, "Training")

if display_test_complete:
    visual_evaluate.evaluate(test_images, test_labels, neural_network, output_layer, X, "Test")

print("Complete!")