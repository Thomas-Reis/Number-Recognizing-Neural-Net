import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import data_handler
import random
from random import randint

test_directory = 'Hard_Numbers/'
train_directory = 'Numbers/'
validation_directory = 'Numbers/'

# the default label for classification (no classification), 10 values for each digit
default_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

# Loads the training, testing, and validation images, as well as their labels
train_images, train_labels = data_handler.load_data(train_directory, default_label)
test_images, test_labels = data_handler.load_data(test_directory, default_label)
validation_images, validation_labels = data_handler.load_data(validation_directory, default_label)

# Number of training examples we have
n_train = len(train_images)
n_validation = len(validation_images)
n_test = len(test_images)

# set the number of neurons per layer
# 45 inputs for each pixel (do not touch unless image sizes change)
n_input = 45
# 5 because the example has 5 on hidden layer 1
n_hidden1 = 5
# 10 for output as classification
n_output = 10

# Neural Network Variables
learning_rate = 0.001
n_iterations = 100
n_epochs = 200
dropout = 0.5

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.01)),
    'out': tf.Variable(tf.truncated_normal([n_hidden1, n_output], stddev=0.01)),
}

# setting low biases as the inputs should not change TOO much
biases = {
    'b1': tf.Variable(tf.constant(0.001, shape=[n_hidden1])),
    'out': tf.Variable(tf.constant(0.001, shape=[n_output]))
}

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_drop = tf.nn.dropout(layer_1, keep_prob)
output_layer = tf.matmul(layer_1, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)


loss_plot = []
accuracy_plot = []
for j in range(n_epochs):
    # train on batch (all images)
    sess.run(train_step, feed_dict={X: train_images, Y: train_labels, keep_prob:dropout})
    # Gets the Loss and Accuracy on the set as of the current epoch
    epoch_loss, epoch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: train_images, Y: train_labels, keep_prob:1.0})
    loss_plot.append(epoch_loss)
    accuracy_plot.append(epoch_accuracy)
    print("Epoch ", str(j), "\t| Loss =", str(epoch_loss), "\t| Accuracy =", str(epoch_accuracy))

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

# Pick 10 random images from the Test Set
sample_indexes = random.sample(range(len(test_images)), 10)
sample_images = [test_images[i] for i in sample_indexes]
sample_labels = [test_labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run(tf.argmax(output_layer,1), feed_dict={X: sample_images})



# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 5))

for i in range(len(sample_images)):
    actual = np.where(sample_labels[i] == 1)
    prediction = predicted[i]
    # 5 rows, 2 columns, for 10 images. 1+i for the index of which subplot to write to (min of 1)
    plt.subplot(3, 4, i + 1)
    plt.axis('off')
    # sets the output color based on whether or not the Neural Net predicted correctly
    color = 'green' if actual == prediction else 'red'
    # cleans the prediction for display
    predict_string = str(actual[0]).replace('[', '').replace(']', '')
    plt.text(5, 5, "Actual = {}\nPredicted = {}".format(predict_string, prediction), color=color)
    plt.imshow(np.reshape(sample_images[i], (-1, 5)), cmap="gray")

test_accuracy = sess.run(accuracy, feed_dict={X: sample_images, Y: sample_labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)

fig.suptitle("Test Set Results", fontsize=36)
plt.show()

print("Complete!")