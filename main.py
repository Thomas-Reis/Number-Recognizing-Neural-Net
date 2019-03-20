import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import data_handler
import random
from random import randint

test_directory = 'Numbers/'
train_directory = 'Numbers/'
validation_directory = 'Numbers/'

i = 0
default_label = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
train_images, train_labels = data_handler.load_data(train_directory, default_label)

print(train_labels.ndim)
print(train_images.shape)
tmp_imgs = train_images.flatten()
print(tmp_imgs.shape)

validation_images, validation_labels = data_handler.load_data(validation_directory, default_label)
test_images, test_labels = data_handler.load_data(validation_directory, default_label)

# Number of training examples we have
n_train = len(train_images)
n_validation = len(validation_images)
n_test = len(test_images)

# set the number of neurons per layer
# 45 inputs for each pixel
n_input = 45
# 5 because the example has 5 on hidden layer 1
n_hidden1 = 5
# 10 for output as classification
n_output = 10

# Neural Network Variables
learning_rate = 0.0001
n_iterations = 1000
batch_size = 10
dropout = 0.5

X = tf.placeholder("float", [None, n_input])
Y = tf.placeholder("float", [None, n_output])
keep_prob = tf.placeholder(tf.float32)

weights = {
    'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
    'out': tf.Variable(tf.truncated_normal([n_hidden1, n_output], stddev=0.1)),
}

biases = {
    'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
}

layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
layer_drop = tf.nn.dropout(layer_1, keep_prob)
output_layer = tf.matmul(layer_1, weights['out']) + biases['out']

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# train on mini batches
for i in range(n_iterations):
    batch_x, batch_y = train_images, train_labels
    sess.run(train_step, feed_dict={X: batch_x, Y: batch_y, keep_prob:dropout})

    # print loss and accuracy (per minibatch)
    if i%100==0:
        minibatch_loss, minibatch_accuracy = sess.run([cross_entropy, accuracy], feed_dict={X: batch_x, Y: batch_y, keep_prob:1.0})
        print("Iteration", str(i), "\t| Loss =", str(minibatch_loss), "\t| Accuracy =", str(minibatch_accuracy))

test_accuracy = sess.run(accuracy, feed_dict={X: test_images, Y: test_labels, keep_prob: 1.0})
print("\nAccuracy on test set:", test_accuracy)

rand_test = randint(0,9)
prediction = sess.run(tf.argmax(output_layer,1), feed_dict={X: [test_images[rand_test]]})
print ("Prediction for test image:", np.squeeze(prediction))
print ("actual value for test image: " + str(rand_test))





# Pick 10 random images
sample_indexes = random.sample(range(len(train_images)), 10)
sample_images = [train_images[i] for i in sample_indexes]
sample_labels = [train_labels[i] for i in sample_indexes]

# Run the "correct_pred" operation
predicted = sess.run(tf.argmax(output_layer,1), feed_dict={X: sample_images})

# Print the real and predicted labels
print(sample_labels)
print(predicted)

# Display the predictions and the ground truth visually.
fig = plt.figure(figsize=(10, 10))
for i in range(len(sample_images)):
    actual = np.where(sample_labels[i] == 1)
    prediction = predicted[i]
    plt.subplot(5, 2, 1 + i)
    plt.axis('off')
    color = 'green' if actual == prediction else 'red'
    predict_string = str(actual[0]).replace('[', '').replace(']', '')
    plt.text(5, 5, "Actual:       {}\nPredicted: {}".format(predict_string, prediction),
             fontsize=12, color=color)
    plt.imshow(np.reshape(sample_images[i], (-1, 5)), cmap="gray")

plt.show()

print("Complete!")