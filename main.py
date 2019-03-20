import tensorflow as tf
from tensorflow import keras
import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import data_handler

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

# Initialize placeholders
x = tf.placeholder(dtype = tf.float32, shape = [None, train_images.shape[1], train_images.shape[2]])
y = tf.placeholder(dtype = tf.int32, shape = [None])

images_flat = tf.contrib.layers.flatten(x)
logits = tf.contrib.layers.fully_connected(images_flat, 10, tf.nn.relu)
loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels = y, logits = logits))
train_op = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss)
correct_pred = tf.argmax(logits, 1)
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

tf.set_random_seed(1234)
sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(0, 50):
    for i in range(500):
            print('EPOCH', i)
            _, accuracy_val = sess.run([train_op, accuracy], feed_dict={x: train_images, y: train_labels})
            if i % 10 == 0:
                print("Loss: ", loss)
            print('DONE WITH EPOCH')

    import matplotlib.pyplot as plt
    import random

    # Pick 10 random images
    sample_indexes = random.sample(range(len(train_images)), 10)
    sample_images = [train_images[i] for i in sample_indexes]
    sample_labels = [train_labels[i] for i in sample_indexes]

    # Run the "correct_pred" operation
    predicted = sess.run([correct_pred], feed_dict={x: sample_images})[0]

    # Print the real and predicted labels
    print(sample_labels)
    print(predicted)

    # Display the predictions and the ground truth visually.
    fig = plt.figure(figsize=(10, 10))
    for i in range(len(sample_images)):
        truth = sample_labels[i]
        prediction = predicted[i]
        plt.subplot(5, 2, 1 + i)
        plt.axis('off')
        color = 'green' if truth == prediction else 'red'
        plt.text(5, 5, "Truth:        {0}\nPrediction: {1}".format(truth, prediction),
                 fontsize=12, color=color)
        plt.imshow(sample_images[i], cmap="gray")

    plt.show()

print("break here")


# validation_images, validation_labels = data_handler.load_data(validation_directory, default_label)
# test_images, test_labels = data_handler.load_data(validation_directory, default_label)
#
# # Number of training examples we have
# n_train = len(train_images)
# # Number of validation examples we have
# n_validation = len(validation_images)
# # Number of testing examples we have
# n_test = len(test_images)
#
# # set the number of neurons per layer
# # 45 input for each pixel
# n_input = 45
# # 5 because the example has 5 on hidden layer 1
# n_hidden1 = 5
# # 10 for output as classification
# n_output = 10
#
# # Neural Network Variables
# learning_rate = 1e-4
# n_iterations = 1000
# batch_size = 10
# dropout = 0.5





# train_data = keras.preprocessing.sequence.pad_sequences(train_images,
#                                                         value=0,
#                                                         padding='post',
#                                                         maxlen=45)
#
# test_data = keras.preprocessing.sequence.pad_sequences(test_images,
#                                                        value=0,
#                                                        padding='post',
#                                                        maxlen=10)
#
# # input shape is the vocabulary count used for the movie reviews (10,000 words)
# vocab_size = 10
#
# model = keras.Sequential()
# model.add(keras.layers.Embedding(vocab_size, 45))
# model.add(keras.layers.GlobalAveragePooling1D())
# model.add(keras.layers.Dense(5, activation=tf.nn.sigmoid))
# model.add(keras.layers.Dense(10, activation=tf.nn.sigmoid))
#
# model.summary()
#
# model.compile(optimizer='adam',
#               loss='binary_crossentropy',
#               metrics=['acc'])
#
#
# x_val = train_data[:10]
# partial_x_train = train_data[:10]
#
# y_val = train_labels[:10]
# partial_y_train = train_labels[:10]
#
#
# history = model.fit(partial_x_train,
#                     partial_y_train,
#                     epochs=40,
#                     batch_size=10,
#                     validation_data=(x_val, y_val),
#                     verbose=1)
#
# results = model.evaluate(test_data, test_labels)
#
# print(results)
#
# history_dict = history.history
# history_dict.keys()
#
#
# acc = history_dict['acc']
# val_acc = history_dict['val_acc']
# loss = history_dict['loss']
# val_loss = history_dict['val_loss']
#
# epochs = range(1, len(acc) + 1)
#
# # "bo" is for "blue dot"
# plt.plot(epochs, loss, 'bo', label='Training loss')
# # b is for "solid blue line"
# plt.plot(epochs, val_loss, 'b', label='Validation loss')
# plt.title('Training and validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
#
# plt.show()
#
# plt.clf()   # clear figure
#
# plt.plot(epochs, acc, 'bo', label='Training acc')
# plt.plot(epochs, val_acc, 'b', label='Validation acc')
# plt.title('Training and validation accuracy')
# plt.xlabel('Epochs')
# plt.ylabel('Accuracy')
# plt.legend()
#
# plt.show()
#
# print('wait here')







# X = tf.placeholder("float", [None, n_input])
# Y = tf.placeholder("float", [None, n_output])
# keep_prob = tf.placeholder(tf.float32)
#
#
# weights = {
#     'w1': tf.Variable(tf.truncated_normal([n_input, n_hidden1], stddev=0.1)),
#     'out': tf.Variable(tf.truncated_normal([n_hidden1, n_output], stddev=0.1)),
# }
#
# biases = {
#     'b1': tf.Variable(tf.constant(0.1, shape=[n_hidden1])),
#     'out': tf.Variable(tf.constant(0.1, shape=[n_output]))
# }
#
# layer_1 = tf.add(tf.matmul(X, weights['w1']), biases['b1'])
# layer_drop = tf.nn.dropout(layer_1, keep_prob)
# output_layer = tf.matmul(layer_1, weights['out']) + biases['out']
#
# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=output_layer))
# train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
#
# correct_pred = tf.equal(tf.argmax(output_layer, 1), tf.argmax(Y, 1))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
#
# init = tf.global_variables_initializer()
# sess = tf.Session()
# sess.run(init)
