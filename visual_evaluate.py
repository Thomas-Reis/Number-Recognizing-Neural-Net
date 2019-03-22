import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


def evaluate(images, labels, neural_network, output_layer, X, set_name):
    # Pick 10 random images from the Given Set
    sample_indexes = random.sample(range(len(images)), 10)
    sample_images = [images[i] for i in sample_indexes]
    sample_labels = [labels[i] for i in sample_indexes]
    # Runs the neural net to classify the Given Images
    results = neural_network.run(tf.argmax(output_layer, 1), feed_dict={X: sample_images})

    # Display the prediction and Results Visually.
    fig = plt.figure(figsize=(10, 5))

    num_correct = 0
    # iterates through results from sample set classification
    for i in range(len(sample_images)):
        # sets the actual result as the INDEX of the output label (eg [0,0,1,0...0] its a result of 2
        actual_result = np.where(sample_labels[i] == 1)
        yielded_result = results[i]
        # 3 rows, 4 columns, for 10 images. 1+i for the index of which subplot to write to (min of 1)
        plt.subplot(3, 4, i + 1)
        # disables the axis for the subplot for easier viewing
        plt.axis('off')
        # sets the output color based on whether or not the Neural Net predicted correctly
        if actual_result == yielded_result:
            color = 'green'
            num_correct += 1
        else:
            color = 'red'
        # cleans the prediction for display
        predict_string = str(actual_result[0]).replace('[', '').replace(']', '')
        # Displays text at 5,5 within the subplot (0,0 is the top left of the subplot)
        plt.text(5, 5, "Actual = {}\nPredicted = {}".format(predict_string, yielded_result), color=color)
        # Adds the image to the plot
        plt.imshow(np.reshape(sample_images[i], (-1, 5)), cmap="gray")

    print("Accuracy on " + set_name + " set: " + str(num_correct) + '/' + str(len(sample_images)))
    fig.suptitle(set_name + " Set Results", fontsize=36)
    plt.show()