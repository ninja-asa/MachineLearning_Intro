#!/usr/bin/env python
"""This script prompts a user to enter a message to encode or decode
using a classic Caeser shift substitution (3 letter shift)"""

from sklearn.datasets import load_digits
import matplotlib.pyplot as plt 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn import metrics
if __name__ == "__main__":
    DIGITS_DATASET = load_digits()

    # Print to show there are 1797 images (8 by 8 images for a dimensionality of 64)
    print("Image Data Shape", DIGITS_DATASET.data.shape)

    # Print to show there are 1797 labels (integers from 0-9)
    print("Label Data Shape", DIGITS_DATASET.data.shape)

    plt.figure(figsize=(20,4))
    for index, (image, label) in enumerate(zip(DIGITS_DATASET.data[0:5], DIGITS_DATASET.target[0:5])):
        plt.subplot(1, 5, index + 1)
        plt.imshow(np.reshape(image, (8,8)), cmap=plt.cm.gray)
        plt.title('Training: %i\n' % label, fontsize = 20)
    plt.show()

    x_train, x_test, y_train, y_test = train_test_split(DIGITS_DATASET.data, 
        DIGITS_DATASET.target, test_size=0.25, random_state=0)

    # all parameters not specified are set to their defaults
    logisticRegr = LogisticRegression() 

    logisticRegr.fit(x_train, y_train)
    # Returns a NumPy Array
    # Predict for One Observation (image)
    logisticRegr.predict(x_test[0].reshape(1,-1))
    logisticRegr.predict(x_test[0:10])
    predictions = logisticRegr.predict(x_test)
    # Use score method to get accuracy of model
    score = logisticRegr.score(x_test, y_test)
    print(score)
    cm = metrics.confusion_matrix(y_test, predictions)
    print(cm)

    plt.figure(figsize=(9,9))
    sns.heatmap(cm, annot=True, fmt=".3f", linewidths=.5, square = True, cmap = 'Blues_r');
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    all_sample_title = 'Accuracy Score: {0}'.format(score)
    plt.title(all_sample_title, size = 15)
    plt.show()
