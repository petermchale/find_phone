from PIL import Image
import random
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import cPickle
from sklearn.linear_model import LogisticRegression
from utility import create_window, design_matrix, open_table, normalized_phone_coordinates, basename
import sys


def split_files(folder, training_fraction=0.8):
    """ Split files into a training set and a test set """

    image_paths = glob(folder + '/*.jpg')
    training_test_split = int(training_fraction * len(image_paths))
    image_paths_training = image_paths[:training_test_split]
    with open('image_paths_training.cPickle', 'wb') as file_out:
        cPickle.dump(image_paths_training, file_out)
    image_paths_testing = image_paths[training_test_split:]
    with open('image_paths_testing.cPickle', 'wb') as file_out:
        cPickle.dump(image_paths_testing, file_out)

    return image_paths_training, image_paths_testing


def create_labeled_examples(folder, image_paths, half_window_side_normalized):
    """
    For each image, extract a window containing the phone and a window not containing the phone.
    """

    df = open_table(folder)
    labeled_examples = []
    for image_path in image_paths:
        image = Image.open(image_path).convert('L')
        image_file_name = basename(image_path)
        x_normalized, y_normalized = normalized_phone_coordinates(df, image_file_name)
        box, window = create_window(image, x_normalized, y_normalized, half_window_side_normalized)
        labeled_examples += [(window, 'positive')]
        box, window = create_window(image, random.random(), random.random(), half_window_side_normalized)
        labeled_examples += [(window, 'negative')]
    return labeled_examples


def create_labeled_examples_training_testing(folder, seed=0, half_window_side_normalized=0.1):
    """
    Create training and testing examples.
    """

    random.seed(seed)

    with open('half_window_side_normalized.cPickle', 'wb') as file_out:
        cPickle.dump(half_window_side_normalized, file_out)

    image_paths_training, image_paths_testing = split_files(folder, training_fraction=0.8)
    labeled_examples_training = create_labeled_examples(folder,
                                                        image_paths_training,
                                                        half_window_side_normalized)
    labeled_examples_testing = create_labeled_examples(folder,
                                                       image_paths_testing,
                                                       half_window_side_normalized)

    return labeled_examples_training, labeled_examples_testing


def train_model(X_training, y_training,
                classifier=LogisticRegression(),
                model_filename='model.cPickle'):
    """ Train a machine-learning model to recognize a phone """

    classifier.fit(X_training, y_training)
    with open(model_filename, 'wb') as file_out:
        cPickle.dump(classifier, file_out)


def show_examples(labeled_examples, message, number_examples_to_show=10):
    """ Create a figure containing a sample of images and corresponding labels (predicted or true) """

    for index, (image, label) in enumerate(labeled_examples[:number_examples_to_show]):
        plt.subplot(1, number_examples_to_show, index + 1)
        plt.axis('off')
        plt.imshow(np.array(image), cmap=plt.cm.gray_r)
        plt.title('1' if label == 'positive' else '0')
    plt.suptitle(message)
    plt.show()


def test_model(X_testing, images_testing,
               model_filename='model.cPickle'):
    """ Predict labels on test images """

    with open(model_filename, 'rb') as file_in:
        classifier = cPickle.load(file_in)
    yhat_testing = classifier.predict(X_testing)
    show_examples(zip(images_testing, yhat_testing), 'predicted labels:')


def train_and_test_model(folder):
    """
    Create design matrices for training and testing sets.
    Use design matrices to train and test a machine learning model.
    """

    labeled_examples_training, labeled_examples_testing = \
        create_labeled_examples_training_testing(folder)

    # training and test data
    images_training, labels_training = zip(*labeled_examples_training)
    images_testing, labels_testing = zip(*labeled_examples_testing)

    # design matrices
    X_training = design_matrix(images_training)
    X_testing = design_matrix(images_testing)

    # train and test
    train_model(X_training, labels_training)
    test_model(X_testing, images_testing)


if __name__ == '__main__':
    train_and_test_model(sys.argv[1])
