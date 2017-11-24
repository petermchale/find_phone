import numpy as np
import cPickle
from utility import create_window, design_matrix
from PIL import Image, ImageDraw
import sys
import matplotlib.pyplot as plt


def window_coordinates(half_window_side_normalized, number_windows):
    """ Create an equally-spaced array of coordinate values of sliding-window positions """

    return np.linspace(half_window_side_normalized,
                       1.0 - half_window_side_normalized,
                       number_windows)


def window_generator(image, number_windows_x=50, number_windows_y=50):
    """
    Generate a sequence of sliding windows
    of the same dimensions and resolution used to train the classifier
    """

    with open('half_window_side_normalized.cPickle', 'rb') as file_in:
        half_window_side_normalized = cPickle.load(file_in)

    for x_normalized in window_coordinates(half_window_side_normalized, number_windows_x):
        for y_normalized in window_coordinates(half_window_side_normalized, number_windows_y):
            box, window = create_window(image, x_normalized, y_normalized, half_window_side_normalized)
            yield x_normalized, y_normalized, box, window


def scan(image_path, show_image=True):
    """
    Slide window across given image.
    Find window most likely to contain the phone.
    """

    # open image and create a sequence of sliding windows
    image = Image.open(image_path).convert('L')
    xs_normalized, ys_normalized, boxes, windows = zip(*window_generator(image))

    # fetch classifier
    with open('model.cPickle', 'rb') as file_in:
        classifier = cPickle.load(file_in)

    # find which window most likely contains phone
    probabilities_positive = classifier.predict_proba(design_matrix(windows))[:, 1]
    argmax = np.argmax(probabilities_positive)

    # draw most-likely window on original image for visual confirmation
    if show_image:
        image_draw = ImageDraw.Draw(image)
        box_max = boxes[argmax]
        image_draw.rectangle(box_max)
        image_draw.text((box_max[0], box_max[1]), 'prob. = %.4f' % probabilities_positive[argmax])
        plt.imshow(np.array(image))
        plt.show()

    # fetch coordinates of center of maximum-likelihood window
    return xs_normalized[argmax], ys_normalized[argmax]


if __name__ == '__main__':

    x_predicted, y_predicted = scan(image_path=sys.argv[1], show_image=True)
    print x_predicted, y_predicted
