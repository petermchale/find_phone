from utility import open_table, normalized_phone_coordinates, basename
import numpy as np
from find_phone import scan
import cPickle


def accuracy(folder='data'):

    df = open_table(folder)

    # fetch image paths of testing data
    with open('image_paths_testing.cPickle', 'rb') as file_in:
        image_paths_testing = cPickle.load(file_in)

    number_correct_predictions = 0
    number_false_predictions = 0
    for image_path in image_paths_testing:

        image_file_name = basename(image_path)
        x_exact, y_exact = normalized_phone_coordinates(df, image_file_name)

        x_predicted, y_predicted = scan(image_path, show_image=False)

        error_x = x_predicted - x_exact
        error_y = y_predicted - y_exact
        if np.sqrt(error_x**2 + error_y**2) < 0.05:
            number_correct_predictions += 1
        else:
            number_false_predictions += 1

    print 'accuracy of object detection algorithm = %.1f' % \
          (number_correct_predictions * 100 / float(number_correct_predictions + number_false_predictions)), '%'
    print 'number of testing examples = ', len(image_paths_testing)


if __name__ == '__main__':
    accuracy()