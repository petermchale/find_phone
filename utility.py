import numpy as np
import pandas as pd
import os


def create_window(image, x_normalized, y_normalized, half_window_side_normalized,
                  down_sample_width=30, down_sample_height=30):
    """
    Create a window (sub-image) into the given image centered on the given coordinates and with the given dimensions.
    Down-sample the resulting sub-image to reduce the number of features,
    which in turn reduces model complexity and avoids over-fitting.
    """

    x_top_left = (x_normalized - half_window_side_normalized) * image.size[0]
    y_top_left = (y_normalized - half_window_side_normalized) * image.size[1]
    x_bottom_right = (x_normalized + half_window_side_normalized) * image.size[0]
    y_bottom_right = (y_normalized + half_window_side_normalized) * image.size[1]
    box = map(int, (x_top_left, y_top_left, x_bottom_right, y_bottom_right))
    return box, image.crop(box).resize((down_sample_width, down_sample_height))


def design_matrix(images):
    """
    Flatten the images, and combine them into a design matrix:
    rows = examples; cols = features
    """

    return np.array([np.array(image).flatten() for image in images])


def normalized_phone_coordinates(df, image_file_name):
    """ Locate the phone coordinates of an image in a data frame """

    image_df = df[df['image'] == image_file_name]
    x_normalized = float(image_df['x'])
    y_normalized = float(image_df['y'])
    return x_normalized, y_normalized


def open_table(folder):
    """ Open the given text file and insert its table into a data frame """

    return pd.read_table(folder + '/labels.txt',
                         sep=' ',
                         names=['image', 'x', 'y'])


def basename(path_string):
    """ Fetch only the last part of a path """

    return os.path.basename(os.path.normpath(path_string))
