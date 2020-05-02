import math
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt


class OneHotEncoder:
    """
    One Hot Encoder
    One hot encoding is a process by which categorical variables are converted into a form that
    could be provided to ML algorithms to do a better job in prediction.

    0 indicates non existent while 1 indicates existent.
    """

    #
    @staticmethod
    def encode(values: np.array) -> np.array:
        # calling the functions recursively if the array has more than 1 dimension
        if len(values.shape) is not 1:
            return [OneHotEncoder.encode(vals) for vals in values]

        # creates a new array containing zeros and setting (1) to the locations the highest value i represented
        encoded = np.zeros(len(values))
        hot_index = np.argmax(values)
        encoded[hot_index] = 1

        return encoded

    #
    @staticmethod
    def hot_indexes(encodings: np.array) -> np.array:
        # calling the functions recursively if the array has more than 1 dimension
        if len(encodings.shape) is not 1:
            return [OneHotEncoder.hot_indexes(vals) for vals in encodings]

        # finds the highest value (1) and returns the index
        hot_encoded_indexes = np.argmax(encodings)
        return hot_encoded_indexes

    #
    @staticmethod
    def encoding(values: np.array, zero_entry: bool = True) -> np.array:
        # calling the functions recursively if the array has more than 1 dimension
        if len(values.shape) is not 1:
            return [OneHotEncoder.encoding(vals) for vals in values]

        # creates a new array containing zeros with dimensions based on length and max value from the argument
        # the (+ 1) allows zero values to be used as well
        encoded = np.zeros((len(values), max(values) + 1))

        # setting (1) to the locations the highest value i represented
        values.sort()
        for index, value in enumerate(values):
            encoded[index, value] = 1

        return encoded


class Visualizer:

    #
    @staticmethod
    def image(index: int, images: np.array, labels: np.array, prediction: int = None, save: bool = False):
        """
        Displays image corresponding with the given index
        Example - display_image(109, x_test, y_test) 
        """
        label = labels[index]
        image = images[index]
        size = int(math.sqrt(image.size))
        if prediction:
            plt.title(
                f'index: {index},  label: {label}, predicted: {prediction}')
        else:
            plt.title(f'index: {index},  label: {label}')
        plt.imshow(image.reshape([size, size]), cmap='gray_r')

        if save:
            plt.savefig(f'digit_{index}.png', dpi=400, transparent=True)

    #
    @staticmethod
    def heatmap(conf_matrix, save: bool = False):
        conf_matrix = np.around([row/sum(row) for row in conf_matrix], decimals=2)
        plot = sns.heatmap(conf_matrix, cmap=sns.color_palette("Blues"), annot=True)
        figure = plot.get_figure()

        if save:
            figure.savefig('normalization.png', dpi=400, transparent=True)
