"""
Label encoding utilities.
Converts integer labels to one-hot vectors and vice versa.
"""

import numpy as np


class LabelEncoder:
    """
    One-hot encodes integer labels for multi-class classification.

    Attributes
    ----------
    num_classes : int
        Total number of classes in the dataset.
    """

    def __init__(self, num_classes):
        """
        Initialize label encoder.

        Parameters
        ----------
        num_classes : int
            The number of distinct class labels.
        """
        self.num_classes = num_classes

    def encode_single(self, label):
        """
        Convert a single integer label to a one-hot vector.

        Parameters
        ----------
        label : int
            Integer class label (0 to num_classes-1).

        Returns
        -------
        np.ndarray
            1D array of shape (num_classes,) with 1.0 at the label index.
        """
        if not 0 <= label < self.num_classes:
            raise ValueError(
                f"Label {label} out of range [0, {self.num_classes})"
            )
        
        vector = np.zeros(self.num_classes, dtype=np.float32)
        vector[label] = 1.0
        return vector

    def encode(self, labels):
        """
        Convert multiple integer labels to one-hot vectors.

        Parameters
        ----------
        labels : list of int or np.ndarray
            Integer class labels.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, num_classes).
        """
        encoded = np.zeros((len(labels), self.num_classes), dtype=np.float32)
        
        for i, label in enumerate(labels):
            encoded[i] = self.encode_single(label)
        
        return encoded

    def encode_dataset(self, data):
        """
        Convenience method to encode labels from a dataset.

        Parameters
        ----------
        data : list of dict
            Each dict must have a 'label' key.

        Returns
        -------
        np.ndarray
            2D array of shape (n_samples, num_classes).
        """
        labels = [record["label"] for record in data]
        return self.encode(labels)

    def decode_single(self, one_hot_vector):
        """
        Convert a one-hot or probability vector back to an integer label.

        Uses argmax to find the index of the maximum value.

        Parameters
        ----------
        one_hot_vector : np.ndarray
            1D array of shape (num_classes,).

        Returns
        -------
        int
            The predicted class label.
        """
        return int(np.argmax(one_hot_vector))

    def decode(self, one_hot_matrix):
        """
        Convert multiple one-hot or probability vectors to integer labels.

        Parameters
        ----------
        one_hot_matrix : np.ndarray
            2D array of shape (n_samples, num_classes).

        Returns
        -------
        np.ndarray
            1D array of integer labels.
        """
        return np.argmax(one_hot_matrix, axis=1)