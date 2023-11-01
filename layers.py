# Custom L1 Distance Layer Module

# Import dependencies
import tensorflow as tf
from tensorflow.keras.layers import Layer

class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init()

    def call(self, input_embedding, validation_embedding):
        """
        Calculate L1 distance between input and validation embeddings.

        Args:
            input_embedding: The embedding to compare.
            validation_embedding: The validation embedding to compare against.

        Returns:
            L1 distance (absolute difference) between the embeddings.
        """
        return tf.math.abs(input_embedding - validation_embedding)
