import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Layer, Dense, Dropout, Softmax


class MultiHeadAttention(Layer):
    def __init__(self, d_model: int, head_number: int, dropout_rate: float, masked: bool = False, **kwargs):
        super().__init__(**kwargs)

        head_size: int = d_model // head_number
        self._head_list: list[HeadAttention] = [HeadAttention(head_size=head_size,
                                                              dropout_rate=dropout_rate,
                                                              masked=masked)
                                                for _ in range(head_number)]
        self._final_linear_layer = Dense(d_model)

    def call(self, input_batch, *args, **kwargs):
        # Process the input_batch using all the head attentions
        head_result_list = [head(input_batch) for head in self._head_list]

        # Concatenate the outputs of the heads
        output = tf.concat(head_result_list, axis=-1)

        # Apply the final linear layer
        output = self._final_linear_layer(output)

        return output


class HeadAttention(Layer):
    def __init__(self, head_size: int, dropout_rate: float, masked: bool, **kwargs):
        super().__init__(**kwargs)

        self._head_size: int = head_size
        self._masked: bool = masked

        self._key_layer = Dense(head_size, use_bias=False)
        self._query_layer = Dense(head_size, use_bias=False)
        self._value_layer = Dense(head_size, use_bias=False)
        self._softmax_layer = Softmax(axis=-1)
        self._dropout = Dropout(dropout_rate)

    def call(self, input_batch, *args, **kwargs):
        # Create query, key and value matrices
        q = self._query_layer(input_batch)
        k = self._key_layer(input_batch)
        v = self._value_layer(input_batch)

        # Compute the transpose of the key matrix, taking into account the rank of the input tensor
        k_t = tf.transpose(k, perm=[0, 2, 1]) if k.shape.rank == 3 else tf.transpose(k)

        # Calculate the attention scores -> Softmax(Q @ K^T / sqrt(d_k))
        # NOTE: d_k (head_size) is expected to be a float32 value
        temp_matrix = q @ k_t / tf.sqrt(tf.cast(self._head_size, dtype=tf.float32))

        # If masked is True, apply the mask in order to prevent the model from attending to future tokens
        if self._masked:
            # Create a mask matrix with the same shape as the attention scores.
            # The mask is a lower triangular matrix with -inf above the diagonal,
            # and 0 in the diagonal and below.
            mask = tf.linalg.band_part(tf.ones_like(temp_matrix), num_lower=-1, num_upper=0)
            mask = tf.where(mask == 0, -np.inf, 0)

            # Apply the mask to the attention scores
            temp_matrix += mask

        # Apply the softmax function to get the attention scores and apply dropout to prevent overfitting
        attention = self._softmax_layer(temp_matrix)
        attention = self._dropout(attention)

        # Return the final output (attention @ V)
        return attention @ v
