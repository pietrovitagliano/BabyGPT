from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer, Dense, ReLU, Dropout, LayerNormalization

from Model.multi_head_attention import MultiHeadAttention


class Decoder(Layer):
    def __init__(self, d_model: int, head_number: int, dropout_rate: float, ffn_hidden_layer_multiplier: int, **kwargs):
        super().__init__(**kwargs)

        self._multi_head_attention = Sequential([
            MultiHeadAttention(d_model=d_model,
                               head_number=head_number,
                               dropout_rate=dropout_rate,
                               masked=True),
            Dropout(dropout_rate)
        ])

        self._feed_forward_network = Sequential([
            Dense(ffn_hidden_layer_multiplier * d_model),
            ReLU(),
            Dense(d_model),
            Dropout(dropout_rate)
        ])

        self._attention_norm_layer = LayerNormalization()
        self._ffn_norm_layer = LayerNormalization()

    def call(self, input_batch, *args, **kwargs):
        # Apply the multi-head attention mechanism using the normalized input_batch and add a residual connection
        ffn_input = input_batch + self._multi_head_attention(self._attention_norm_layer(input_batch))

        # Apply the feed-forward network using the normalized input_batch and add a residual connection
        output = ffn_input + self._feed_forward_network(self._ffn_norm_layer(ffn_input))

        return output
